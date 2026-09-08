from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.fmlayer.data.fewshot import K_FULL, load_train_subset
from src.fmlayer.data.specs import DATASET_SPECS, get_spec
from src.fmlayer.encoders.registry import LINEAR_PROBE_CELLS
from src.fmlayer.features.cache import load_split
from src.fmlayer.models.probe_bank import ProbeBank
from src.fmlayer.models.targets import ClassTargets, build_target_provider
from src.fmlayer.train.diagnostics import diagnose_all, print_diagnostics
from src.fmlayer.train.train_fm import MAIN_K, MAIN_STEPS, main_configs
from src.fmlayer.train.train_linear import to_tensors
from src.fmlayer.utils.results import default_results_root
from src.fmlayer.viz.flow_viz import plot_feature_comparison, plot_flow_dynamics
from src.fmlayer.viz.stage3_charts import (
    plot_accuracy_vs_k,
    plot_config_ablation,
    plot_curve_comparison,
)

TABLE_FILENAME = "stage3_table.csv"
MAIN_TABLE_FILENAME = "stage3_main.csv"
GROUP_COLUMNS = ("encoder", "dataset", "k", "config_name", "objective", "target_type", "steps")
K_SORT_ORDER = {"5": 0, "10": 1, K_FULL: 2}
BASELINE_LABEL = "Stage 1 linear probe (frozen)"

MAIN_METHOD_LABELS = (
    "End-to-end rolled-out (Strategy 1)",
    "Classifier-guided FM (Strategy 2)",
    "Classifier-guided FM, noised sources",
    "Joint fine-tuning (extension)",
)


def main_labels(train_steps: int = MAIN_STEPS) -> dict[str, str]:
    """Map each main configuration's tag onto the name used in the write-up.

    Derived from :func:`main_configs` so a renamed configuration can never fall out of
    the table silently.
    """
    names = [config.name for config in main_configs(train_steps)]
    return dict(zip(names, MAIN_METHOD_LABELS))


def main_comparison_table(
    results: dict,
    k: int | str = MAIN_K,
    steps: int = MAIN_STEPS,
    labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build the headline table: the frozen probe against each Stage 3 method.

    The delta is paired per seed, because every run is scored against the probe of its
    own seed, so its spread is the right error bar rather than the spread of accuracy.

    Args:
        results: Output of :func:`run_stage3_main` or :func:`load_stage3_results`.
        k: Training-set size to report.
        steps: The single T Stage 3 fixed.
        labels: Tag to display-name mapping; defaults to :func:`main_labels`.

    Returns:
        One row per cell and method, with the baseline as the first row of each cell.
    """
    labels = labels if labels is not None else main_labels(steps)
    records = []
    for result in results.values():
        if str(result["k"]) != str(k) or steps not in result["accuracy_by_steps"]:
            continue
        records.append(
            {
                "dataset": result["dataset"],
                "encoder": result["encoder"],
                "config_name": result["config_name"],
                "seed": result["seed"],
                "accuracy": result["accuracy_by_steps"][steps],
                "baseline": result["baseline_accuracy"],
            }
        )
    if not records:
        raise ValueError(f"No Stage 3 results at K={k}, T={steps} to aggregate.")

    frame = pd.DataFrame(records)
    frame["run_delta"] = frame["accuracy"] - frame["baseline"]

    rows = []
    for (dataset, encoder), group in frame.groupby(["dataset", "encoder"], sort=True):
        baseline = group.groupby("seed")["baseline"].first()
        rows.append(
            {
                "dataset": dataset,
                "encoder": encoder,
                "method": BASELINE_LABEL,
                "config_name": "",
                "steps": "-",
                "seeds": int(len(baseline)),
                "acc_mean": float(baseline.mean()),
                "acc_std": float(baseline.std(ddof=1)) if len(baseline) > 1 else 0.0,
                "delta": float("nan"),
                "delta_std": float("nan"),
                "significant": False,
            }
        )
        for name, label in labels.items():
            subset = group[group["config_name"] == name]
            if subset.empty:
                continue
            spread = float(subset["run_delta"].std(ddof=1)) if len(subset) > 1 else 0.0
            delta = float(subset["run_delta"].mean())
            rows.append(
                {
                    "dataset": dataset,
                    "encoder": encoder,
                    "method": label,
                    "config_name": name,
                    "steps": steps,
                    "seeds": int(len(subset)),
                    "acc_mean": float(subset["accuracy"].mean()),
                    "acc_std": float(subset["accuracy"].std(ddof=1)) if len(subset) > 1 else 0.0,
                    "delta": delta,
                    "delta_std": spread,
                    "significant": bool(abs(delta) > 2 * spread > 0),
                }
            )
    return pd.DataFrame(rows)


def markdown_main_table(table: pd.DataFrame) -> str:
    """Render :func:`main_comparison_table` as a markdown table for the write-up.

    Args:
        table: Output of :func:`main_comparison_table`.

    Returns:
        A markdown string, ready to paste into ``docs/stage3.md``.
    """
    lines = [
        "| Dataset | Encoder | Method | Top-1 accuracy | Delta vs probe |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in table.iterrows():
        display = get_spec(row["dataset"]).display_name
        accuracy = f"{row['acc_mean']:.4f}"
        if row["seeds"] > 1:
            accuracy += f" +/- {row['acc_std']:.4f}"
        if pd.isna(row["delta"]):
            delta = "-"
        else:
            delta = f"{row['delta']:+.4f}"
            if row["seeds"] > 1:
                delta += f" +/- {row['delta_std']:.4f}"
            if row["significant"]:
                delta += " *"
        lines.append(
            f"| {display} | {row['encoder']} | {row['method']} | {accuracy} | {delta} |"
        )
    lines.append("")
    lines.append("`*` marks a delta larger than twice its paired standard deviation across seeds.")
    return "\n".join(lines)


def save_main_table(table: pd.DataFrame, results_root: Path | None = None) -> Path:
    """Write the headline table, baselines included, next to ``runs.csv``."""
    root = Path(results_root) if results_root is not None else default_results_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / MAIN_TABLE_FILENAME
    table.to_csv(path, index=False)
    return path


def stage3_table(results: dict) -> pd.DataFrame:
    """Aggregate the Stage 3 run grid over seeds.

    Args:
        results: Output of :func:`run_all_stage3`.

    Returns:
        One row per configuration, cell, K and T, with mean and std over seeds.
    """
    records = []
    for result in results.values():
        for steps, accuracy in result["accuracy_by_steps"].items():
            records.append(
                {
                    "encoder": result["encoder"],
                    "dataset": result["dataset"],
                    "k": str(result["k"]),
                    "config_name": result["config_name"],
                    "objective": result["objective"],
                    "target_type": result["target_type"],
                    "steps": int(steps),
                    "seed": result["seed"],
                    "accuracy": accuracy,
                    "baseline": result["baseline_accuracy"],
                }
            )
    if not records:
        raise ValueError("No Stage 3 results to aggregate.")

    frame = pd.DataFrame(records)
    # Each run is paired with the probe of its own seed, so the delta is a paired
    # difference and its own spread is the right error bar, not the spread of accuracy.
    frame["run_delta"] = frame["accuracy"] - frame["baseline"]
    table = (
        frame.groupby(list(GROUP_COLUMNS))
        .agg(
            acc_mean=("accuracy", "mean"),
            acc_std=("accuracy", "std"),
            baseline_mean=("baseline", "mean"),
            delta=("run_delta", "mean"),
            delta_std=("run_delta", "std"),
            runs=("accuracy", "size"),
        )
        .reset_index()
    )
    table["acc_std"] = table["acc_std"].fillna(0.0)
    table["delta_std"] = table["delta_std"].fillna(0.0)

    table["k_order"] = table["k"].map(K_SORT_ORDER).fillna(len(K_SORT_ORDER))
    table = table.sort_values(["dataset", "encoder", "k_order", "delta"], ascending=[True, True, True, False])
    return table.drop(columns="k_order").reset_index(drop=True)


def save_stage3_table(table: pd.DataFrame, results_root: Path | None = None) -> Path:
    """Write the aggregated Stage 3 table next to ``runs.csv``."""
    root = Path(results_root) if results_root is not None else default_results_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / TABLE_FILENAME
    table.to_csv(path, index=False)
    return path


def leaderboard(table: pd.DataFrame, k: int | str = K_FULL, top: int = 10) -> pd.DataFrame:
    """Rank configurations by their mean gain over the frozen probe.

    Args:
        table: Aggregated table from :func:`stage3_table`.
        k: Training-set size to rank at.
        top: Number of rows to keep per cell.

    Returns:
        The best rows per encoder/dataset cell, sorted by delta.
    """
    rows = table[table["k"].astype(str) == str(k)]
    ranked = rows.sort_values("delta", ascending=False)
    return ranked.groupby(["dataset", "encoder"], as_index=False, group_keys=False).head(top)


def print_stage3_table(table: pd.DataFrame, k: int | str | None = None) -> None:
    """Print the aggregated table grouped by dataset and encoder.

    Args:
        table: Aggregated table from :func:`stage3_table`.
        k: Only print this training-set size; ``None`` prints every size.
    """
    rows = table if k is None else table[table["k"].astype(str) == str(k)]
    for (dataset, encoder, size), group in rows.groupby(["dataset", "encoder", "k"], sort=False):
        baseline = float(group["baseline_mean"].mean())
        print(f"\n=== {get_spec(dataset).display_name} | {encoder} | K = {size} ===")
        print(f"  {'configuration':<34} {'T':>3}  {'accuracy':>17}  {'delta (paired)':>20}")
        print(f"  {'frozen linear probe (baseline)':<34} {'-':>3}  {baseline:>9.4f}          {'-':>20}")
        for _, row in group.sort_values("delta", ascending=False).iterrows():
            # Flag a gain only when it clears two paired standard deviations.
            significant = "*" if row["delta"] > 2 * row["delta_std"] else " "
            print(
                f"  {row['config_name']:<34} {int(row['steps']):>3}  "
                f"{row['acc_mean']:>9.4f} +/- {row['acc_std']:.4f}  "
                f"{row['delta']:>+9.4f} +/- {row['delta_std']:.4f} {significant}"
            )
    print("\n* delta exceeds twice its paired standard deviation across seeds")


def target_table_for_run(
    result: dict,
    feature_root: Path | None = None,
    subset_root: Path | None = None,
) -> np.ndarray | None:
    """Rebuild the per-class target table a run was trained against, for the figures.

    Args:
        result: One entry returned by :func:`run_stage3`.
        feature_root: Feature cache directory.
        subset_root: Subset index directory.

    Returns:
        A ``(num_classes, dim)`` array, or ``None`` for per-sample or absent targets.
    """
    config = result["config"]
    if not config.uses_targets:
        return None

    probe = result["classifier"]
    device = probe.weight.device
    features, labels, _ = load_train_subset(
        result["encoder"], result["dataset"], result["k"], result["seed"], feature_root, subset_root
    )
    train_x, train_y = to_tensors(features, labels, device)
    provider = build_target_provider(
        config.target_type,
        train_x,
        train_y,
        ProbeBank([probe]),
        probe,
        get_spec(result["dataset"]).num_classes,
        config.margin_ratio,
    )
    if not isinstance(provider, ClassTargets):
        return None
    with torch.no_grad():
        return provider.table.cpu().numpy()


def make_stage3_main_figures(
    results: dict,
    k: int | str = MAIN_K,
    seed: int = 0,
    steps: int = MAIN_STEPS,
    labels: dict[str, str] | None = None,
    include_joint: bool = False,
    feature_root: Path | None = None,
    figures_root: Path | None = None,
    show: bool = False,
    save: bool = True,
) -> dict:
    """Render exactly the three deliverables the brief asks for, per cell.

    Unlike :func:`make_stage3_figures`, nothing here is chosen by leaderboard position:
    the methods drawn are the ones the comparison is defined over.

    Args:
        results: Output of :func:`run_stage3_main`.
        k: Training-set size to visualise.
        seed: Seed to visualise.
        steps: The single T Stage 3 fixed; trajectories are drawn at this T only.
        labels: Tag to display-name mapping; defaults to :func:`main_labels`.
        include_joint: Also put the jointly fine-tuned flow in the before/after figure.
        feature_root: Feature cache directory.
        figures_root: Figure directory.
        show: Display the figures.
        save: Write PNGs.

    Returns:
        The paths of every figure written, grouped by kind.
    """
    labels = labels if labels is not None else main_labels(steps)
    joint_names = {config.name for config in main_configs(steps) if config.joint_finetune}

    cells = sorted({(result["encoder"], result["dataset"]) for result in results.values()})
    curves_paths, dynamics_paths, comparison_paths = [], [], []

    for encoder, dataset in cells:
        available = {}
        for name, label in labels.items():
            result = results.get(f"{name}/{encoder}/{dataset}/{k}/{seed}")
            if result is not None:
                available[label] = result
        if not available:
            continue

        val_features, val_labels, metadata = load_split(encoder, dataset, "val", feature_root)
        class_names = metadata["class_names"]
        baseline = next(iter(available.values()))["baseline_accuracy"]

        curves_paths.append(
            plot_curve_comparison(
                available, dataset, encoder, baseline,
                figures_root=figures_root, show=show, save=save,
            )
        )

        for result in available.values():
            dynamics_paths.append(
                plot_flow_dynamics(
                    result, val_features, val_labels, class_names,
                    targets=target_table_for_run(result, feature_root),
                    step_counts=(steps,), figures_root=figures_root, show=show, save=save,
                )
            )

        fields = {
            f"After {label}": result["fm_layer"]
            for label, result in available.items()
            if include_joint or result["config_name"] not in joint_names
        }
        test_features, test_labels, _ = load_split(encoder, dataset, "test", feature_root)
        comparison_paths.append(
            plot_feature_comparison(
                fields, test_features, test_labels, class_names, dataset, encoder,
                steps=steps, figures_root=figures_root, show=show, save=save,
            )
        )

    written = [path for path in curves_paths + dynamics_paths + comparison_paths if path is not None]
    print(f"\n{len(written)} figure(s) written:")
    for path in written:
        print(f"  {path}")
    return {
        "curve_figures": curves_paths,
        "dynamics_figures": dynamics_paths,
        "comparison_figures": comparison_paths,
    }


def make_stage3_figures(
    results: dict,
    k: int | str = K_FULL,
    seed: int = 0,
    controls: tuple[str, ...] = ("standard_centroids",),
    steps: int = 12,
    dynamics_steps: tuple[int, ...] | None = None,
    feature_root: Path | None = None,
    figures_root: Path | None = None,
    show: bool = False,
    save: bool = True,
) -> dict:
    """Render the per-run figures that :func:`make_stage3_report` does not produce.

    For every encoder/dataset cell this draws the learned dynamics of the best
    configuration and of each control, then one before/after comparison putting those
    two side by side against the untransported features.

    Args:
        results: Output of :func:`run_all_stage3`.
        k: Training-set size to visualise.
        seed: Seed to visualise.
        controls: Configurations always drawn alongside the winner, for contrast.
        steps: Euler steps used in the before/after comparison.
        dynamics_steps: Trajectory step counts; defaults to ``steps`` alone.
        feature_root: Feature cache directory.
        figures_root: Figure directory; defaults to ``<results>/figures``.
        show: Display the figures.
        save: Write PNGs.

    Returns:
        The chosen configuration per cell and the paths of every figure written.
    """
    table = stage3_table(results)
    at_k = table[table["k"].astype(str) == str(k)]
    dynamics_steps = dynamics_steps if dynamics_steps is not None else (steps,)
    dynamics_paths = []
    comparison_paths = []
    chosen: dict[str, list[str]] = {}

    for (dataset, encoder), group in at_k.groupby(["dataset", "encoder"]):
        best = group.sort_values("delta", ascending=False).iloc[0]["config_name"]
        names = [best] + [name for name in controls if name != best]
        chosen[f"{encoder}/{dataset}"] = names

        val_features, val_labels, metadata = load_split(encoder, dataset, "val", feature_root)
        class_names = metadata["class_names"]

        fields = {}
        for name in names:
            result = results.get(f"{name}/{encoder}/{dataset}/{k}/{seed}")
            if result is None:
                continue
            targets = target_table_for_run(result, feature_root)
            dynamics_paths.append(
                plot_flow_dynamics(
                    result, val_features, val_labels, class_names, targets=targets,
                    step_counts=dynamics_steps, figures_root=figures_root, show=show, save=save,
                )
            )
            fields[f"After {name}"] = result["fm_layer"]

        if not fields:
            continue

        test_features, test_labels, _ = load_split(encoder, dataset, "test", feature_root)
        reference = results[f"{names[0]}/{encoder}/{dataset}/{k}/{seed}"]
        comparison_paths.append(
            plot_feature_comparison(
                fields, test_features, test_labels, class_names, dataset, encoder,
                targets=target_table_for_run(reference, feature_root), steps=steps,
                figures_root=figures_root, show=show, save=save,
            )
        )

    written = [path for path in dynamics_paths + comparison_paths if path is not None]
    print(f"\n{len(written)} figure(s) written:")
    for path in written:
        print(f"  {path}")
    return {
        "chosen": chosen,
        "dynamics_figures": dynamics_paths,
        "comparison_figures": comparison_paths,
    }


def make_stage3_report(
    results: dict,
    cells: tuple[tuple[str, str], ...] | None = None,
    ablation_k: int | str = K_FULL,
    feature_root: Path | None = None,
    results_root: Path | None = None,
    figures_root: Path | None = None,
    show: bool = True,
    save: bool = False,
    diagnose: bool = True,
) -> dict:
    """Build the Stage 3 table, leaderboard, diagnostics and per-cell figures.

    Args:
        results: Output of :func:`run_all_stage3`.
        cells: ``(encoder, dataset)`` pairs; defaults to the Stage 1 probe cells.
        ablation_k: Training-set size charted in the ablation bars.
        feature_root: Feature cache directory, used by the diagnostics.
        results_root: Results directory.
        figures_root: Figure directory.
        show: Display the figures.
        save: Write PNGs.
        diagnose: Also measure displacement and label flips per run.

    Returns:
        The table, its path, the leaderboard, the diagnostics and the figure paths.
    """
    cells = cells if cells is not None else LINEAR_PROBE_CELLS
    table = stage3_table(results)
    table_path = save_stage3_table(table, results_root)

    print_stage3_table(table, k=ablation_k)
    print(f"\nStage 3 table -> {table_path}")

    diagnostics = None
    if diagnose:
        diagnostics = diagnose_all(results, feature_root=feature_root)
        print()
        print_diagnostics(diagnostics)

    ablation_figures = []
    accuracy_figures = []
    for encoder, dataset in cells:
        if dataset not in DATASET_SPECS:
            continue
        ablation_figures.append(
            plot_config_ablation(table, encoder, dataset, ablation_k, figures_root, show, save)
        )
        accuracy_figures.append(
            plot_accuracy_vs_k(table, encoder, dataset, figures_root=figures_root, show=show, save=save)
        )

    return {
        "table": table,
        "table_path": table_path,
        "leaderboard": leaderboard(table, ablation_k),
        "diagnostics": diagnostics,
        "ablation_figures": ablation_figures,
        "accuracy_figures": accuracy_figures,
    }
