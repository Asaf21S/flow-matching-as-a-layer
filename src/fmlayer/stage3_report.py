from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.fmlayer.data.fewshot import K_FULL, load_train_subset
from src.fmlayer.data.specs import DATASET_SPECS, get_spec
from src.fmlayer.encoders.registry import LINEAR_PROBE_CELLS
from src.fmlayer.models.probe_bank import ProbeBank
from src.fmlayer.models.targets import ClassTargets, build_target_provider
from src.fmlayer.train.diagnostics import diagnose_all, print_diagnostics
from src.fmlayer.train.train_linear import to_tensors
from src.fmlayer.utils.results import default_results_root
from src.fmlayer.viz.stage3_charts import plot_accuracy_vs_k, plot_config_ablation

TABLE_FILENAME = "stage3_table.csv"
GROUP_COLUMNS = ("encoder", "dataset", "k", "config_name", "objective", "target_type", "steps")
K_SORT_ORDER = {"5": 0, "10": 1, K_FULL: 2}


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
    table = (
        frame.groupby(list(GROUP_COLUMNS))
        .agg(
            acc_mean=("accuracy", "mean"),
            acc_std=("accuracy", "std"),
            baseline_mean=("baseline", "mean"),
            runs=("accuracy", "size"),
        )
        .reset_index()
    )
    table["acc_std"] = table["acc_std"].fillna(0.0)
    table["delta"] = table["acc_mean"] - table["baseline_mean"]

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
        print(f"  {'configuration':<40} {'T':>3}  {'accuracy':>17}  {'delta':>8}")
        print(f"  {'frozen linear probe (baseline)':<40} {'-':>3}  {baseline:>9.4f}          {0.0:>+8.4f}")
        for _, row in group.sort_values("delta", ascending=False).iterrows():
            print(
                f"  {row['config_name']:<40} {int(row['steps']):>3}  "
                f"{row['acc_mean']:>9.4f} +/- {row['acc_std']:.4f}  {row['delta']:>+8.4f}"
            )


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
