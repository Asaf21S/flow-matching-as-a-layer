from pathlib import Path

import pandas as pd

from src.fmlayer.data.fewshot import K_FULL
from src.fmlayer.data.specs import DATASET_SPECS, get_spec
from src.fmlayer.models.zeroshot import METHOD as ZEROSHOT_METHOD
from src.fmlayer.train.train_flow_clip import (
    METHODS,
    ROLLED,
    STANDARD,
    STEP_COUNTS,
    zeroshot_accuracy,
)
from src.fmlayer.utils.results import default_results_root, load_runs
from src.fmlayer.viz.flow_clip import (
    VARIANT_LABEL,
    plot_accuracy_vs_k,
    plot_feature_comparison,
    plot_flow_trajectories,
    plot_reverse_retrieval,
    plot_training_curves,
)

TABLE_FILENAME = "flow_accuracy_table.csv"
K_SORT_ORDER = {"5": 0, "10": 1, K_FULL: 2}


def baseline_accuracies(
    runs: list[dict], datasets: list[str], feature_root: Path | None = None
) -> dict:
    """Resolve the Stage 1 prototype baseline of each dataset.

    Prefers the recorded zero-shot row so the table quotes exactly the Stage 1 number, and
    falls back to scoring the cached CLIP features when Stage 1 was never run in this
    results directory, which keeps the Stage 2 notebook standalone.

    Args:
        runs: Every row of ``runs.csv``.
        datasets: Dataset keys a baseline is needed for.
        feature_root: Feature cache directory; defaults to the resolved feature root.

    Returns:
        Baseline accuracy keyed by dataset.
    """
    recorded = {
        row["dataset"]: float(row["accuracy"])
        for row in runs
        if row["method"] == ZEROSHOT_METHOD
    }
    return {
        dataset: recorded[dataset]
        if dataset in recorded
        else zeroshot_accuracy(dataset, feature_root)
        for dataset in datasets
    }


def flow_table(
    results_root: Path | None = None, feature_root: Path | None = None
) -> pd.DataFrame:
    """Aggregate the Stage 2 rows of ``runs.csv`` into mean, std and delta per setting.

    The Stage 1 prototype baseline does not depend on K for the CLIP branch, so it enters as
    a single constant per dataset and the delta is measured against it.

    Args:
        results_root: Results directory; defaults to the resolved results root.
        feature_root: Feature cache directory, used only when the baseline is not recorded.

    Returns:
        One row per dataset, method, step count and K, ordered by the protocol.
    """
    runs = load_runs(results_root)
    flow_methods = set(METHODS.values())
    flow_rows = [row for row in runs if row["method"] in flow_methods]
    if not flow_rows:
        raise FileNotFoundError(
            "No flow-matching rows in runs.csv. Run run_all_flow_clip() first."
        )

    frame = pd.DataFrame(flow_rows)
    frame["accuracy"] = frame["accuracy"].astype(float)
    frame["steps"] = frame["steps"].astype(int)
    baselines = baseline_accuracies(runs, sorted(frame["dataset"].unique()), feature_root)

    table = (
        frame.groupby(["dataset", "method", "steps", "k"])["accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "runs"})
    )
    table["std"] = table["std"].fillna(0.0)
    table["baseline"] = table["dataset"].map(baselines)
    table["delta"] = table["mean"] - table["baseline"]

    table["k_order"] = table["k"].astype(str).map(K_SORT_ORDER).fillna(len(K_SORT_ORDER))
    table = table.sort_values(["dataset", "method", "steps", "k_order"])
    return table.drop(columns="k_order").reset_index(drop=True)


def save_flow_table(table: pd.DataFrame, results_root: Path | None = None) -> Path:
    """Write the aggregated Stage 2 table next to ``runs.csv``.

    Args:
        table: Aggregated results from :func:`flow_table`.
        results_root: Results directory; defaults to the resolved results root.

    Returns:
        Path of the written CSV.
    """
    root = Path(results_root) if results_root is not None else default_results_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / TABLE_FILENAME
    table.to_csv(path, index=False)
    return path


def print_flow_table(table: pd.DataFrame) -> None:
    """Print accuracy and delta-versus-baseline for every setting.

    Args:
        table: Aggregated results from :func:`flow_table`.
    """
    labels = {
        METHODS[STANDARD]: VARIANT_LABEL[STANDARD],
        METHODS[ROLLED]: VARIANT_LABEL[ROLLED],
    }

    for dataset, rows in table.groupby("dataset"):
        baseline = float(rows["baseline"].iloc[0])
        print(f"\n=== {get_spec(dataset).display_name} ===")
        print(f"  Stage 1 prototype baseline: {baseline:.4f}\n")
        print(
            f"  {'variant':<20} {'T':>3} {'K':>5}   {'acc':>8}  {'std':>7}  "
            f"{'delta':>8}   n"
        )

        for _, row in rows.iterrows():
            name = labels.get(row["method"], row["method"])
            print(
                f"  {name:<20} {row['steps']:>3} {str(row['k']):>5}   "
                f"{row['mean']:.4f}  {row['std']:.4f}  {row['delta']:+.4f}   {row['runs']}"
            )


def make_stage2_report(
    datasets: list[str] | None = None,
    k: int | str = K_FULL,
    seed: int = 0,
    steps: int = STEP_COUNTS[-1],
    feature_root: Path | None = None,
    results_root: Path | None = None,
    figures_root: Path | None = None,
    with_retrieval: bool = False,
    show: bool = True,
) -> dict:
    """Produce every Stage 2 deliverable from the recorded runs.

    Args:
        datasets: Dataset keys; defaults to both datasets.
        k: Training-set size of the representative run used for the figures.
        seed: Run seed used for the figures.
        steps: Euler step count used for the feature and trajectory figures.
        feature_root: Feature cache directory; defaults to the resolved feature root.
        results_root: Results directory; defaults to the resolved results root.
        figures_root: Figure directory; defaults to the figures root.
        with_retrieval: Also draw the optional reverse-flow panels, which decode images.
        show: Display the figures instead of closing them.

    Returns:
        The table, its path and the paths of every generated figure.
    """
    datasets = datasets if datasets is not None else sorted(DATASET_SPECS)

    table = flow_table(results_root, feature_root)
    table_path = save_flow_table(table, results_root)
    print_flow_table(table)
    print(f"\nFlow accuracy table -> {table_path}")

    accuracy_figure = plot_accuracy_vs_k(table, datasets, figures_root, show)

    curve_figures = []
    comparison_figures = []
    trajectory_figures = []
    retrieval_figures = []
    for dataset in datasets:
        curve_figures.append(
            plot_training_curves(
                dataset, k, seed, STEP_COUNTS, results_root, figures_root, show
            )
        )
        comparison_figures.append(
            plot_feature_comparison(
                dataset,
                steps,
                k,
                seed,
                feature_root=feature_root,
                results_root=results_root,
                figures_root=figures_root,
                show=show,
            )
        )
        trajectory_figures.append(
            plot_flow_trajectories(
                dataset,
                ROLLED,
                steps,
                k,
                seed,
                feature_root=feature_root,
                results_root=results_root,
                figures_root=figures_root,
                show=show,
            )
        )
        if with_retrieval:
            retrieval_figures.append(
                plot_reverse_retrieval(
                    dataset,
                    ROLLED,
                    steps,
                    k,
                    seed,
                    feature_root=feature_root,
                    results_root=results_root,
                    figures_root=figures_root,
                    show=show,
                )
            )

    return {
        "table": table,
        "table_path": table_path,
        "accuracy_figure": accuracy_figure,
        "curve_figures": curve_figures,
        "comparison_figures": comparison_figures,
        "trajectory_figures": trajectory_figures,
        "retrieval_figures": retrieval_figures,
    }

