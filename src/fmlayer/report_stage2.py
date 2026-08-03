from pathlib import Path

import pandas as pd

from src.fmlayer.data.specs import DATASET_SPECS, get_spec
from src.fmlayer.models.zeroshot import METHOD as ZEROSHOT_METHOD
from src.fmlayer.train.train_flow_clip import METHOD as FLOW_METHOD
from src.fmlayer.utils.results import default_results_root, load_runs
from src.fmlayer.viz.flow_clip import (
    load_curve,
    plot_accuracy_vs_t,
    plot_combined_accuracy_vs_t,
    plot_reverse_retrieval,
    plot_trajectory_embeddings,
)

TABLE_FILENAME = "flow_accuracy_table.csv"


def flow_table(results_root: Path | None = None) -> pd.DataFrame:
    """Aggregate the flow-matching rows of ``runs.csv`` into mean and std per time.

    Args:
        results_root: Results directory; defaults to the resolved results root.

    Returns:
        One row per dataset and integration time, ordered by t.
    """
    runs = [row for row in load_runs(results_root) if row["method"] == FLOW_METHOD]
    if not runs:
        raise FileNotFoundError("No fm_clip rows in runs.csv. Run run_all_flow_clip() first.")

    frame = pd.DataFrame(runs)
    frame["accuracy"] = frame["accuracy"].astype(float)
    frame["t"] = frame["t"].astype(float)

    table = (
        frame.groupby(["dataset", "t"])["accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "runs"})
    )
    table["std"] = table["std"].fillna(0.0)
    return table.sort_values(["dataset", "t"]).reset_index(drop=True)


def save_flow_table(table: pd.DataFrame, results_root: Path | None = None) -> Path:
    """Write the aggregated flow-matching table next to ``runs.csv``.

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


def print_flow_table(table: pd.DataFrame, results_root: Path | None = None) -> None:
    """Print the endpoint comparison and the accuracy at every recorded time.

    Args:
        table: Aggregated results from :func:`flow_table`.
        results_root: Results directory; defaults to the resolved results root.
    """
    zeroshot = {
        row["dataset"]: float(row["accuracy"])
        for row in load_runs(results_root)
        if row["method"] == ZEROSHOT_METHOD
    }

    for dataset, rows in table.groupby("dataset"):
        print(f"\n=== {get_spec(dataset).display_name} ===")
        start = rows[rows["t"] == 0.0]["mean"].item()
        end = rows[rows["t"] == 1.0]["mean"].item()
        spread = rows[rows["t"] == 1.0]["std"].item()

        if dataset in zeroshot:
            print(f"  stage 1 zero-shot     {zeroshot[dataset]:.4f}")
        print(f"  flow layer at t=0     {start:.4f}")
        print(f"  flow layer at t=1     {end:.4f} +/- {spread:.4f}")
        print(f"  absolute gain         {end - start:+.4f}\n")

        for _, row in rows.iterrows():
            print(f"    t={row['t']:.2f}  {row['mean']:.4f} +/- {row['std']:.4f}  (n={row['runs']})")


def make_stage2_report(
    datasets: list[str] | None = None,
    seed: int = 0,
    feature_root: Path | None = None,
    results_root: Path | None = None,
    figures_root: Path | None = None,
    with_retrieval: bool = True,
    show: bool = True,
) -> dict:
    """Produce every Stage 2 deliverable from the recorded flow-matching runs.

    Builds the accuracy-versus-t table and figures, the trajectory snapshots and, when the
    datasets are available in the session, the reverse-flow retrieval panels.

    Args:
        datasets: Dataset keys; defaults to both datasets.
        seed: Run seed whose curves and checkpoints the figures are drawn from.
        feature_root: Feature cache directory; defaults to the resolved feature root.
        results_root: Results directory; defaults to the resolved results root.
        figures_root: Figure directory; defaults to the figures root.
        with_retrieval: Draw the reverse-flow panels, which decode the actual images.
        show: Display the figures instead of closing them.

    Returns:
        The table, its path and the paths of every generated artefact.
    """
    datasets = datasets if datasets is not None else sorted(DATASET_SPECS)

    table = flow_table(results_root)
    table_path = save_flow_table(table, results_root)
    print_flow_table(table, results_root)
    print(f"\nFlow accuracy table -> {table_path}")

    combined = plot_combined_accuracy_vs_t(
        datasets, seed, results_root, figures_root, show
    )

    accuracy_figures = []
    trajectory_figures = []
    retrieval_figures = []
    for dataset in datasets:
        accuracy_figures.append(
            plot_accuracy_vs_t(dataset, seed, results_root, figures_root, show)
        )
        trajectory_figures.append(
            plot_trajectory_embeddings(
                dataset,
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
        "combined_figure": combined,
        "accuracy_figures": accuracy_figures,
        "trajectory_figures": trajectory_figures,
        "retrieval_figures": retrieval_figures,
        "curves": {dataset: load_curve(dataset, seed, results_root) for dataset in datasets},
    }
