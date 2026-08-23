from pathlib import Path

import pandas as pd

from src.fmlayer.data.specs import DATASET_SPECS
from src.fmlayer.utils.results import default_results_root, load_runs
from src.fmlayer.viz.accuracy import plot_accuracy_vs_k_all
from src.fmlayer.viz.confusion import plot_confusion_for_probe, plot_confusion_for_zeroshot
from src.fmlayer.viz.curves import plot_representative_curves
from src.fmlayer.viz.embeddings import plot_embeddings_for_dataset

TABLE_FILENAME = "accuracy_table.csv"
# ``steps`` and ``t`` are constant for every stage 1 method, so they only split rows for
# later stages that sweep the Euler step count or the integration time.
GROUP_COLUMNS = ("method", "dataset", "encoder", "k", "steps", "t")
# Training-set size is a protocol order, not an alphabetical one.
K_SORT_ORDER = {"5": 0, "10": 1, "full": 2, "none": 3}


def accuracy_table(results_root: Path | None = None) -> pd.DataFrame:
    """Aggregate ``runs.csv`` into mean and standard deviation per setting.

    Args:
        results_root: Results directory; defaults to the resolved results root.

    Returns:
        One row per method/dataset/encoder/K with mean accuracy, standard deviation
        and the number of runs behind it, ordered K = 5, 10, full.
    """
    runs = load_runs(results_root)
    if not runs:
        raise FileNotFoundError("runs.csv is empty. Run the baselines first.")

    frame = pd.DataFrame(runs)
    frame["accuracy"] = frame["accuracy"].astype(float)

    table = (
        frame.groupby(list(GROUP_COLUMNS))["accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "runs"})
    )
    # A single run has no spread; report 0 rather than NaN.
    table["std"] = table["std"].fillna(0.0)

    table["k_order"] = table["k"].astype(str).map(K_SORT_ORDER).fillna(len(K_SORT_ORDER))
    table = table.sort_values(["method", "dataset", "encoder", "k_order"])
    return table.drop(columns="k_order").reset_index(drop=True)



def save_accuracy_table(
    table: pd.DataFrame, results_root: Path | None = None
) -> Path:
    """Write the aggregated accuracy table next to ``runs.csv``.

    Args:
        table: Aggregated results from :func:`accuracy_table`.
        results_root: Results directory; defaults to the resolved results root.

    Returns:
        Path of the written CSV.
    """
    root = Path(results_root) if results_root is not None else default_results_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / TABLE_FILENAME
    table.to_csv(path, index=False)
    return path


def print_accuracy_table(table: pd.DataFrame) -> None:
    """Print the accuracy table as ``mean +/- std`` per setting.

    Args:
        table: Aggregated results from :func:`accuracy_table`.
    """
    for dataset, rows in table.groupby("dataset"):
        print(f"\n=== {DATASET_SPECS[dataset].display_name} ===")
        for _, row in rows.iterrows():
            print(
                f"  {row['method']:<14} {row['encoder']:<14} K={str(row['k']):<5} "
                f"{row['mean']:.4f} +/- {row['std']:.4f}  (n={row['runs']})"
            )


def make_report(
    datasets: list[str] | None = None,
    confusion_k: int | str = "full",
    embedding_method: str = "pca",
    feature_root: Path | None = None,
    results_root: Path | None = None,
    figures_root: Path | None = None,
    show: bool = True,
) -> dict:
    """Produce every Stage 1 deliverable from the recorded runs.

    Builds the accuracy table, the accuracy-versus-K plots, the representative training
    curves, one confusion matrix per dataset and the joint feature visualisations.

    Args:
        datasets: Dataset keys; defaults to both Stage 1 datasets.
        confusion_k: Training-set size of the linear probe used for the confusion matrices.
        embedding_method: ``"pca"`` or ``"tsne"``.
        feature_root: Feature cache directory; defaults to the resolved feature root.
        results_root: Results directory; defaults to the resolved results root.
        figures_root: Figure directory; defaults to the figures root.
        show: Display the figures instead of closing them.

    Returns:
        The accuracy table and the paths of every generated artefact.
    """
    datasets = datasets if datasets is not None else sorted(DATASET_SPECS)

    table = accuracy_table(results_root)
    table_path = save_accuracy_table(table, results_root)
    print_accuracy_table(table)
    print(f"\nAccuracy table -> {table_path}")

    accuracy_figures = plot_accuracy_vs_k_all(table, datasets, figures_root, show)
    curve_figures = plot_representative_curves(
        results_root=results_root, figures_root=figures_root, show=show
    )

    confusion_figures = []
    embedding_figures = []
    for dataset in datasets:
        confusion_figures.append(
            plot_confusion_for_zeroshot(dataset, feature_root, figures_root, show)
        )
        confusion_figures.append(
            plot_confusion_for_probe(
                "resnet18",
                dataset,
                confusion_k,
                feature_root=feature_root,
                results_root=results_root,
                figures_root=figures_root,
                show=show,
            )
        )
        embedding_figures.extend(
            plot_embeddings_for_dataset(
                dataset,
                method=embedding_method,
                feature_root=feature_root,
                figures_root=figures_root,
                show=show,
            )
        )

    return {
        "table": table,
        "table_path": table_path,
        "accuracy_figures": accuracy_figures,
        "curve_figures": curve_figures,
        "confusion_figures": confusion_figures,
        "embedding_figures": embedding_figures,
    }
