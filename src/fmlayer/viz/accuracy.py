from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.fmlayer.data.fewshot import K_FULL
from src.fmlayer.data.specs import get_spec
from src.fmlayer.models.zeroshot import METHOD as ZEROSHOT_METHOD
from src.fmlayer.train.train_linear import METHOD as PROBE_METHOD
from src.fmlayer.viz.figures import save_figure

K_ORDER = ("5", "10", K_FULL)
ZEROSHOT_COLOR = "grey"


def k_position(k: str) -> float:
    """Map a training-set size onto the x axis.

    Args:
        k: ``"5"``, ``"10"`` or ``"full"``.

    Returns:
        The x coordinate of that setting.
    """
    return float(K_ORDER.index(k))


def plot_accuracy_vs_k(
    dataset: str,
    table: pd.DataFrame,
    figures_root: Path | None = None,
    show: bool = True,
) -> Path:
    """Plot top-1 accuracy against training-set size, with error bars.

    Each encoder gets a line over K, and the zero-shot CLIP result is drawn as a
    horizontal reference line since it uses no labelled training images.

    Args:
        dataset: Dataset key.
        table: Aggregated results as returned by the report module.
        figures_root: Figure directory; defaults to the figures root.
        show: Display the figure instead of closing it.

    Returns:
        Path of the written figure.
    """
    subset = table[table["dataset"] == dataset]
    fig, ax = plt.subplots(figsize=(7, 5))

    probes = subset[subset["method"] == PROBE_METHOD]
    for encoder, rows in probes.groupby("encoder"):
        rows = rows.copy()
        rows["position"] = rows["k"].map(k_position)
        rows = rows.sort_values("position")
        ax.errorbar(
            rows["position"],
            rows["mean"],
            yerr=rows["std"],
            marker="o",
            capsize=4,
            label=f"linear probe ({encoder})",
        )

    zeroshot = subset[subset["method"] == ZEROSHOT_METHOD]
    for _, row in zeroshot.iterrows():
        ax.axhline(
            row["mean"],
            color=ZEROSHOT_COLOR,
            linestyle="--",
            label=f"zero-shot CLIP RN50 ({row['mean']:.3f})",
        )

    ax.set_xticks(range(len(K_ORDER)), [str(k) for k in K_ORDER])
    ax.set_xlabel("training images per class (K)")
    ax.set_ylabel("top-1 accuracy on the test split")
    ax.set_title(f"{get_spec(dataset).display_name}: accuracy vs training-set size")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()

    return save_figure(fig, f"accuracy_vs_k_{dataset}", figures_root, show)


def plot_accuracy_vs_k_all(
    table: pd.DataFrame,
    datasets: list[str] | None = None,
    figures_root: Path | None = None,
    show: bool = True,
) -> list[Path]:
    """Draw the accuracy-versus-K figure for several datasets.

    Args:
        table: Aggregated results as returned by the report module.
        datasets: Dataset keys; defaults to those present in the table.
        figures_root: Figure directory; defaults to the figures root.
        show: Display the figures instead of closing them.

    Returns:
        Paths of the written figures.
    """
    datasets = datasets if datasets is not None else sorted(np.unique(table["dataset"]))
    return [plot_accuracy_vs_k(dataset, table, figures_root, show) for dataset in datasets]
