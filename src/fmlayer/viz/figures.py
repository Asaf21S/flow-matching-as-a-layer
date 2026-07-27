from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.fmlayer.utils.results import default_results_root

FIGURES_DIRNAME = "figures"
FIGURE_DPI = 150


def default_figures_root(results_root: Path | None = None) -> Path:
    """Resolve the directory figures are written to.

    Args:
        results_root: Results directory; defaults to the resolved results root.

    Returns:
        The ``figures`` directory inside the results root.
    """
    root = Path(results_root) if results_root is not None else default_results_root()
    return root / FIGURES_DIRNAME


def save_figure(
    fig: Figure, name: str, figures_root: Path | None = None, show: bool = True
) -> Path:
    """Write a figure to disk and optionally display it.

    Args:
        fig: The figure to save.
        name: File name without extension.
        figures_root: Output directory; defaults to :func:`default_figures_root`.
        show: Display the figure instead of closing it, which suits notebooks.

    Returns:
        Path of the written PNG.
    """
    root = Path(figures_root) if figures_root is not None else default_figures_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.png"

    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path
