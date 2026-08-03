import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.fmlayer.utils.results import default_results_root

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

FIGURES_DIRNAME = "figures"
FIGURE_DPI = 180

# Global Encoder Styling Standard
ENCODER_COLORS = {
    "resnet18": "#1f77b4",       # Standard Deep Blue
    "dinov2_vits14": "#2ca02c",  # Standard Green
    "clip_rn50": "#d62728",      # Standard Crimson Red
}

ENCODER_MARKERS = {
    "resnet18": "o",
    "dinov2_vits14": "s",
    "clip_rn50": "^",
}

ENCODER_LABELS = {
    "resnet18": "ResNet-18",
    "dinov2_vits14": "DINOv2 ViT-S/14",
    "clip_rn50": "CLIP RN50",
}


def apply_plot_style() -> None:
    """Apply a clean, modern matplotlib theme for notebook inline figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "axes.labelweight": "normal",
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "grid.color": "#b0bec5",
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.facecolor": "#ffffff",
        "legend.edgecolor": "#d0d0d0",
        "legend.fontsize": 9,
        "figure.facecolor": "#ffffff",
        "figure.dpi": FIGURE_DPI,
        "savefig.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


def default_figures_root(results_root: Path | None = None) -> Path:
    """Resolve the directory figures are written to."""
    root = Path(results_root) if results_root is not None else default_results_root()
    return root / FIGURES_DIRNAME


def save_figure(
    fig: Figure, name: str = "", figures_root: Path | None = None, show: bool = True, save: bool = False
) -> Path | None:
    """Display the figure inline in the notebook cell output.

    Args:
        fig: The matplotlib figure.
        name: File name without extension.
        figures_root: Output directory.
        show: Display figure inline in notebook cell.
        save: Save PNG file to disk (default False).

    Returns:
        Path of saved file if save=True, else None.
    """
    path = None
    if save and name:
        root = Path(figures_root) if figures_root is not None else default_figures_root()
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{name}.png"
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return path
