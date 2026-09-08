import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.fmlayer.data.specs import get_spec
from src.fmlayer.encoders.registry import LINEAR_PROBE_CELLS
from src.fmlayer.train.train_linear import curves_path
from src.fmlayer.viz.figures import (
    ENCODER_COLORS,
    ENCODER_LABELS,
    apply_plot_style,
    save_figure,
)

REPRESENTATIVE_K = 10
REPRESENTATIVE_SEED = 0


def load_history(
    encoder: str, dataset: str, k: int | str, seed: int, results_root: Path | None = None
) -> dict:
    """Read a saved per-epoch history."""
    path = curves_path(encoder, dataset, k, seed, results_root)
    if not path.is_file():
        raise FileNotFoundError(f"No saved curves at {path}. Run the linear probe first.")
    return json.loads(path.read_text(encoding="utf-8"))


def plot_combined_curves(
    cells: tuple[tuple[str, str], ...] | None = None,
    k: int | str = REPRESENTATIVE_K,
    seed: int = REPRESENTATIVE_SEED,
    results_root: Path | None = None,
    show: bool = True,
) -> None:
    """Plot representative loss and accuracy training curves together in a combined figure."""
    apply_plot_style()
    cells = cells if cells is not None else LINEAR_PROBE_CELLS
    fig, axes = plt.subplots(2, len(cells), figsize=(3.5 * len(cells), 5.5))

    for i, (encoder, dataset) in enumerate(cells):
        run = load_history(encoder, dataset, k, seed, results_root)
        epochs = [entry["epoch"] for entry in run["history"]]
        train_loss = [entry["train_loss"] for entry in run["history"]]
        val_loss = [entry["val_loss"] for entry in run["history"]]
        val_accuracy = [entry["val_accuracy"] for entry in run["history"]]

        best_epoch = run["best_epoch"]
        main_color = ENCODER_COLORS.get(encoder, "#1f77b4")
        label = ENCODER_LABELS.get(encoder, encoder)
        spec = get_spec(dataset)

        loss_ax = axes[0, i]
        acc_ax = axes[1, i]

        loss_ax.plot(epochs, train_loss, color=main_color, linewidth=2, label="Train Loss")
        loss_ax.plot(epochs, val_loss, color="#dc2626", linewidth=2, linestyle="--", label="Val Loss")
        loss_ax.axvline(best_epoch, color="#555555", linestyle=":", linewidth=1.5, label=f"Best ({best_epoch})")
        loss_ax.set_xlabel("Epoch", labelpad=6)
        loss_ax.set_ylabel("Cross-Entropy Loss", labelpad=6)
        loss_ax.set_title(f"{label} ({spec.display_name})\nLoss Curves", pad=8)
        loss_ax.legend(loc="upper right", frameon=True)

        acc_ax.plot(epochs, val_accuracy, color=main_color, linewidth=2, label="Val Accuracy")
        acc_ax.axvline(best_epoch, color="#555555", linestyle=":", linewidth=1.5)
        acc_ax.set_xlabel("Epoch", labelpad=6)
        acc_ax.set_ylabel("Validation Accuracy", labelpad=6)
        acc_ax.set_title(f"Val Accuracy ({spec.display_name})", pad=8)
        acc_ax.legend(loc="lower right", frameon=True)

    fig.suptitle(f"Linear Probe Training Curves (K={k}, Seed {seed})", fontsize=13, y=1.02)
    fig.tight_layout()
    save_figure(fig, "curves_combined", show=show, save=False)


def plot_curves(
    encoder: str,
    dataset: str,
    k: int | str = REPRESENTATIVE_K,
    seed: int = REPRESENTATIVE_SEED,
    results_root: Path | None = None,
    figures_root: Path | None = None,
    show: bool = True,
    save: bool = False,
) -> Path | None:
    """Plot training and validation loss for a single run."""
    apply_plot_style()
    run = load_history(encoder, dataset, k, seed, results_root)
    epochs = [entry["epoch"] for entry in run["history"]]
    train_loss = [entry["train_loss"] for entry in run["history"]]
    val_loss = [entry["val_loss"] for entry in run["history"]]
    val_accuracy = [entry["val_accuracy"] for entry in run["history"]]

    best_epoch = run["best_epoch"]
    main_color = ENCODER_COLORS.get(encoder, "#1f77b4")
    label = ENCODER_LABELS.get(encoder, encoder)

    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(8.5, 3.5))

    loss_ax.plot(epochs, train_loss, color=main_color, linewidth=2, label="Train Loss")
    loss_ax.plot(epochs, val_loss, color="#dc2626", linewidth=2, linestyle="--", label="Val Loss")
    loss_ax.axvline(
        best_epoch, color="#555555", linestyle=":", linewidth=1.5, label=f"Best Epoch ({best_epoch})"
    )
    loss_ax.set_xlabel("Epoch", labelpad=6)
    loss_ax.set_ylabel("Cross-Entropy Loss", labelpad=6)
    loss_ax.set_title("Training & Validation Loss")
    loss_ax.legend(loc="upper right", frameon=True)

    acc_ax.plot(epochs, val_accuracy, color=main_color, linewidth=2, label="Val Accuracy")
    acc_ax.axvline(best_epoch, color="#555555", linestyle=":", linewidth=1.5)
    acc_ax.set_xlabel("Epoch", labelpad=6)
    acc_ax.set_ylabel("Validation Accuracy", labelpad=6)
    acc_ax.set_title("Validation Accuracy")
    acc_ax.legend(loc="lower right", frameon=True)

    fig.suptitle(
        f"Linear Probe Training Curves: {label} | {dataset.upper()} | K={k} (Seed {seed})",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()

    return save_figure(fig, f"curves_{encoder}_{dataset}_k{k}_seed{seed}", figures_root, show=show, save=save)


def plot_representative_curves(
    cells: tuple[tuple[str, str], ...] | None = None,
    k: int | str = REPRESENTATIVE_K,
    seed: int = REPRESENTATIVE_SEED,
    results_root: Path | None = None,
    figures_root: Path | None = None,
    show: bool = True,
    save: bool = False,
) -> list[Path | None]:
    """Plot representative curves for each encoder/dataset cell."""
    cells = cells if cells is not None else LINEAR_PROBE_CELLS
    return [
        plot_curves(encoder, dataset, k, seed, results_root, figures_root, show=show, save=save)
        for encoder, dataset in cells
    ]
