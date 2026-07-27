import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.fmlayer.encoders.registry import LINEAR_PROBE_CELLS
from src.fmlayer.train.train_linear import curves_path
from src.fmlayer.viz.figures import save_figure

REPRESENTATIVE_K = 10
REPRESENTATIVE_SEED = 0


def load_history(
    encoder: str, dataset: str, k: int | str, seed: int, results_root: Path | None = None
) -> dict:
    """Read a saved per-epoch history.

    Args:
        encoder: Encoder key.
        dataset: Dataset key.
        k: Shots per class, or ``"full"``.
        seed: Run seed.
        results_root: Results directory; defaults to the resolved results root.

    Returns:
        The stored run description including its ``history`` list.
    """
    path = curves_path(encoder, dataset, k, seed, results_root)
    if not path.is_file():
        raise FileNotFoundError(f"No saved curves at {path}. Run the linear probe first.")
    return json.loads(path.read_text(encoding="utf-8"))


def plot_curves(
    encoder: str,
    dataset: str,
    k: int | str = REPRESENTATIVE_K,
    seed: int = REPRESENTATIVE_SEED,
    results_root: Path | None = None,
    figures_root: Path | None = None,
    show: bool = True,
) -> Path:
    """Plot the training and validation loss of one linear-probe run.

    The selected epoch is marked, which makes overfitting after that point visible.

    Args:
        encoder: Encoder key.
        dataset: Dataset key.
        k: Shots per class, or ``"full"``.
        seed: Run seed.
        results_root: Results directory; defaults to the resolved results root.
        figures_root: Figure directory; defaults to the figures root.
        show: Display the figure instead of closing it.

    Returns:
        Path of the written figure.
    """
    run = load_history(encoder, dataset, k, seed, results_root)
    epochs = [entry["epoch"] for entry in run["history"]]
    train_loss = [entry["train_loss"] for entry in run["history"]]
    val_loss = [entry["val_loss"] for entry in run["history"]]
    val_accuracy = [entry["val_accuracy"] for entry in run["history"]]

    best_epoch = run["best_epoch"]
    min_loss_epoch = epochs[int(min(range(len(val_loss)), key=val_loss.__getitem__))]

    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(11, 4))

    loss_ax.plot(epochs, train_loss, label="train loss")
    loss_ax.plot(epochs, val_loss, label="val loss")
    loss_ax.axvline(
        best_epoch, color="grey", linestyle="--", label=f"selected epoch {best_epoch} (max val acc)"
    )
    loss_ax.axvline(
        min_loss_epoch,
        color="tab:red",
        linestyle=":",
        label=f"min val loss epoch {min_loss_epoch}",
    )
    loss_ax.set_xlabel("epoch")
    loss_ax.set_ylabel("cross-entropy loss")
    loss_ax.legend(fontsize=8)

    acc_ax.plot(epochs, val_accuracy, color="tab:green")
    acc_ax.axvline(best_epoch, color="grey", linestyle="--")
    acc_ax.axvline(min_loss_epoch, color="tab:red", linestyle=":")
    acc_ax.set_xlabel("epoch")
    acc_ax.set_ylabel("validation accuracy")

    fig.suptitle(
        f"{encoder} / {dataset} / K={k} / seed={seed} ({run['num_train']} training images)"
    )
    fig.tight_layout()

    return save_figure(fig, f"curves_{encoder}_{dataset}_k{k}_seed{seed}", figures_root, show)


def plot_representative_curves(
    cells: tuple[tuple[str, str], ...] | None = None,
    k: int | str = REPRESENTATIVE_K,
    seed: int = REPRESENTATIVE_SEED,
    results_root: Path | None = None,
    figures_root: Path | None = None,
    show: bool = True,
) -> list[Path]:
    """Plot one representative run per encoder/dataset cell.

    Args:
        cells: ``(encoder, dataset)`` pairs; defaults to the Stage 1 probe cells.
        k: Shots per class of the representative run.
        seed: Seed of the representative run.
        results_root: Results directory; defaults to the resolved results root.
        figures_root: Figure directory; defaults to the figures root.
        show: Display the figures instead of closing them.

    Returns:
        Paths of the written figures.
    """
    cells = cells if cells is not None else LINEAR_PROBE_CELLS
    return [
        plot_curves(encoder, dataset, k, seed, results_root, figures_root, show)
        for encoder, dataset in cells
    ]
