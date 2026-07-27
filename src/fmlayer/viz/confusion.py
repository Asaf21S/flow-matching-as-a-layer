from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.fmlayer.data.class_names import normalize_class_names
from src.fmlayer.data.specs import get_spec
from src.fmlayer.features.cache import load_split
from src.fmlayer.models.zeroshot import run_zeroshot
from src.fmlayer.train.evaluate import confusion_matrix
from src.fmlayer.train.train_linear import run_linear_probe
from src.fmlayer.viz.figures import save_figure

MAX_TICK_LABELS = 50


def plot_confusion(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    title: str,
    name: str,
    figures_root: Path | None = None,
    show: bool = True,
) -> Path:
    """Draw a row-normalised confusion matrix.

    Args:
        predictions: Predicted labels of the test split.
        labels: Ground-truth labels of the test split.
        class_names: Class names in label-index order.
        title: Figure title.
        name: File name without extension.
        figures_root: Figure directory; defaults to the figures root.
        show: Display the figure instead of closing it.

    Returns:
        Path of the written figure.
    """
    num_classes = len(class_names)
    matrix = confusion_matrix(predictions, labels, num_classes, normalize=True)

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)

    # Dense axes become unreadable, so labels are only drawn for small class counts.
    if num_classes <= MAX_TICK_LABELS:
        ax.set_xticks(range(num_classes), normalize_class_names(class_names), rotation=90, fontsize=6)
        ax.set_yticks(range(num_classes), normalize_class_names(class_names), fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, label="fraction of true class")
    fig.tight_layout()

    return save_figure(fig, name, figures_root, show)


def plot_confusion_for_probe(
    encoder: str,
    dataset: str,
    k: int | str = "full",
    seed: int = 0,
    feature_root: Path | None = None,
    results_root: Path | None = None,
    figures_root: Path | None = None,
    show: bool = True,
) -> Path:
    """Retrain one linear probe and draw its confusion matrix.

    Training on cached features takes seconds, so the run is repeated rather than
    storing predictions for all 27 runs.

    Args:
        encoder: Encoder key.
        dataset: Dataset key.
        k: Shots per class, or ``"full"``.
        seed: Run seed.
        feature_root: Feature cache directory; defaults to the resolved feature root.
        results_root: Results directory; defaults to the resolved results root.
        figures_root: Figure directory; defaults to the figures root.
        show: Display the figure instead of closing it.

    Returns:
        Path of the written figure.
    """
    result = run_linear_probe(
        encoder,
        dataset,
        k,
        seed,
        feature_root=feature_root,
        results_root=results_root,
        record=False,
        verbose=False,
    )
    _, _, metadata = load_split(encoder, dataset, "test", feature_root)
    title = (
        f"{get_spec(dataset).display_name} - linear probe on {encoder} "
        f"(K={k}, top-1 {result['test_accuracy']:.3f})"
    )
    return plot_confusion(
        result["predictions"],
        result["labels"],
        metadata["class_names"],
        title,
        f"confusion_probe_{encoder}_{dataset}_k{k}_seed{seed}",
        figures_root,
        show,
    )


def plot_confusion_for_zeroshot(
    dataset: str,
    feature_root: Path | None = None,
    figures_root: Path | None = None,
    show: bool = True,
) -> Path:
    """Draw the confusion matrix of the zero-shot CLIP baseline.

    Args:
        dataset: Dataset key.
        feature_root: Feature cache directory; defaults to the resolved feature root.
        figures_root: Figure directory; defaults to the figures root.
        show: Display the figure instead of closing it.

    Returns:
        Path of the written figure.
    """
    result = run_zeroshot(dataset, feature_root, record=False)
    _, _, metadata = load_split("clip_rn50", dataset, "test", feature_root)
    title = (
        f"{get_spec(dataset).display_name} - zero-shot CLIP RN50 "
        f"(top-1 {result['accuracy']:.3f})"
    )
    return plot_confusion(
        result["predictions"],
        result["labels"],
        metadata["class_names"],
        title,
        f"confusion_zeroshot_{dataset}",
        figures_root,
        show,
    )
