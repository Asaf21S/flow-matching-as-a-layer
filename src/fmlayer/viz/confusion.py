from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.fmlayer.data.class_names import normalize_class_names
from src.fmlayer.data.specs import get_spec
from src.fmlayer.features.cache import load_split
from src.fmlayer.models.zeroshot import run_zeroshot
from src.fmlayer.train.evaluate import confusion_matrix
from src.fmlayer.train.train_linear import run_linear_probe
from src.fmlayer.viz.figures import ENCODER_LABELS, apply_plot_style, save_figure

MAX_TICK_LABELS = 50


def plot_combined_confusion(
    datasets: list[str] | None = None,
    k: int | str = "full",
    seed: int = 0,
    feature_root: Path | None = None,
    results_root: Path | None = None,
    show: bool = True,
) -> None:
    """Plot confusion matrices for Zero-Shot CLIP and Linear Probes together in a combined grid."""
    apply_plot_style()
    datasets = datasets if datasets is not None else ["dtd", "aircraft"]
    fig, axes = plt.subplots(len(datasets), 2, figsize=(13, 5.8 * len(datasets)))

    for i, dataset in enumerate(datasets):
        spec = get_spec(dataset)

        # Zero-shot CLIP
        zs_res = run_zeroshot(dataset, feature_root, record=False)
        _, _, zs_meta = load_split("clip_rn50", dataset, "test", feature_root)
        zs_classes = zs_meta["class_names"]
        zs_num = len(zs_classes)
        zs_mat = confusion_matrix(zs_res["predictions"], zs_res["labels"], zs_num, normalize=True)

        ax_zs = axes[i, 0]
        img_zs = ax_zs.imshow(zs_mat, cmap="Blues", vmin=0.0, vmax=1.0)
        if zs_num <= MAX_TICK_LABELS:
            ax_zs.set_xticks(range(zs_num), normalize_class_names(zs_classes), rotation=90, fontsize=6)
            ax_zs.set_yticks(range(zs_num), normalize_class_names(zs_classes), fontsize=6)
        else:
            ax_zs.set_xticks([])
            ax_zs.set_yticks([])
        ax_zs.set_xlabel("Predicted Class")
        ax_zs.set_ylabel("True Class")
        ax_zs.set_title(f"{spec.display_name} - Zero-Shot CLIP RN50 ({zs_res['accuracy']*100:.1f}%)", pad=8)
        fig.colorbar(img_zs, ax=ax_zs, fraction=0.046, pad=0.04)

        # Linear Probe (ResNet-18)
        probe_res = run_linear_probe(
            "resnet18", dataset, k, seed, feature_root=feature_root, results_root=results_root, record=False, verbose=False
        )
        _, _, probe_meta = load_split("resnet18", dataset, "test", feature_root)
        probe_classes = probe_meta["class_names"]
        probe_num = len(probe_classes)
        probe_mat = confusion_matrix(probe_res["predictions"], probe_res["labels"], probe_num, normalize=True)

        ax_probe = axes[i, 1]
        img_probe = ax_probe.imshow(probe_mat, cmap="Blues", vmin=0.0, vmax=1.0)
        if probe_num <= MAX_TICK_LABELS:
            ax_probe.set_xticks(range(probe_num), normalize_class_names(probe_classes), rotation=90, fontsize=6)
            ax_probe.set_yticks(range(probe_num), normalize_class_names(probe_classes), fontsize=6)
        else:
            ax_probe.set_xticks([])
            ax_probe.set_yticks([])
        ax_probe.set_xlabel("Predicted Class")
        ax_probe.set_ylabel("True Class")
        ax_probe.set_title(f"{spec.display_name} - Linear Probe (ResNet-18, K={k}) ({probe_res['test_accuracy']*100:.1f}%)", pad=8)
        fig.colorbar(img_probe, ax=ax_probe, fraction=0.046, pad=0.04)

    fig.suptitle("Row-Normalized Confusion Matrices Comparison", fontsize=13, y=1.01)
    fig.tight_layout()
    save_figure(fig, "confusion_combined", show=show, save=False)


def plot_confusion(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    title: str,
    name: str,
    figures_root: Path | None = None,
    show: bool = True,
    save: bool = False,
) -> Path | None:
    """Draw a single row-normalised confusion matrix."""
    apply_plot_style()
    num_classes = len(class_names)
    matrix = confusion_matrix(predictions, labels, num_classes, normalize=True)

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)

    if num_classes <= MAX_TICK_LABELS:
        ax.set_xticks(range(num_classes), normalize_class_names(class_names), rotation=90, fontsize=6.5)
        ax.set_yticks(range(num_classes), normalize_class_names(class_names), fontsize=6.5)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_xlabel("Predicted Class", labelpad=8)
    ax.set_ylabel("True Class", labelpad=8)
    ax.set_title(title, pad=12)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fraction of True Class", labelpad=8)
    fig.tight_layout()

    return save_figure(fig, name, figures_root, show=show, save=save)


def plot_confusion_for_probe(
    encoder: str,
    dataset: str,
    k: int | str = "full",
    seed: int = 0,
    feature_root: Path | None = None,
    results_root: Path | None = None,
    figures_root: Path | None = None,
    show: bool = True,
    save: bool = False,
) -> Path | None:
    """Draw confusion matrix for a linear probe run."""
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
    enc_label = ENCODER_LABELS.get(encoder, encoder)
    title = (
        f"{get_spec(dataset).display_name} - Linear Probe on {enc_label} "
        f"(K={k}, top-1 {result['test_accuracy'] * 100:.1f}%)"
    )
    return plot_confusion(
        result["predictions"],
        result["labels"],
        metadata["class_names"],
        title,
        f"confusion_probe_{encoder}_{dataset}_k{k}_seed{seed}",
        figures_root,
        show=show,
        save=save,
    )


def plot_confusion_for_zeroshot(
    dataset: str,
    feature_root: Path | None = None,
    figures_root: Path | None = None,
    show: bool = True,
    save: bool = False,
) -> Path | None:
    """Draw confusion matrix for zero-shot CLIP baseline."""
    result = run_zeroshot(dataset, feature_root, record=False)
    _, _, metadata = load_split("clip_rn50", dataset, "test", feature_root)
    enc_label = ENCODER_LABELS.get("clip_rn50", "CLIP RN50")
    title = (
        f"{get_spec(dataset).display_name} - Zero-Shot {enc_label} "
        f"(top-1 {result['accuracy'] * 100:.1f}%)"
    )
    return plot_confusion(
        result["predictions"],
        result["labels"],
        metadata["class_names"],
        title,
        f"confusion_zeroshot_{dataset}",
        figures_root,
        show=show,
        save=save,
    )
