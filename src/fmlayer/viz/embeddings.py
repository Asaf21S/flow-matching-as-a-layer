from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.fmlayer.data.class_names import normalize_class_name
from src.fmlayer.data.specs import get_spec
from src.fmlayer.encoders.clip_rn50 import ClipRN50Encoder
from src.fmlayer.encoders.registry import ENCODER_DATASETS
from src.fmlayer.features.cache import load_split
from src.fmlayer.models.prototypes import image_prototypes, l2_normalize
from src.fmlayer.viz.figures import ENCODER_LABELS, apply_plot_style, save_figure

DEFAULT_NUM_CLASSES = 8
CLASS_SELECTION_SEED = 0
PROJECTION_SEED = 0
COLORMAP = "tab10"


def select_classes(
    dataset: str, num_classes: int = DEFAULT_NUM_CLASSES, seed: int = CLASS_SELECTION_SEED
) -> np.ndarray:
    """Pick a readable subset of classes to visualise."""
    rng = np.random.default_rng(seed)
    total = get_spec(dataset).num_classes
    return np.sort(rng.choice(total, size=num_classes, replace=False))


def class_prototypes(
    encoder: str, dataset: str, class_ids: np.ndarray, feature_root: Path | None = None
) -> tuple[np.ndarray, str]:
    """Build prototypes to overlay for one encoder."""
    if encoder == ClipRN50Encoder.NAME:
        prototypes, _, _ = load_split(encoder, dataset, None, feature_root)
        return prototypes[class_ids], "text prototype"

    features, labels, _ = load_split(encoder, dataset, "train", feature_root)
    prototypes = image_prototypes(features, labels, get_spec(dataset).num_classes)
    return prototypes[class_ids], "image prototype"


def project_2d(points: np.ndarray, method: str = "pca", seed: int = PROJECTION_SEED) -> np.ndarray:
    """Project features to two dimensions."""
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(points)
    if method == "tsne":
        perplexity = min(30.0, max(5.0, (len(points) - 1) / 3.0))
        return TSNE(
            n_components=2, random_state=seed, perplexity=perplexity, init="pca"
        ).fit_transform(points)
    raise ValueError(f"Unknown projection {method!r}; use 'pca' or 'tsne'.")


def plot_combined_embeddings_for_dataset(
    dataset: str,
    encoders: list[str] | None = None,
    method: str = "pca",
    num_classes: int = DEFAULT_NUM_CLASSES,
    feature_root: Path | None = None,
    show: bool = True,
) -> None:
    """Visualise 2D embeddings for several encoders side-by-side in a single combined figure."""
    apply_plot_style()
    if encoders is None:
        encoders = [
            name for name, datasets in ENCODER_DATASETS.items() if dataset in datasets
        ]
    class_ids = select_classes(dataset, num_classes)
    colors = matplotlib.colormaps[COLORMAP]

    fig, axes = plt.subplots(1, len(encoders), figsize=(5.2 * len(encoders), 4.8))
    if len(encoders) == 1:
        axes = [axes]

    for ax, encoder in zip(axes, encoders):
        features, labels, metadata = load_split(encoder, dataset, "test", feature_root)
        mask = np.isin(labels, class_ids)
        features, labels = l2_normalize(features[mask]), labels[mask]

        prototypes, prototype_label = class_prototypes(encoder, dataset, class_ids, feature_root)
        prototypes = l2_normalize(prototypes)

        coordinates = project_2d(np.concatenate([features, prototypes]), method)
        feature_xy, prototype_xy = coordinates[: len(features)], coordinates[len(features) :]

        class_names = metadata["class_names"]
        for position, class_id in enumerate(class_ids):
            color = colors(position % colors.N)
            points = feature_xy[labels == class_id]
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=20,
                alpha=0.55,
                color=color,
                label=normalize_class_name(class_names[class_id]),
            )
            ax.scatter(
                prototype_xy[position, 0],
                prototype_xy[position, 1],
                s=240,
                marker="*",
                color=color,
                edgecolor="black",
                linewidth=1.0,
                zorder=5,
            )

        enc_label = ENCODER_LABELS.get(encoder, encoder)
        ax.set_title(f"{enc_label} ({method.upper()})", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])

    handles, labels_list = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels_list, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=True, framealpha=0.95)
    fig.suptitle(f"{get_spec(dataset).display_name}: 2D Feature Visualizations ({method.upper()})", fontsize=13, y=1.02)
    fig.tight_layout()
    save_figure(fig, f"embedding_combined_{method}_{dataset}", show=show, save=False)


def plot_embedding(
    encoder: str,
    dataset: str,
    class_ids: np.ndarray | None = None,
    method: str = "pca",
    feature_root: Path | None = None,
    figures_root: Path | None = None,
    show: bool = True,
    show_class_means: bool = False,
    save: bool = False,
) -> Path | None:
    """Visualise test features together with class prototypes inline in the notebook."""
    apply_plot_style()
    class_ids = class_ids if class_ids is not None else select_classes(dataset)

    features, labels, metadata = load_split(encoder, dataset, "test", feature_root)
    mask = np.isin(labels, class_ids)
    features, labels = l2_normalize(features[mask]), labels[mask]

    prototypes, prototype_label = class_prototypes(encoder, dataset, class_ids, feature_root)
    prototypes = l2_normalize(prototypes)

    coordinates = project_2d(np.concatenate([features, prototypes]), method)
    feature_xy, prototype_xy = coordinates[: len(features)], coordinates[len(features) :]

    colors = matplotlib.colormaps[COLORMAP]
    class_names = metadata["class_names"]

    fig, ax = plt.subplots(figsize=(8.5, 6))
    for position, class_id in enumerate(class_ids):
        color = colors(position % colors.N)
        points = feature_xy[labels == class_id]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=22,
            alpha=0.55,
            color=color,
            label=normalize_class_name(class_names[class_id]),
        )
        ax.scatter(
            prototype_xy[position, 0],
            prototype_xy[position, 1],
            s=280,
            marker="*",
            color=color,
            edgecolor="black",
            linewidth=1.2,
            zorder=5,
        )
        if show_class_means and len(points):
            ax.scatter(
                points[:, 0].mean(),
                points[:, 1].mean(),
                s=100,
                marker="X",
                color=color,
                edgecolor="black",
                linewidth=1.0,
                zorder=4,
            )

    markers = f"stars = {prototype_label}"
    if show_class_means:
        markers += ", crosses = projected class mean"
    enc_label = ENCODER_LABELS.get(encoder, encoder)
    ax.set_title(
        f"{get_spec(dataset).display_name}: {enc_label} 2D Feature Projection ({method.upper()})\n({markers})",
        pad=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, framealpha=0.95)
    fig.tight_layout()

    return save_figure(fig, f"embedding_{method}_{encoder}_{dataset}", figures_root, show=show, save=save)


def plot_embeddings_for_dataset(
    dataset: str,
    encoders: list[str] | None = None,
    method: str = "pca",
    num_classes: int = DEFAULT_NUM_CLASSES,
    feature_root: Path | None = None,
    figures_root: Path | None = None,
    show: bool = True,
    show_class_means: bool = False,
    save: bool = False,
) -> list[Path | None]:
    """Visualise 2D embeddings for several encoders on the same dataset."""
    if encoders is None:
        encoders = [
            name for name, datasets in ENCODER_DATASETS.items() if dataset in datasets
        ]
    class_ids = select_classes(dataset, num_classes)
    return [
        plot_embedding(
            encoder,
            dataset,
            class_ids,
            method,
            feature_root,
            figures_root,
            show=show,
            show_class_means=show_class_means,
            save=save,
        )
        for encoder in encoders
    ]
