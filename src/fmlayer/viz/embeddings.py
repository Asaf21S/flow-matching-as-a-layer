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
from src.fmlayer.viz.figures import save_figure

DEFAULT_NUM_CLASSES = 8
CLASS_SELECTION_SEED = 0
PROJECTION_SEED = 0
COLORMAP = "tab10"


def select_classes(
    dataset: str, num_classes: int = DEFAULT_NUM_CLASSES, seed: int = CLASS_SELECTION_SEED
) -> np.ndarray:
    """Pick a readable subset of classes to visualise.

    The choice depends only on the dataset and the seed, so every encoder is compared
    on exactly the same classes.

    Args:
        dataset: Dataset key.
        num_classes: How many classes to show, typically 8 to 10.
        seed: Seed making the choice reproducible.

    Returns:
        Sorted class indices.
    """
    rng = np.random.default_rng(seed)
    total = get_spec(dataset).num_classes
    return np.sort(rng.choice(total, size=num_classes, replace=False))


def class_prototypes(
    encoder: str, dataset: str, class_ids: np.ndarray, feature_root: Path | None = None
) -> tuple[np.ndarray, str]:
    """Build the prototypes to overlay for one encoder.

    CLIP uses its text prototypes; the image encoders use prototypes derived from the
    full training split.

    Args:
        encoder: Encoder key.
        dataset: Dataset key.
        class_ids: Classes being visualised.
        feature_root: Feature cache directory; defaults to the resolved feature root.

    Returns:
        Prototypes of the selected classes and a label describing their origin.
    """
    if encoder == ClipRN50Encoder.NAME:
        prototypes, _, _ = load_split(encoder, dataset, None, feature_root)
        return prototypes[class_ids], "text prototype"

    features, labels, _ = load_split(encoder, dataset, "train", feature_root)
    prototypes = image_prototypes(features, labels, get_spec(dataset).num_classes)
    return prototypes[class_ids], "image prototype"


def project_2d(points: np.ndarray, method: str = "pca", seed: int = PROJECTION_SEED) -> np.ndarray:
    """Project features to two dimensions.

    Args:
        points: Array of shape ``(num_items, dim)``, features and prototypes stacked.
        method: ``"pca"`` or ``"tsne"``.
        seed: Seed for the stochastic projection.

    Returns:
        Coordinates of shape ``(num_items, 2)``.
    """
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(points)
    if method == "tsne":
        perplexity = min(30.0, max(5.0, (len(points) - 1) / 3.0))
        return TSNE(
            n_components=2, random_state=seed, perplexity=perplexity, init="pca"
        ).fit_transform(points)
    raise ValueError(f"Unknown projection {method!r}; use 'pca' or 'tsne'.")


def plot_embedding(
    encoder: str,
    dataset: str,
    class_ids: np.ndarray | None = None,
    method: str = "pca",
    feature_root: Path | None = None,
    figures_root: Path | None = None,
    show: bool = True,
) -> Path:
    """Visualise test features together with their class prototypes.

    The projection is fitted jointly on the features and the prototypes that appear in
    the plot, and class colours follow the position in ``class_ids`` so they match
    across encoders.

    Args:
        encoder: Encoder key.
        dataset: Dataset key.
        class_ids: Classes to show; defaults to :func:`select_classes`.
        method: ``"pca"`` or ``"tsne"``.
        feature_root: Feature cache directory; defaults to the resolved feature root.
        figures_root: Figure directory; defaults to the figures root.
        show: Display the figure instead of closing it.

    Returns:
        Path of the written figure.
    """
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

    fig, ax = plt.subplots(figsize=(8, 7))
    for position, class_id in enumerate(class_ids):
        color = colors(position % colors.N)
        points = feature_xy[labels == class_id]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=14,
            alpha=0.6,
            color=color,
            label=normalize_class_name(class_names[class_id]),
        )
        ax.scatter(
            prototype_xy[position, 0],
            prototype_xy[position, 1],
            s=260,
            marker="*",
            color=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )

    ax.set_title(
        f"{get_spec(dataset).display_name} - {encoder} test features "
        f"({method.upper()}, stars = {prototype_label})"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    fig.tight_layout()

    return save_figure(fig, f"embedding_{method}_{encoder}_{dataset}", figures_root, show)


def plot_embeddings_for_dataset(
    dataset: str,
    encoders: list[str] | None = None,
    method: str = "pca",
    num_classes: int = DEFAULT_NUM_CLASSES,
    feature_root: Path | None = None,
    figures_root: Path | None = None,
    show: bool = True,
) -> list[Path]:
    """Visualise several encoders on the same classes, examples and colours.

    Args:
        dataset: Dataset key.
        encoders: Encoder keys; defaults to every encoder cached for this dataset.
        method: ``"pca"`` or ``"tsne"``.
        num_classes: How many classes to show.
        feature_root: Feature cache directory; defaults to the resolved feature root.
        figures_root: Figure directory; defaults to the figures root.
        show: Display the figures instead of closing them.

    Returns:
        Paths of the written figures.
    """
    if encoders is None:
        encoders = [
            name for name, datasets in ENCODER_DATASETS.items() if dataset in datasets
        ]
    class_ids = select_classes(dataset, num_classes)
    return [
        plot_embedding(encoder, dataset, class_ids, method, feature_root, figures_root, show)
        for encoder in encoders
    ]
