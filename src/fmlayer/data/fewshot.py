import os
from pathlib import Path

import numpy as np

from src.fmlayer.data.specs import REPO_ROOT, get_spec
from src.fmlayer.features.cache import load_split

K_FULL = "full"
K_VALUES = (5, 10, K_FULL)
SEEDS = (0, 1, 2)
SUBSET_DIRNAME = "subsets"
SUBSET_FILENAME = "{dataset}_k{k}_seed{seed}.npy"


def default_subset_root() -> Path:
    """Resolve where the sampled subset indices are stored.

    Returns:
        ``$FMLAYER_FEATURE_ROOT/subsets`` when set, else the Colab or repo equivalent.
    """
    env = os.environ.get("FMLAYER_FEATURE_ROOT")
    if env:
        return Path(env) / SUBSET_DIRNAME
    if Path("/content").is_dir():
        return Path("/content/features") / SUBSET_DIRNAME
    return REPO_ROOT / "features" / SUBSET_DIRNAME


def subset_path(
    dataset: str, k: int | str, seed: int, subset_root: Path | None = None
) -> Path:
    """Build the path of one persisted subset index file.

    Args:
        dataset: Dataset key.
        k: Shots per class, or ``"full"``.
        seed: Subset seed.
        subset_root: Directory holding the index files; defaults to :func:`default_subset_root`.

    Returns:
        Path of the ``.npy`` index file.
    """
    root = Path(subset_root) if subset_root is not None else default_subset_root()
    return root / SUBSET_FILENAME.format(dataset=dataset, k=k, seed=seed)


def sample_balanced_indices(
    labels: np.ndarray, k: int, seed: int, num_classes: int
) -> np.ndarray:
    """Draw exactly ``k`` training examples per class, without replacement.

    Args:
        labels: Labels of the official training split, in dataset order.
        k: Shots per class.
        seed: Seed making the draw reproducible.
        num_classes: Number of classes; each must have at least ``k`` examples.

    Returns:
        Sorted indices into the training split, of length ``k * num_classes``.
    """
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in range(num_classes):
        class_indices = np.flatnonzero(labels == class_id)
        if len(class_indices) < k:
            raise ValueError(
                f"Class {class_id} has {len(class_indices)} training images, needs {k}."
            )
        selected.append(rng.choice(class_indices, size=k, replace=False))

    indices = np.sort(np.concatenate(selected))
    assert len(indices) == k * num_classes
    assert len(np.unique(indices)) == len(indices)
    return indices


def get_subset_indices(
    dataset: str,
    k: int | str,
    seed: int,
    labels: np.ndarray,
    subset_root: Path | None = None,
    overwrite: bool = False,
) -> np.ndarray:
    """Return the indices of a training subset, reusing the persisted file when present.

    The indices depend only on the dataset, ``k`` and the seed, so every encoder sees
    exactly the same images and the cells stay directly comparable.

    Args:
        dataset: Dataset key.
        k: Shots per class, or ``"full"`` for the complete training split.
        seed: Subset seed, ignored when ``k`` is ``"full"``.
        labels: Labels of the official training split, in dataset order.
        subset_root: Directory holding the index files; defaults to :func:`default_subset_root`.
        overwrite: Resample even when an index file already exists.

    Returns:
        Indices into the training split.
    """
    if k == K_FULL:
        return np.arange(len(labels))

    path = subset_path(dataset, k, seed, subset_root)
    if path.is_file() and not overwrite:
        return np.load(path)

    num_classes = get_spec(dataset).num_classes
    indices = sample_balanced_indices(labels, int(k), seed, num_classes)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, indices)
    return indices


def load_train_subset(
    encoder: str,
    dataset: str,
    k: int | str,
    seed: int,
    feature_root: Path | None = None,
    subset_root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the cached training features restricted to one K-shot subset.

    Args:
        encoder: Encoder key whose cached training features are used.
        dataset: Dataset key.
        k: Shots per class, or ``"full"``.
        seed: Subset seed, ignored when ``k`` is ``"full"``.
        feature_root: Feature cache directory; defaults to the resolved feature root.
        subset_root: Directory holding the index files; defaults to :func:`default_subset_root`.

    Returns:
        The subset features, the subset labels and the indices that produced them.
    """
    features, labels, _ = load_split(encoder, dataset, "train", feature_root)
    indices = get_subset_indices(dataset, k, seed, labels, subset_root)
    return features[indices], labels[indices], indices


def build_all_subsets(
    datasets: list[str] | None = None,
    reference_encoder: str = "clip_rn50",
    feature_root: Path | None = None,
    subset_root: Path | None = None,
    overwrite: bool = False,
) -> dict:
    """Materialise and persist every K-shot subset Stage 1 needs.

    Args:
        datasets: Dataset keys; defaults to both Stage 1 datasets.
        reference_encoder: Encoder whose cached training labels are read; labels are
            identical across encoders, so this only decides which cache file is opened.
        feature_root: Feature cache directory; defaults to the resolved feature root.
        subset_root: Directory holding the index files; defaults to :func:`default_subset_root`.
        overwrite: Resample even when index files already exist.

    Returns:
        The subset sizes, keyed by ``"dataset/k/seed"``.
    """
    datasets = datasets if datasets is not None else ["dtd", "aircraft"]
    sizes = {}

    for dataset in datasets:
        _, labels, _ = load_split(reference_encoder, dataset, "train", feature_root)
        num_classes = get_spec(dataset).num_classes
        print(f"\n=== {dataset} ({len(labels)} training images, {num_classes} classes) ===")

        for k in K_VALUES:
            for seed in SEEDS:
                indices = get_subset_indices(
                    dataset, k, seed, labels, subset_root, overwrite
                )
                counts = np.bincount(labels[indices], minlength=num_classes)
                sizes[f"{dataset}/{k}/{seed}"] = len(indices)
                print(
                    f"    k={str(k):<4} seed={seed}  {len(indices):>5} images, "
                    f"{counts.min()}-{counts.max()} per class"
                )
                if k == K_FULL:
                    break  # the full split does not depend on the subset seed

    return sizes
