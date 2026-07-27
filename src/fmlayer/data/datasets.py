from pathlib import Path
from typing import Callable

import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import DTD, FGVCAircraft

from src.fmlayer.data.specs import SPLITS, DatasetSpec, Split, default_data_root, get_spec

DATASET_CLASSES = {"dtd": DTD, "aircraft": FGVCAircraft}


def build_dataset(
    name: str,
    split: Split,
    transform: Callable | None = None,
    root: str | Path | None = None,
    download: bool = True,
) -> Dataset:
    """Instantiate one official split of a Stage 1 dataset.

    Args:
        name: Dataset key, ``"dtd"`` or ``"aircraft"``.
        split: Official split to load; splits are never merged.
        transform: Frozen encoder's eval preprocessing; ``None`` yields PIL images.
        root: Download directory; defaults to :func:`default_data_root`.
        download: Whether to fetch the archive when it is missing.

    Returns:
        The torchvision dataset for that split.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    spec = get_spec(name)
    root = Path(root) if root is not None else default_data_root()
    root.mkdir(parents=True, exist_ok=True)

    return DATASET_CLASSES[name](
        root=str(root),
        split=split,
        transform=transform,
        download=download,
        **spec.extra_kwargs,
    )


def get_targets(dataset: Dataset) -> np.ndarray:
    """Read the integer labels of every item without decoding any image.

    Args:
        dataset: A dataset built by :func:`build_dataset`.

    Returns:
        Labels in dataset order, used for balanced K-shot sampling and split checks.
    """
    # torchvision spells this attribute differently per dataset.
    for attr in ("_labels", "targets", "labels"):
        values = getattr(dataset, attr, None)
        if values is not None:
            return np.asarray(values, dtype=np.int64)

    samples = getattr(dataset, "_samples", None)
    if samples is not None:
        return np.asarray([label for _, label in samples], dtype=np.int64)

    return np.asarray([label for _, label in dataset], dtype=np.int64)


def get_class_names(dataset: Dataset) -> list[str]:
    """Read the raw class names of a dataset.

    Args:
        dataset: A dataset built by :func:`build_dataset`.

    Returns:
        Class names in label-index order, exactly as torchvision provides them.
    """
    classes = getattr(dataset, "classes", None)
    if classes is None:
        raise AttributeError(f"{type(dataset).__name__} exposes no `.classes`")
    return list(classes)


def verify_split(dataset: Dataset, spec: DatasetSpec, split: Split) -> dict:
    """Assert that a materialised split matches the Stage 1 protocol.

    Args:
        dataset: A dataset built by :func:`build_dataset`.
        spec: Expected dataset specification.
        split: Split the dataset was built for.

    Returns:
        A summary of image count, class count and per-class extremes.
        Raises AssertionError on any mismatch.
    """
    targets = get_targets(dataset)
    class_names = get_class_names(dataset)
    counts = np.bincount(targets, minlength=spec.num_classes)

    n_images = len(targets)
    expected = spec.split_sizes.get(split)

    assert len(class_names) == spec.num_classes, (
        f"{spec.key}/{split}: expected {spec.num_classes} classes, got {len(class_names)}"
    )
    assert len(dataset) == n_images, (
        f"{spec.key}/{split}: len(dataset)={len(dataset)} but {n_images} targets"
    )
    if expected is not None:
        assert n_images == expected, (
            f"{spec.key}/{split}: expected {expected} images, got {n_images}. "
            "Check the partition / annotation_level settings."
        )
    assert counts.min() > 0, (
        f"{spec.key}/{split}: classes with no images: "
        f"{np.flatnonzero(counts == 0).tolist()}"
    )

    return {
        "dataset": spec.key,
        "split": split,
        "images": n_images,
        "classes": len(class_names),
        "min_per_class": int(counts.min()),
        "max_per_class": int(counts.max()),
    }
