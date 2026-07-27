import hashlib
import json
import os
from pathlib import Path

import numpy as np

from src.fmlayer.data.specs import REPO_ROOT

IMAGE_FEATURE_FILENAME = "{dataset}_{split}.npz"
TEXT_FEATURE_FILENAME = "{dataset}_text.npz"


def default_feature_root() -> Path:
    """Resolve the feature cache directory: ``FMLAYER_FEATURE_ROOT``, then Colab, then the repo.

    Returns:
        Directory the ``.npz`` caches are written to.
    """
    env = os.environ.get("FMLAYER_FEATURE_ROOT")
    if env:
        return Path(env)
    if Path("/content").is_dir():
        return Path("/content/features")
    return REPO_ROOT / "features"


def feature_path(
    encoder: str, dataset: str, split: str | None = None, feature_root: Path | None = None
) -> Path:
    """Build the cache path of one encoder/dataset/split combination.

    Args:
        encoder: Encoder key.
        dataset: Dataset key.
        split: Official split, or ``None`` for the CLIP text prototypes.
        feature_root: Cache directory; defaults to :func:`default_feature_root`.

    Returns:
        Path of the ``.npz`` cache file.
    """
    root = Path(feature_root) if feature_root is not None else default_feature_root()
    filename = (
        TEXT_FEATURE_FILENAME.format(dataset=dataset)
        if split is None
        else IMAGE_FEATURE_FILENAME.format(dataset=dataset, split=split)
    )
    return root / encoder / filename


def config_hash(metadata: dict) -> str:
    """Hash the identifying fields of a cache so a stale file can be detected.

    Args:
        metadata: Metadata describing how the features were produced.

    Returns:
        A short hex digest.
    """
    keys = ("encoder", "dataset", "split", "embed_dim", "num_items", "transform")
    payload = {key: metadata.get(key) for key in keys}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def save_features(
    path: Path, features: np.ndarray, labels: np.ndarray, metadata: dict
) -> Path:
    """Write features, labels and metadata to a compressed cache file.

    Args:
        path: Destination ``.npz`` path.
        features: Array of shape ``(num_items, embed_dim)``.
        labels: Integer labels of shape ``(num_items,)``; empty for text prototypes.
        metadata: Provenance recorded alongside the arrays.

    Returns:
        The path that was written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {**metadata, "hash": config_hash(metadata)}
    np.savez_compressed(
        path,
        features=features.astype(np.float32),
        labels=labels.astype(np.int64),
        metadata=json.dumps(metadata),
    )
    return path


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read a cache file written by :func:`save_features`.

    Args:
        path: Cache file to read.

    Returns:
        The features, the labels and the metadata dict.
    """
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        return data["features"], data["labels"], metadata


def is_cached(path: Path, expected: dict) -> bool:
    """Check whether a usable cache already exists for a given configuration.

    Args:
        path: Cache file to check.
        expected: Metadata the cache must match.

    Returns:
        True when the file exists and its hash matches, False otherwise.
    """
    if not path.is_file():
        return False
    try:
        _, _, metadata = load_features(path)
    except (OSError, ValueError, KeyError):
        return False
    return metadata.get("hash") == config_hash(expected)


def load_split(
    encoder: str, dataset: str, split: str | None = None, feature_root: Path | None = None
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load a cached split by key instead of by path.

    Args:
        encoder: Encoder key.
        dataset: Dataset key.
        split: Official split, or ``None`` for the CLIP text prototypes.
        feature_root: Cache directory; defaults to :func:`default_feature_root`.

    Returns:
        The features, the labels and the metadata dict.
    """
    path = feature_path(encoder, dataset, split, feature_root)
    if not path.is_file():
        raise FileNotFoundError(f"No cached features at {path}. Run extract_all() first.")
    return load_features(path)
