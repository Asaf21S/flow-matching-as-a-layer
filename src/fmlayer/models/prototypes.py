import numpy as np

EPSILON = 1e-12


def l2_normalize(features: np.ndarray) -> np.ndarray:
    """Scale every row to unit L2 norm."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, EPSILON)


def image_prototypes(
    features: np.ndarray, labels: np.ndarray, num_classes: int
) -> np.ndarray:
    """Build one class prototype per class from labelled image features."""
    normalized = l2_normalize(features)
    sums = np.zeros((num_classes, normalized.shape[1]), dtype=np.float64)
    np.add.at(sums, labels, normalized)

    counts = np.bincount(labels, minlength=num_classes)
    if counts.min() == 0:
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"No features for classes {missing}; cannot build prototypes.")

    return l2_normalize(sums / counts[:, None]).astype(np.float32)


def cosine_similarities(features: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """Score every item against every class prototype."""
    return l2_normalize(features) @ l2_normalize(prototypes).T


def classify(features: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """Assign each item to its nearest class prototype in cosine similarity."""
    return cosine_similarities(features, prototypes).argmax(axis=1)
