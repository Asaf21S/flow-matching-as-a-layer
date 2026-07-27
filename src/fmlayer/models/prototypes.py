import numpy as np

EPSILON = 1e-12


def l2_normalize(features: np.ndarray) -> np.ndarray:
    """Scale every row to unit L2 norm.

    Args:
        features: Array of shape ``(num_items, dim)``.

    Returns:
        The row-normalised array, with zero rows left untouched.
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, EPSILON)


def image_prototypes(
    features: np.ndarray, labels: np.ndarray, num_classes: int
) -> np.ndarray:
    """Build one class prototype per class from labelled image features.

    Implements ``mu_c = normalize(mean_{i in S_c} normalize(z_i))`` as specified.

    Args:
        features: Image features of shape ``(num_items, dim)``.
        labels: Integer labels of shape ``(num_items,)``.
        num_classes: Number of classes; every class must appear at least once.

    Returns:
        Unit-norm prototypes of shape ``(num_classes, dim)``.
    """
    normalized = l2_normalize(features)
    sums = np.zeros((num_classes, normalized.shape[1]), dtype=np.float64)
    np.add.at(sums, labels, normalized)

    counts = np.bincount(labels, minlength=num_classes)
    if counts.min() == 0:
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"No features for classes {missing}; cannot build prototypes.")

    return l2_normalize(sums / counts[:, None]).astype(np.float32)


def cosine_similarities(features: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """Score every item against every class prototype.

    Args:
        features: Image features of shape ``(num_items, dim)``.
        prototypes: Class prototypes of shape ``(num_classes, dim)``.

    Returns:
        Cosine similarities of shape ``(num_items, num_classes)``.
    """
    return l2_normalize(features) @ l2_normalize(prototypes).T


def classify(features: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """Assign each item to its nearest class prototype in cosine similarity.

    Args:
        features: Image features of shape ``(num_items, dim)``.
        prototypes: Class prototypes of shape ``(num_classes, dim)``.

    Returns:
        Predicted labels of shape ``(num_items,)``.
    """
    return cosine_similarities(features, prototypes).argmax(axis=1)
