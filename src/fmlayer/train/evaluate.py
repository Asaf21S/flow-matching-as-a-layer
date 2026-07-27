import numpy as np


def top1_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Compute top-1 accuracy.

    Args:
        predictions: Predicted labels of shape ``(num_items,)``.
        labels: Ground-truth labels of shape ``(num_items,)``.

    Returns:
        The fraction of correct predictions, in ``[0, 1]``.
    """
    if predictions.shape != labels.shape:
        raise ValueError(f"Shape mismatch: {predictions.shape} vs {labels.shape}")
    return float((predictions == labels).mean())


def per_class_accuracy(
    predictions: np.ndarray, labels: np.ndarray, num_classes: int
) -> np.ndarray:
    """Compute accuracy within each class.

    Args:
        predictions: Predicted labels of shape ``(num_items,)``.
        labels: Ground-truth labels of shape ``(num_items,)``.
        num_classes: Number of classes.

    Returns:
        Per-class accuracy of shape ``(num_classes,)``; NaN for absent classes.
    """
    correct = np.bincount(labels, weights=(predictions == labels), minlength=num_classes)
    counts = np.bincount(labels, minlength=num_classes)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, correct / counts, np.nan)


def confusion_matrix(
    predictions: np.ndarray, labels: np.ndarray, num_classes: int, normalize: bool = True
) -> np.ndarray:
    """Build the confusion matrix of a prediction set.

    Args:
        predictions: Predicted labels of shape ``(num_items,)``.
        labels: Ground-truth labels of shape ``(num_items,)``.
        num_classes: Number of classes.
        normalize: Divide each row by its true-class count.

    Returns:
        Matrix of shape ``(num_classes, num_classes)`` with true classes on the rows.
    """
    matrix = np.zeros((num_classes, num_classes), dtype=np.float64)
    np.add.at(matrix, (labels, predictions), 1.0)
    if normalize:
        counts = matrix.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            matrix = np.where(counts > 0, matrix / counts, np.nan)
    return matrix
