import numpy as np

from src.fmlayer.models.prototypes import l2_normalize


def nearest_neighbors(
    queries: np.ndarray, gallery: np.ndarray, top_k: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Find the closest gallery rows to each query in cosine similarity.

    Args:
        queries: Query vectors of shape ``(num_queries, dim)``.
        gallery: Gallery vectors of shape ``(num_gallery, dim)``.
        top_k: Number of neighbours returned per query.

    Returns:
        Gallery indices of shape ``(num_queries, top_k)`` and their cosine similarities.
    """
    if top_k > len(gallery):
        raise ValueError(f"top_k={top_k} exceeds the {len(gallery)}-row gallery.")

    similarities = l2_normalize(queries) @ l2_normalize(gallery).T
    indices = np.argsort(-similarities, axis=1)[:, :top_k]
    return indices, np.take_along_axis(similarities, indices, axis=1)
