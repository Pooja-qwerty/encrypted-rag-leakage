"""Public search arithmetic; contains no SDK, keys, or corpus access."""

import numpy as np

METRICS = {"cosine", "dot_product", "l2"}


def score_vectors(index, query, metric="cosine"):
    matrix, vector = np.asarray(index, dtype=float), np.asarray(query, dtype=float)
    if (matrix.ndim != 2 or vector.ndim != 1 or not matrix.shape[0]
            or matrix.shape[1] != len(vector) or not len(vector)
            or not np.isfinite(matrix).all() or not np.isfinite(vector).all()):
        raise ValueError("Expected a finite index matrix and matching query vector")
    if metric == "cosine":
        denominator = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector)
        return np.divide(matrix @ vector, denominator, out=np.zeros(len(matrix)),
                         where=denominator > 0)
    if metric == "dot_product":
        return matrix @ vector
    if metric == "l2":
        return -np.sum((matrix - vector) ** 2, axis=1)
    raise ValueError("Metric must be cosine, dot_product, or l2")


def rank_scores(scores, record_ids, k):
    """Phase 2 tie rule: descending score, then ascending opaque record ID."""
    scores = np.asarray(scores, dtype=float)
    if (scores.ndim != 1 or len(scores) != len(record_ids) or not np.isfinite(scores).all()
            or type(k) is not int or not 1 <= k <= len(scores)):
        raise ValueError("Invalid scores or k")
    return np.lexsort((np.asarray(record_ids), -scores))[:k].tolist()
