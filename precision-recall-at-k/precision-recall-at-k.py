import numpy as np

def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    recommended = np.asarray(recommended, dtype=float)
    relevant = np.asarray(relevant, dtype=float)
    top_k = recommended[:k]

    top_k_relevant = np.intersect1d(top_k, relevant)
    n = len(top_k_relevant)
    return [n / k, n / len(relevant)]