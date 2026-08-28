import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    
    log_p = np.log2(p, out=np.zeros_like(p), where=(p > 0))
    return float(-np.sum(p * log_p))