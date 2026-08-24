import numpy as np

def softmax(x: list) -> np.ndarray:
    """Return stable softmax probabilities with the same shape as x."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        m = np.max(x)
        x_m = x-m
        return np.exp(x_m) / np.sum(np.exp(x_m))
    m = np.max(x,axis=1, keepdims=True)
    x_m = x-m
    return np.exp(x_m) / np.sum(np.exp(x_m), axis=1, keepdims=True)
    