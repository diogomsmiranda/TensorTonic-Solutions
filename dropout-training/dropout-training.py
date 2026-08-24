import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.asarray(x, dtype=float)
    
    keep = 1 - p
    if rng:
        mask = (rng.random(x.shape) < keep).astype(float) / keep
    else:
        mask = (np.random.random(x.shape) < keep).astype(float) / keep

    return (x * mask, mask)