import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    if x is None:
        return None

    x_arr = np.asarray(x, dtype=float)
    return 1 / (1 + np.exp(-x_arr))