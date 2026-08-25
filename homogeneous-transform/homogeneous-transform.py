import numpy as np

def apply_homogeneous_transform(T: list, points: list) -> np.ndarray:
    """
    Returns transformed points with shape (3,) or (N, 3).
    """
    T = np.asarray(T, dtype=float)
    points = np.asarray(points, dtype=float)

    if points.ndim > 1:
        ones_col = np.ones((points.shape[0], 1))
        points = np.hstack((points, ones_col))
    else:
        points = np.append(points, [1])

    
    return (T @ points.T)[:-1].T