import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    rows = len(A)
    cols = len(A[0])
    # Write code here
    A_t = np.zeros((cols, rows))
    for i in range(rows):
        for j in range(cols):
            A_t[j][i] = A[i][j]
    return A_t