import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    n = X.shape[0]
    w = np.zeros(X.shape[1])
    b = 0.0

    for _ in range(steps):
        p = _sigmoid(X @ w + b)
        
        gradient_w = X.T @ (p-y) / n
        gradient_b = np.mean(p-y)

        w = w - lr * gradient_w
        b = b - lr * gradient_b
    
    return w, b