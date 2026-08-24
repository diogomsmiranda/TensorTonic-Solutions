import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Return the mean multiclass cross-entropy loss.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)
    N , K = y_pred.shape

    row_indices = np.arange(N) # just an array with all the indexs for later use [0,1,2,3...]
    p = y_pred[row_indices, y_true] # gets the probabilities assigned to the true class
    return np.mean(-np.log(p)) # mean negative log