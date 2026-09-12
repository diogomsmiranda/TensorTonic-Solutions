def f1_micro(y_true: list[int], y_pred: list[int]) -> float:
    """
    Returns the micro-averaged F1 score as a Python float rounded to four decimals.
    """
    tp = sum (actual == predicted for actual, predicted in zip(y_true, y_pred))
    errors = len(y_true) - tp
    denominator = 2 * tp + 2 * errors
    return round(2 * tp / denominator, 4)