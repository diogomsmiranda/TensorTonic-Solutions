def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Return final x after 'steps' iterations.
    """
    if steps == 0:
        return x0
    else:
        # x_t+1 = x_t - lr * f'(x_t) where f'(x_t) = 2ax + b
        x0 = x0 - lr * (2 * a * x0 + b)
        steps -= 1
        return gradient_descent_quadratic(a,b,c,x0,lr,steps)
    pass