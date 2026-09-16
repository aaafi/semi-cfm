import numpy as np


def sinint_1(x: np.ndarray | float) -> np.ndarray:
    """
    Compute the sine integral function Si(x) using numerical approximation.

    Calculates the integral of sin(t)/t from 0 to x. Uses a linear approximation
    for small arguments (|x| < pi / 100) and uniform Riemann sum integration
    for larger values.

    Args:
    ---------
        x (np.ndarray | float):
            Input argument or array of arguments in radians.

    Returns:
    ---------
        np.ndarray:
            Calculated sine integral approximation evaluated element-wise.

    Example:
    ---------
    >>> import numpy as np
    >>> from CFM.utils.sinint import sinint_1
    >>> values = np.array([0.01, 1.5, 3.14])
    >>> si_vals = sinint_1(values)
    """
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)

    step = np.pi / 1000

    for i, val in enumerate(x):
        if abs(val) < np.pi / 100:
            result[i] = val
        else:
            t = np.arange(step, abs(val) + step, step)
            integral = step * (1 + np.sum(np.sin(t) / t))
            result[i] = np.sign(val) * integral

    return result