import numpy as np

def sinint_1(x):
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