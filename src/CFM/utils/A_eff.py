import numpy as np
def A_eff(f_vec, c, NA_vec, a_vec):

    a = a_vec.T
    NA = NA_vec.T
    f = f_vec.T

    V_inv = c / (2*np.pi * a * NA * f)

    correction = (
        0.65
        + 1.619 * (V_inv**(3/2))
        + 2.879 * (V_inv**6)
    )

    y = np.pi * (a**2) * (correction**2)

    return y

def A_eff_2_D(f_mat, c, NA, a):
    # f_mat: frequency [Hz]
    v = c / (2 * np.pi * a * NA * f_mat)
    return np.pi * (a ** 2) * (0.65 + 1.619 * (v ** 1.5) + 2.879 * (v ** 6)) ** 2