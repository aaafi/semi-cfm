import numpy as np


def A_eff(f_vec: np.ndarray, c: float, NA_vec: np.ndarray, a_vec: np.ndarray) -> np.ndarray:
    """
    Calculate the effective mode area (A_eff) of a step-index optical fiber.

    Computes the effective cross-sectional area as a function of optical frequency
    using Marcuse's empirical Gaussian approximation for single-mode fiber mode field diameter.

    Args:
    ---------
        f_vec (np.ndarray): 
            Vector of optical frequencies in Hz.
        c (float): 
            Speed of light in vacuum in m/s.
        NA_vec (np.ndarray): 
            Numerical aperture vector of the fiber core.
        a_vec (np.ndarray): 
            Core radius vector of the fiber in meters.

    Returns:
    ---------
        np.ndarray: 
            Effective mode area (A_eff) in square meters (m^2).

    Example:
    ---------
    >>> import numpy as np
    >>> from CFM.utils.A_eff import A_eff
    >>> f = np.array([193.1e12, 193.2e12])
    >>> c = 299792458
    >>> NA = np.array([0.1182])
    >>> a = np.array([4.3e-6])
    >>> a_effective = A_eff(f, c, NA, a)
    """
    a = a_vec.T
    NA = NA_vec.T
    f = f_vec.T

    V_inv = c / (2 * np.pi * a * NA * f)

    correction = (
        0.65
        + 1.619 * (V_inv ** (3 / 2))
        + 2.879 * (V_inv ** 6)
    )

    y = np.pi * (a ** 2) * (correction ** 2)

    return y


def A_eff_2_D(f_mat: np.ndarray, c: float, NA: float, a: float) -> np.ndarray:
    """
    Calculate the effective mode area (A_eff) for 2D frequency arrays or scalar parameters.

    Provides the effective area evaluation over multidimensional optical frequency matrices
    based on Marcuse's empirical formulation.

    Args:
    ---------
        f_mat (np.ndarray): 
            Frequency grid or matrix in Hz.
        c (float): 
            Speed of light in vacuum in m/s.
        NA (float): 
            Numerical aperture of the fiber core.
        a (float): 
            Core radius of the fiber in meters.

    Returns:
    ---------
        np.ndarray: 
            Effective mode area matrix in square meters (m^2).

    Example:
    ---------
    >>> import numpy as np
    >>> from CFM.utils.A_eff import A_eff_2_D
    >>> f_grid = np.array([[193.1e12, 193.2e12], [193.3e12, 193.4e12]])
    >>> a_eff_grid = A_eff_2_D(f_grid, c=299792458, NA=0.1182, a=4.3e-6)
    """
    v = c / (2 * np.pi * a * NA * f_mat)
    return np.pi * (a ** 2) * (0.65 + 1.619 * (v ** 1.5) + 2.879 * (v ** 6)) ** 2