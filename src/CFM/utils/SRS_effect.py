from typing import Tuple
import numpy as np
from scipy.integrate import solve_ivp
from CFM.utils.A_eff import A_eff_2_D


def SRS_effect(y0: np.ndarray,
               frequencies: np.ndarray,
               attenuation_dB_km: np.ndarray,
               f_ref_raman: float,
               C_R: np.ndarray,
               f_C_R: np.ndarray,
               channel_spacing: float,
               deltaz: float,
               Ls_km: float,
               c: float,
               NA: float,
               a: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate forward channel power evolution under Inter-Channel Stimulated Raman Scattering (ISRS).

    Solves the coupled ordinary differential equations governing forward power transfer
    across WDM channels due to stimulated Raman scattering and fiber attenuation using
    a vectorized Runge-Kutta solver (RK45).

    Args:
    ---------
        y0 (np.ndarray):
            Initial channel launch powers at z = 0 in Watts.
        frequencies (np.ndarray):
            Channel carrier center frequencies in Hz.
        attenuation_dB_km (np.ndarray):
            Fiber attenuation coefficient vector per channel in dB/km.
        f_ref_raman (float):
            Reference Raman pump frequency in Hz.
        C_R (np.ndarray):
            Normalized reference Raman gain efficiency coefficients.
        f_C_R (np.ndarray):
            Frequency grid corresponding to the reference Raman gain coefficients in Hz.
        channel_spacing (float):
            Frequency spacing between adjacent WDM channels in Hz.
        deltaz (float):
            Spatial step size along the fiber link in kilometers.
        Ls_km (float):
            Total physical link length in kilometers.
        c (float):
            Speed of light in vacuum in m/s.
        NA (float):
            Numerical aperture of the fiber core.
        a (float):
            Fiber core radius in meters.

    Returns:
    ---------
        Tuple[np.ndarray, np.ndarray]:
            - z (np.ndarray): Array of spatial distance evaluation coordinates in kilometers.
            - Pout (np.ndarray): Spatial power evolution matrix across distance and channels
              in Watts with shape (N_z, N_channels).

    Example:
    ---------
    >>> import numpy as np
    >>> from CFM.utils.SRS_effect import SRS_effect
    >>> z, P_profile = SRS_effect(
    ...     y0=p_launch_watts,
    ...     frequencies=carrier_freqs,
    ...     attenuation_dB_km=alpha_db,
    ...     f_ref_raman=220e12,
    ...     C_R=raman_gain,
    ...     f_C_R=raman_freqs,
    ...     channel_spacing=50e9,
    ...     deltaz=0.2,
    ...     Ls_km=80.0,
    ...     c=299792458,
    ...     NA=0.1182,
    ...     a=4.3e-6
    ... )
    """
    y0 = np.array(y0).flatten()
    N = len(frequencies)

    A_eff = np.array([A_eff_2_D(f, c, NA, a) for f in frequencies])

    max_delta = N - 1
    A_eff_sample = np.zeros(max_delta + 1)
    for i in range(len(A_eff_sample)):
        f_sample = 219.96e12 - (i) * channel_spacing
        A_eff_sample[max_delta - i] = A_eff_2_D(f_sample, c, NA, a)

    required_offsets = np.arange(1, max_delta + 1) * channel_spacing

    if np.mean(f_C_R) < 0:
        query_points = -1 * required_offsets
    else:
        query_points = required_offsets

    sorted_idx = np.argsort(f_C_R)
    f_C_R_sorted = np.array(f_C_R)[sorted_idx]
    C_R_sorted = np.array(C_R)[sorted_idx]

    raman_gain_sample = np.interp(query_points, f_C_R_sorted, C_R_sorted, left=0, right=0)

    M = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if j < i:
                delta = i - j
                if delta <= len(raman_gain_sample) and delta < (len(A_eff_sample) - 1) and (i - delta) >= 0:
                    cr_tmp = raman_gain_sample[delta - 1]
                    cr = (cr_tmp * (frequencies[i - delta] /
                                    (f_ref_raman - channel_spacing * delta)) *
                          ((A_eff_sample[-1] + A_eff_sample[-1 - delta]) / 2) /
                          ((A_eff[i] + A_eff[i - delta]) / 2))
                    M[i, j] = -(frequencies[i] / frequencies[j]) * cr
            elif j > i:
                delta = j - i
                if delta <= len(raman_gain_sample) and delta < (len(A_eff_sample) - 1) and (j - delta) >= 0:
                    cr_tmp = raman_gain_sample[delta - 1]
                    cr = (cr_tmp * (frequencies[j - delta] /
                                    (f_ref_raman - channel_spacing * delta)) *
                          ((A_eff_sample[-1] + A_eff_sample[-1 - delta]) / 2) /
                          ((A_eff[j] + A_eff[j - delta]) / 2))
                    M[i, j] = cr

    alpha = np.array(attenuation_dB_km) / 4.343

    def ode_raman(z: float, y: np.ndarray) -> np.ndarray:
        return -alpha * y + y * (M @ y)

    z_span = (0, Ls_km)
    z_eval = np.arange(0, Ls_km + deltaz, deltaz)

    sol = solve_ivp(ode_raman, z_span, y0, t_eval=z_eval, method='RK45', vectorized=False)

    return sol.t, sol.y.T