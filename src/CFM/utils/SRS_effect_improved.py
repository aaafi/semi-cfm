from typing import Tuple
import numpy as np
from scipy.integrate import solve_ivp


def A_eff_2_D(f_mat: np.ndarray, c: float, NA: float, a: float) -> np.ndarray:
    """
    Calculate the effective cross-sectional mode area (A_eff) of the fiber.

    Uses Marcuse's empirical Gaussian approximation to estimate the mode 
    field diameter and effective area across optical frequencies.

    Args:
    ---------
        f_mat (np.ndarray): 
            Optical frequency grid or matrix in Hz.
        c (float): 
            Speed of light in vacuum in m/s.
        NA (float): 
            Numerical aperture of the fiber core.
        a (float): 
            Core radius of the fiber in meters.

    Returns:
    ---------
        np.ndarray: 
            Calculated effective mode area in square meters (m^2).

    Example:
    ---------
    >>> from CFM.utils.SRS_effect_improved import A_eff_2_D
    >>> import numpy as np
    >>> freqs = np.array([193.1e12, 193.2e12])
    >>> a_eff = A_eff_2_D(freqs, c=299792458, NA=0.1182, a=4.3e-6)
    """
    y = np.pi * (a**2) * (0.65 + 1.619 * (c / (2 * np.pi * a * NA * f_mat))**1.5 +
                          2.879 * (c / (2 * np.pi * a * NA * f_mat))**6)**2
    return y


def CRR(c: float,
        a: float,
        NA: float,
        f: np.ndarray,
        f_p: np.ndarray,
        f_p_r: float,
        C_R: np.ndarray,
        f_C_R: np.ndarray) -> np.ndarray:
    """
    Compute the frequency-dependent and effective-area-scaled Raman gain coefficient.

    Interpolates the normalized Raman gain spectrum and applies frequency-dependent
    effective area scaling factors between pump and probe channels.

    Args:
    ---------
        c (float): 
            Speed of light in vacuum in m/s.
        a (float): 
            Fiber core radius in meters.
        NA (float): 
            Numerical aperture of the fiber core.
        f (np.ndarray): 
            Frequency separation matrix |f_pump - f_probe| in Hz.
        f_p (np.ndarray): 
            Pump frequency matrix in Hz.
        f_p_r (float): 
            Reference Raman pump frequency in Hz.
        C_R (np.ndarray): 
            Reference Raman gain efficiency spectrum array.
        f_C_R (np.ndarray): 
            Frequency offsets corresponding to reference Raman gain data in Hz.

    Returns:
    ---------
        np.ndarray: 
            Scaled Raman gain efficiency matrix in 1/(W*m).
    """
    crr = np.interp(f, f_C_R, C_R)
    scaling = ((f_p - f) / (f_p_r - f)) * \
              ((A_eff_2_D(f_p_r, c, NA, a) + A_eff_2_D(f_p_r - f, c, NA, a)) /
               (A_eff_2_D(f_p, c, NA, a) + A_eff_2_D(f_p - f, c, NA, a)))
    crr = crr * scaling
    crr[np.isinf(crr) | np.isnan(crr)] = 0
    return crr


def vdp1(z: float, y: np.ndarray, C: np.ndarray, alpha: np.ndarray, N: int) -> np.ndarray:
    """
    Ordinary differential equation (ODE) modeling forward power evolution under ISRS.

    Evaluates spatial derivative dy/dz along the fiber coordinate z, accounting
    for multi-channel Raman power exchange and linear attenuation.

    Args:
    ---------
        z (float): 
            Current longitudinal position along the fiber in meters.
        y (np.ndarray): 
            Channel optical power array at coordinate z in Watts.
        C (np.ndarray): 
            Raman cross-coupling coefficient matrix of shape (N, N).
        alpha (np.ndarray): 
            Attenuation coefficient vector per channel in 1/m.
        N (int): 
            Total number of optical channels.

    Returns:
    ---------
        np.ndarray: 
            Spatial derivatives dy/dz representing power evolution in W/m.
    """
    y = y * (y > 0)
    y_col = y.reshape(-1, 1)
    dydz = +np.sum((y_col * C) * y_col.T, axis=1) - (alpha * y)
    return dydz


def SRS_effect_improved_FLP(c: float,
                            a: float,
                            NA: float,
                            f_ref_raman: float,
                            P_in_dBm_vector_in_z0: np.ndarray,
                            CCFV: np.ndarray,
                            f0: float,
                            C_R: np.ndarray,
                            f_C_R: np.ndarray,
                            alpha_0_vec_freq_dep: np.ndarray,
                            deltaz: float,
                            Ls_km: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate forward power propagation under Inter-Channel Stimulated Raman Scattering.

    Integrates the forward nonlinear coupled differential equations (Forward Launch
    Power - FLP) along a fiber span using a high-precision Runge-Kutta solver.

    Args:
    ---------
        c (float): 
            Speed of light in vacuum in m/s.
        a (float): 
            Fiber core radius in meters.
        NA (float): 
            Numerical aperture of the fiber.
        f_ref_raman (float): 
            Reference Raman pump frequency in Hz.
        P_in_dBm_vector_in_z0 (np.ndarray): 
            Input launch power vector in Watts per channel at span input (z = 0).
        CCFV (np.ndarray): 
            Vector of channel carrier frequency offsets relative to f0 in Hz.
        f0 (float): 
            Reference center frequency in Hz.
        C_R (np.ndarray): 
            Reference Raman gain curve array.
        f_C_R (np.ndarray): 
            Frequency grid of reference Raman gain curve in Hz.
        alpha_0_vec_freq_dep (np.ndarray): 
            Frequency-dependent fiber loss vector in 1/m.
        deltaz (float): 
            Spatial sampling interval in meters.
        Ls_km (float): 
            Total fiber span length in kilometers.

    Returns:
    ---------
        Tuple[np.ndarray, np.ndarray]:
            - Pout_dBm (np.ndarray): Power evolution array across distance and channels
              in Watts of shape (N_z, N_channels).
            - z (np.ndarray): Spatial coordinate array along the span in meters.

    Example:
    ---------
    >>> from CFM.utils.SRS_effect_improved import SRS_effect_improved_FLP
    >>> P_profile, z_axis = SRS_effect_improved_FLP(
    ...     c=299792458,
    ...     a=4.3e-6,
    ...     NA=0.1182,
    ...     f_ref_raman=220e12,
    ...     P_in_dBm_vector_in_z0=p_launch_watts,
    ...     CCFV=carrier_offsets,
    ...     f0=193.5e12,
    ...     C_R=raman_gain,
    ...     f_C_R=raman_freqs,
    ...     alpha_0_vec_freq_dep=alpha_m,
    ...     deltaz=200,
    ...     Ls_km=80.0
    ... )
    """
    alpha = np.array(alpha_0_vec_freq_dep).flatten()
    P_lin_W = np.array(P_in_dBm_vector_in_z0).flatten()
    N = len(P_lin_W)

    freq = f0 + np.array(CCFV).flatten()
    F1, F2 = np.meshgrid(freq, freq, indexing='xy')

    C = (F1 >= F2) * CRR(c, a, NA, F1 - F2, F1, f_ref_raman, C_R, f_C_R) - \
        (F2 > F1) * (F2 / F1) * CRR(c, a, NA, F2 - F1, F2, f_ref_raman, C_R, f_C_R)
    
    Ls_m = Ls_km * 1000
    t_eval = np.arange(0, Ls_m + 0.00001, deltaz)

    sol = solve_ivp(
        fun=vdp1, 
        t_span=(0, Ls_m), 
        y0=P_lin_W, 
        t_eval=t_eval, 
        method='RK45', 
        args=(C, alpha, N), 
        rtol=1e-12, 
        atol=1e-22
    )

    Pout_dBm = sol.y.T
    Pout_dBm[Pout_dBm <= 0] = 1e-300
    
    z = sol.t
    return Pout_dBm, z