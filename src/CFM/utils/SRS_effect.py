import numpy as np
from scipy.integrate import solve_ivp

def SRS_effect(y0, frequencies, attenuation_dB_km, f_ref_raman,
                C_R, f_C_R, channel_spacing, deltaz, Ls_km, c, NA, a):
    """
    Complete SRS solver with vectorized ODE function in Python.
    Equivalent of the MATLAB version using SciPy's solve_ivp.
    """

    y0 = np.array(y0).flatten()
    N = len(frequencies)

    # =====================================================================
    # DYNAMIC EFFECTIVE AREA CALCULATION (Marcuse Formula)
    # =====================================================================
    def A_eff_2_D(f_mat, c, NA, a):
        # f_mat: frequency [Hz]
        v = c / (2 * np.pi * a * NA * f_mat)
        return np.pi * (a ** 2) * (0.65 + 1.619 * (v ** 1.5) + 2.879 * (v ** 6)) ** 2

    A_eff = np.array([A_eff_2_D(f, c, NA, a) for f in frequencies])

    # A_eff_sample grid
    max_delta = N - 1
    A_eff_sample = np.zeros(max_delta + 1)
    for i in range(len(A_eff_sample)):
        f_sample = 219.96e12 - (i) * channel_spacing
        A_eff_sample[max_delta - i] = A_eff_2_D(f_sample, c, NA, a)

    # =====================================================================
    # Raman gain mapping via linear interpolation
    # =====================================================================
    required_offsets = np.arange(1, max_delta + 1) * channel_spacing

    if np.mean(f_C_R) < 0:
        query_points = -1 * required_offsets
    else:
        query_points = required_offsets

    sorted_idx = np.argsort(f_C_R)
    f_C_R_sorted = np.array(f_C_R)[sorted_idx]
    C_R_sorted = np.array(C_R)[sorted_idx]

    raman_gain_sample = np.interp(query_points, f_C_R_sorted, C_R_sorted, left=0, right=0)

    # =====================================================================
    # Precompute Interaction Matrix M and Attenuation for Vectorized ODE
    # =====================================================================
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

    # Ensure attenuation is a NumPy array for element-wise operations
    alpha = np.array(attenuation_dB_km) / 4.343

    # =====================================================================
    # Define the ODE system
    # =====================================================================
    def ode_raman(z, y):
        # Vectorized implementation: 
        # dydz_i = -alpha_i * y_i + y_i * sum_j(M_ij * y_j)
        return -alpha * y + y * (M @ y)

    # =====================================================================
    # Solve ODE with SciPy
    # =====================================================================
    z_span = (0, Ls_km)
    z_eval = np.arange(0, Ls_km + deltaz, deltaz)

    sol = solve_ivp(ode_raman, z_span, y0, t_eval=z_eval, method='RK45', vectorized=False)
    #solving the diff eq
    return sol.t, sol.y.T
