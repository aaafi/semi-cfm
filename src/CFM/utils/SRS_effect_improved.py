from scipy.integrate import solve_ivp
import numpy as np
from scipy.io import loadmat
import time
from scipy.special import factorial

#similar to ode45 from matlab
def A_eff_2_D(f_mat, c, NA, a):
    y = np.pi * (a**2) * (0.65 + 1.619 * (c / (2 * np.pi * a * NA * f_mat))**1.5 + \
                          2.879 * (c / (2 * np.pi * a * NA * f_mat))**6)**2
    return y

def CRR(c, a, NA, f, f_p, f_p_r, C_R, f_C_R):

    crr = np.interp(f, f_C_R, C_R)
    scaling = ((f_p - f) / (f_p_r - f)) * \
              ((A_eff_2_D(f_p_r, c, NA, a) + A_eff_2_D(f_p_r - f, c, NA, a)) / \
               (A_eff_2_D(f_p, c, NA, a) + A_eff_2_D(f_p - f, c, NA, a)))
    crr = crr * scaling
    crr[np.isinf(crr) | np.isnan(crr)] = 0
    return crr

def vdp1(z, y, C, alpha, N):
    y = y * (y > 0)
    y_col = y.reshape(-1, 1) 
    dydz = +np.sum((y_col * C) * y_col.T, axis=1) - (alpha * y)
    return dydz

def vdp2(z, y, C, alpha, N):
    y = y * (y > 0)
    y_col = y.reshape(-1, 1) 
    dydz = -np.sum((y_col * C) * y_col.T, axis=1) + (alpha * y)
    return dydz


def SRS_effect_improved_FLP(c, a, NA, f_ref_raman, P_in_dBm_vector_in_z0, CCFV, f0, C_R, f_C_R, alpha_0_vec_freq_dep, deltaz, Ls_km):
    # a = float(a)
    # NA = float(NA)
    # f_ref_raman = float(f_ref_raman)
    alpha = np.array(alpha_0_vec_freq_dep).flatten()
    P_lin_W = np.array(P_in_dBm_vector_in_z0).flatten()
    N = len(P_lin_W)

    freq = f0 + np.array(CCFV).flatten()
    F1, F2 = np.meshgrid(freq, freq, indexing='xy')

    C = (F1 >= F2) * CRR(c, a, NA, F1 - F2, F1, f_ref_raman, C_R, f_C_R) - \
        (F2 > F1) * (F2 / F1) * CRR(c, a, NA, F2 - F1, F2, f_ref_raman, C_R, f_C_R)
    
    Ls_m = Ls_km * 1000
    t_eval = np.arange(0, Ls_m + 0.00001, deltaz)

    #solving the diff eq
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


def SRS_effect_improved_FRP(c, a, NA, f_ref_raman, P_in_dBm_vector_in_z0, CCFV, f0, C_R, f_C_R, alpha_0_vec_freq_dep, deltaz, Ls_km):
    a = float(a)
    NA = float(NA)
    f_ref_raman = float(f_ref_raman)
    alpha = np.array(alpha_0_vec_freq_dep).flatten()
    P_lin_W = np.array(P_in_dBm_vector_in_z0).flatten()
    N = len(P_lin_W)

    freq = f0 + np.array(CCFV).flatten()
    F1, F2 = np.meshgrid(freq, freq, indexing='xy')

    C = (F1 >= F2) * CRR(c, a, NA, F1 - F2, F1, f_ref_raman, C_R, f_C_R) - \
        (F2 > F1) * (F2 / F1) * CRR(c, a, NA, F2 - F1, F2, f_ref_raman, C_R, f_C_R)
    
    Ls_m = Ls_km * 1000
    t_eval = np.arange(0, Ls_m + 0.00001, deltaz)

    #solving the diff eq
    sol = solve_ivp(
        fun=vdp2, 
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