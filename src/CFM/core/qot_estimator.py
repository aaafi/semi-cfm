import numpy as np
from scipy.io import loadmat
import math
from scipy.special import factorial
from scipy.special import erfcinv
from CFM.utils.SRS_effect_improved import SRS_effect_improved_FLP, SRS_effect_improved_FRP
from CFM.utils.SRS_effect import SRS_effect
from CFM.utils.calc_opt_all_param_ISRS import calc_opt_all_param_ISRS
from CFM.utils.A_eff import A_eff, A_eff_2_D
from CFM.utils.sinint import sinint_1
from CFM.utils.constant.physical import SPEED_OF_LIGHT
from CFM.utils.constant.model import DEFAULT_CHANNEL_BAUD
from CFM.utils.constant.units import DBM_TO_WATT, THZ_TO_HZ
from CFM.utils.constant.raman import F_REF_RAMAN

workspace_Raman_Gain_effi = loadmat('..\.\data\workspace_Raman_Gain_effi.mat')


class ParameterBuilder:
   """
   Responsible for initializing and building all parameter arrays exactly 
   as done in the original CFM constructor. 
   """

   def __init__(self, 
                link,
                bands,
                P_in,
                grid_center,
                workspace_Raman_Gain_effi=workspace_Raman_Gain_effi,
                alpha_dB_LCS=np.full((1, 1), 0.2),
                max_input_limiation=6,
                PHI = 13/21,
                G_booster_dB = 15.0,  # <-- Added Booster Gain
                F_booster_dB = 5.5):  # <-- Added Booster Noise Figure

      self.link = link
      self.bands = bands
      self.alpha_dB_LCS = alpha_dB_LCS
      self.grid_center = grid_center
      self.workspace_Raman_Gain_effi = workspace_Raman_Gain_effi
      self.P_in = P_in
      self.max_input_limiation = max_input_limiation
      self.PHI = PHI
      
      self.G_booster_dB = G_booster_dB
      self.F_booster_dB = F_booster_dB

      self._build()

      self.link = link
      self.bands = bands
      self.alpha_dB_LCS = alpha_dB_LCS
      self.grid_center = grid_center
      self.workspace_Raman_Gain_effi = workspace_Raman_Gain_effi
      self.P_in = P_in
      self.max_input_limiation = max_input_limiation
      self.PHI = PHI

      self._build()

   def _build(self):

      #self.N_ss = self.link.num_span
      self.N_ss = self.link.num_span #We assume all spans length are equal
      self.N_c = len(self.grid_center)
      self.nuu = self.link.link_params.nuu

      # alpha vector
      if len(self.alpha_dB_LCS[0]) == 1:
         self.alpha_dB_per_km_vec = np.full((1, self.N_c), self.alpha_dB_LCS[0][0])
      else:
         if len(self.alpha_dB_LCS[0]) != self.N_c:
               raise TypeError
         else:
               self.alpha_dB_per_km_vec = self.alpha_dB_LCS

      self.alpha_dB_per_km_vec = self.alpha_dB_per_km_vec * np.ones((self.N_ss, 1))
      # General parameters
      self.ro_coh = 0
      self.ro_MCI = 0
      self.ro_SCI = 1
      self.ro_XCI = 1
      self.ISRS = 1
      if self.ISRS == 1:
         self.ro_MCI = 0

      self.lambda_nm = self.link.link_params.lambda_nm

      # Fiber link parameter vectorization (exactly same as in CFM)
      self.n2_vec = self.link.link_params.factor_gamma * self.link.link_params.n2 * np.ones((1, self.N_ss))
      self.n2 = self.link.link_params.n2
      self.NA_vec = self.link.link_params.NA * np.ones((1, self.N_ss))
      self.a_vec = self.link.link_params.r * np.ones((1, self.N_ss))
      self.gamma_per_wat_per_km_vec = self.link.link_params.gamma_per_wat_per_km * np.ones((1, self.N_ss))
      self.betta2_ps_squared_per_km = self.link.link_params.betta2_ps_squared_per_km
      self.betta3_ps_cube_per_km = self.link.link_params.betta3_ps_cube_per_km
      self.betta4_ps_4_per_km = self.link.link_params.betta4_ps_4_per_km * np.ones((1, self.N_ss))
      self.D_ps_per_nm_per_km_vec = self.link.link_params.D * 1e6
      self.betta_DCU_ps_square = 0 * np.ones((1, self.N_ss))

      # Roll-off and baud rate
      self.roll_off_vec = np.concatenate([b.opt_params.rof * np.ones(b.num_channels) for b in self.bands])
      self.CH_BR_vec = 64e9 * np.ones((1, self.N_c))
      self.c = self.link.link_params.c
      self.CCFV = self.grid_center - self.nuu
      self.SCFV = self.CCFV - 0.5 * self.CH_BR_vec * (1 + self.roll_off_vec)
      self.ECFV = self.CCFV + 0.5 * self.CH_BR_vec * (1 + self.roll_off_vec)
      self.nuu_vec = self.grid_center

      # Modulation format vectors
      self.PHI = self.PHI * np.ones((1, self.N_c))
      self.SAI = (-5548 / 3087) * np.ones((1, self.N_c))

      # Raman gain parameters
      self.f_ref_raman =  220e12 * np.ones((1, self.N_ss))
      self.C_r_max = 1 * 4.562e-4 * np.ones((1, self.N_ss))
      self.RGE = self.workspace_Raman_Gain_effi['RGE']
      self.freq_Thz = self.workspace_Raman_Gain_effi['freq_Thz']
      self.f_C_R = self.freq_Thz * 1e12
      self.C_R = self.C_r_max.transpose() * self.RGE
      self.deltaz = 200
      self.m_pow = 2
      self.opt_loop = 6

      # Noise
      self.noise_figure_LCS = np.concatenate([b.noise_figure * np.ones(b.num_channels) for b in self.bands])
      self.F11_dB = self.noise_figure_LCS
      self.F_dB = np.ones((self.N_ss, 1)) * self.F11_dB

      # Initialize result arrays
      self.z_max_pos = np.empty((self.N_ss,), dtype="float")
      self.Loss_dB = np.empty((self.N_ss, self.N_c), dtype="float")
      self.alfa_0 = np.empty((self.N_ss, self.N_c), dtype="float")
      self.alfa_1 = np.empty((self.N_ss, self.N_c), dtype="float")
      self.sigma = np.empty((self.N_ss, self.N_c), dtype="float")
      self.Gain_dB_vec = np.empty((self.N_ss, self.N_c), dtype="float")
      self.G_tot_dB = np.empty((self.N_ss, self.N_c), dtype="float")

      self.L_s_km_vec = self.link.length
      self.P_in_dBm = self.P_in* np.ones((self.N_ss + 1, self.N_c))
      if isinstance(self.P_in, (float, np.floating)):
         self.P_in_dBm = self.P_in * np.ones((self.N_ss + 1, self.N_c))

   def reset(self):
      if isinstance(self.P_in, (float, np.floating)):
         self.P_in_dBm = self.P_in * np.ones((self.N_ss + 1, self.N_c))
       

class MultiCore_Parameters:

   def __init__(
         self,
         link,
         grid_center,
         N_c,
         core_pitch,
         adjacent_core_vec,
         r_bending=140e-3,
         a1=4.5e-6
      ):

      # external inputs
      self.link = link
      self.grid_center = grid_center
      self.N_c = N_c
      self.Core_pitch = core_pitch
      self.adjacent_core_vec = adjacent_core_vec
      self.r_bending = r_bending
      self.a1 = a1

      # build all internal parameters
      self._build()


   def _build(self):

      # -------------------------------
      # General physical parameters
      # -------------------------------
      self.BER = 1.5e-2
      self.Q_factor_BER = 20*np.log10(np.sqrt(2)*erfcinv(2*self.BER))

      self.snr_penalty = 1
      self.etta_MF = np.array([0.5, 1, 2/(2-np.sqrt(2)), 5, 10, 21])

      self.ICXT_th_1dB = 10*np.log10(
         (1 - 10**(-self.snr_penalty/10)) /
         (self.etta_MF * 10**(self.Q_factor_BER/10))
      )

      # -------------------------------
      # Fiber geometry
      # -------------------------------
      self.n1 = 1.45
      self.a2 = 2*self.a1
      self.a3 = 3*self.a1
      self.wtr = 1.5*self.a1
      L_s_km_vec = self.link.length[0]

      # -------------------------------
      # Allocate arrays
      # -------------------------------
      self.ic_xt_dB = np.zeros((len(self.adjacent_core_vec), self.N_c))
      self.ic_xt = np.zeros((len(self.adjacent_core_vec), self.N_c))

      # -------------------------------
      # Compute core-to-core coupling
      # -------------------------------
      beta = 2*np.pi*self.grid_center / (self.n1 * 3e8)
      k = 2*np.pi*self.grid_center / (3e8)

      Core_pitch = self.Core_pitch

      Delta1 = 0.35*self.n1 / 100

      V1 = k*self.a1*self.n1*np.sqrt(2*Delta1)

      Delta_W21 = 0.607*V1 + 0.608
      W2 = 1.750*V1 - 0.388
      W1 = W2 - Delta_W21

      GAMMA = W1 / (W1 + (Delta_W21*self.wtr/Core_pitch))

      U1_power2 = self.a1**2 * (k**2*self.n1**2 - beta**2)
      K1W1 = np.sqrt(np.pi/(2*W1)) * np.exp(-W1)

      k1 = np.sqrt(GAMMA)*np.sqrt(Delta1)/self.a1
      k2 = U1_power2 / ((V1**3) * (K1W1**2))
      k3 = np.sqrt((np.pi*self.a1)/(W1*Core_pitch))
      k4 = np.exp(-(W1*Core_pitch + 2*Delta_W21*self.wtr)/self.a1)

      self.kappa = k1*k2*k3*k4

      # XT per unit length
      self.h_XT = (
         2*self.kappa**2 * self.r_bending / (beta*Core_pitch)
      ) * np.ones((len(self.adjacent_core_vec), self.N_c))

      self.h_XT_dBkm = 10*np.log10(self.h_XT * 1e3)

      adj = np.array(self.adjacent_core_vec)[:, None]        # (5,1)
      L = float(L_s_km_vec) * np.ones((1, self.N_c)) * 1e6   # shape (1, 268)

      self.ic_xt = adj * self.h_XT * L
      self.ic_xt_dB = 10 * np.log10(self.ic_xt)


    

class ISRSSolver:
   """
   Calculator for ISRS parameters — identical to CFM.calc_ISRS_params()
   """

   def __init__(self, params, model='FLP'):
      self.p = params
      self.model = model

   def solve(self):
      self.p.reset()
      for nn in range(self.p.N_ss):

         alpha_0_vec_freq_dep = (self.p.alpha_dB_per_km_vec[nn, :]) / (20000 * np.log10(np.exp(1)))

         if(self.model=='FLP'):

            Pout_W_00, z_dis = SRS_effect_improved_FLP(
                  self.p.c,
                  self.p.a_vec[0, nn],
                  self.p.NA_vec[0, nn],
                  self.p.f_ref_raman[0, nn],
                  10 ** (self.p.P_in_dBm[nn, :] / 10) * 1e-3,
                  self.p.CCFV,
                  self.p.nuu,
                  self.p.C_R[0],
                  self.p.f_C_R[0],
                  2 * alpha_0_vec_freq_dep,
                  self.p.deltaz,
                  self.p.L_s_km_vec[0]
            )
            Pout_dBm_00 = 10*np.log10(Pout_W_00 * 1e3)  # infact it is P_in
         
         elif(self.model=='FRP'):

            Pout_W_00, z_dis = SRS_effect_improved_FRP(
                  self.p.c,
                  self.p.a_vec[0, nn],
                  self.p.NA_vec[0, nn],
                  self.p.f_ref_raman[0, nn],
                  10 ** (self.p.P_in_dBm[nn, :] / 10) * 1e-3,
                  self.p.CCFV,
                  self.p.nuu,
                  self.p.C_R[0],
                  self.p.f_C_R[0],
                  2 * alpha_0_vec_freq_dep,
                  self.p.deltaz,
                  self.p.L_s_km_vec[0]
            )
            self.p.P_in_dBm[self.p.N_ss, :] = 10 * np.ones((1, self.p.N_c))
            Pout_dBm_00 = np.flip(10 * np.log10(Pout_W_00 * 1e3), axis=0)

            if max(Pout_dBm_00[0, :]) > self.p.max_input_limiation:
                  print('Calculated input power is larger than max input limitation.')
                  return None, None, None

         P_out_dBm = Pout_dBm_00[:, 0:self.p.N_c]
         self.p.P_in_dBm[nn, :] = P_out_dBm[0, :]

         z_max = np.empty((self.p.N_c,), dtype="int")
         for hh1 in range(self.p.N_c):
               z_max[hh1] = int(z_dis[min(np.argwhere(P_out_dBm[:, hh1] == max(P_out_dBm[:, hh1])))])
         self.p.z_max_pos[nn] = np.mean(z_max)

         self.p.Loss_dB[nn, :] = (self.p.P_in_dBm[nn, :] - P_out_dBm[-1, :])

         su1, su2, su3 = calc_opt_all_param_ISRS(
               alpha_0_vec_freq_dep[0:(len(self.p.CCFV))],
               self.p.deltaz,
               P_out_dBm,
               self.p.N_c,
               z_dis,
               self.p.m_pow,
               self.p.opt_loop
         )

         self.p.alfa_0[nn, :] = su1[0:self.p.N_c]
         self.p.alfa_1[nn, :] = su2[0:self.p.N_c]
         self.p.sigma[nn, :] = su3[0:self.p.N_c]
         self.p.Gain_dB_vec[nn, :] = (self.p.P_in_dBm[nn + 1, :] - P_out_dBm[-1, :])

         if nn == 0:
               self.p.G_tot_dB[nn, :] = self.p.Gain_dB_vec[nn, :] - self.p.Loss_dB[nn, :]
         else:
               self.p.G_tot_dB[nn, :] = self.p.G_tot_dB[nn - 1, :] + self.p.Gain_dB_vec[nn, :] - self.p.Loss_dB[nn, :]

      return Pout_dBm_00, self.p.alfa_0, self.p.alfa_1, self.p.sigma
   
   def solve_Online(self):
      z_dis, Pout_W_00 = SRS_effect(
         self.p.P_in,
         self.p.grid_center,
         self.p.alpha_dB_per_km_vec[0],
         self.p.f_ref_raman[0],
         self.p.C_R[0]*1000,
         self.p.f_C_R[0],
         self.p.bands[0].channel_spacing*1e12,
         self.p.deltaz/1000,
         self.p.L_s_km_vec,
         300000000,
         self.p.NA_vec[0],
         self.p.a_vec[0]
      )      

      alpha_0_vec_freq_dep = (self.p.alpha_dB_per_km_vec[0]) / (20000 * np.log10(np.exp(1)))
      P_out_dBm = 10*np.log10(Pout_W_00 * 1e3)
      su1, su2, su3 = calc_opt_all_param_ISRS(
            alpha_0_vec_freq_dep[0:(len(self.p.CCFV))],
            self.p.deltaz,
            P_out_dBm,
            self.p.N_c,
            z_dis*1000,
            self.p.m_pow,
            self.p.opt_loop
      )

      self.p.alfa_0[0, :] = su1[0:self.p.N_c]
      self.p.alfa_1[0, :] = su2[0:self.p.N_c]
      self.p.sigma[0, :] = su3[0:self.p.N_c]

      return self.p.alfa_0, self.p.alfa_1, self.p.sigma

class NLISolver:

   def __init__(self, params, alfa_0, alfa_1, sigma):
      self.p = params
      self.alfa_0 = alfa_0
      self.alfa_1 = alfa_1
      self.sigma = sigma

   def solve(self):

      p = self.p

      N_s_max = p.N_ss
      N_c = p.N_c
      ro_SCI = p.ro_SCI
      ro_XCI = p.ro_XCI

      G_NLI = np.zeros((N_s_max, N_c))

      # ---- model coefficients ----
      a = np.array([
      +1.0436e0,-1.1878e0,+1.0573e0,-1.8309e+1,
      +1.6665e0,-1.0020e0,+9.0933e0,+6.6420e-3,
      +8.4481e-1,-1.8530e0,+9.4539e-1,-1.5421e+1,
      +1.0229e0,-1.1440e0,+1.1393e-2,+3.8070e+5,
      +1.4785e+3,-2.2593e0,-6.7997e-1,+2.0215e0,
      -2.9781e-1,+5.5130e-1,-3.6718e-1,+1.1486e0
      ])

      if p.ISRS != 1:
         return G_NLI

      # ---- unit conversion ----
      lambda1 = p.lambda_nm * 1e-9
      L_s = (np.asarray(p.L_s_km_vec) * 1000).reshape(-1, 1)

      D = p.D_ps_per_nm_per_km_vec * 1e-6

      betta2 = np.array([-(lambda1**2) * D / (2*np.pi*p.c)]).reshape(-1,1)*np.ones((N_s_max, 1))
      betta3 = np.array([p.betta3_ps_cube_per_km * 1e-39]).reshape(-1,1)*np.ones((N_s_max, 1))
      betta4 = np.array([p.betta4_ps_4_per_km * 1e-51]).reshape(-1,1)*np.ones((N_s_max, 1))

      # ---- accumulated gain ----
      G_tot_lin = 10**(0.1 * p.G_tot_dB)

      G_tot_lin_minus_1 = np.zeros((N_s_max, N_c))
      G_tot_lin_minus_1[0,:] = 1
      G_tot_lin_minus_1[1:] = G_tot_lin[:-1]

      # ---- bandwidth ----
      BWCV = p.ECFV - p.SCFV
      CCFV = (p.ECFV + p.SCFV)/2

      CH_BR_vec = BWCV / (1 + p.roll_off_vec)

      BW_eff = np.ones((N_s_max,1)) * CH_BR_vec
      f_s = np.ones((N_s_max,1)) * (CCFV - 0.5*CH_BR_vec)
      f_e = np.ones((N_s_max,1)) * (CCFV + 0.5*CH_BR_vec)

      # ---- channel powers ----
      P_CH_VEC1 = 10**(-3 + 0.1*p.P_in_dBm[0,:])
      P_in_W_ext = np.ones((N_s_max,1)) * P_CH_VEC1

      G_ch = (P_CH_VEC1 / CH_BR_vec)
      G_ch_ext = np.ones((N_s_max,1)) * G_ch

      # ---- sigma handling ----
      sigma_temp = self.sigma.copy()
      sigma_temp[sigma_temp==0] = np.inf

      M_const = 10
      M_p = 5 + np.floor(np.max(np.abs(M_const*2*self.alfa_1/sigma_temp)))

      # ---- h coefficients ----
      h_cons = np.zeros((N_s_max, N_c, int(M_p+1)))
      alfa_1_temp = self.alfa_1

      for m in range(1, int(M_p+1)):
         for m_p in range(1, int(M_p+1)):
               h_cons[:,:,m-1] += (
                  (1/factorial(m_p-1)) *
                  ((2*self.alfa_1/sigma_temp)**(m_p-1)) *
                  (1/(4*self.alfa_0+(m+m_p-2)*sigma_temp))
               )

      undersigma_extra = np.zeros((N_s_max,1))

      # ======================================
      # Main CUT loop
      # ======================================

      for n_CUT in range(N_c):

         roll_off_vec_CUT_ex = p.roll_off_vec[n_CUT] * np.ones((N_s_max,N_c))
         roll_off_vec_no_CUT_ex = np.ones((N_s_max,1)) * p.roll_off_vec

         PHI_ex = np.ones((N_s_max,1)) * p.PHI
         R_s_ex = 1e-12 * CH_BR_vec[0][n_CUT] * np.ones((N_s_max,N_c))

         B_acc = np.zeros((N_s_max,N_c))
         B_acc_1 = np.zeros((N_s_max,N_c))

         # ---- accumulated dispersion ----
         for n_s in range(N_s_max):

               if n_s == 0:

                  term = (
                     betta2[0]
                     + np.pi*betta3[0]*(CCFV[0][n_CUT] + CCFV)
                     + (2/3)*np.pi**2*betta4[0] *
                     (CCFV[0][n_CUT]**2 + CCFV[0][n_CUT]*CCFV + CCFV**2)
                  )

                  B_acc[0,:] = 1e24 * term * p.z_max_pos[0]
                  B_acc_1[0, :] = np.zeros(N_c)

               else:

                  term_ns = (
                     betta2[n_s]
                     + np.pi*betta3[n_s]*(CCFV[0][n_CUT] + CCFV)
                     + (2/3)*np.pi**2*betta4[n_s] *
                     (CCFV[0][n_CUT]**2 + CCFV[0][n_CUT]*CCFV + CCFV**2)
                  )

                  term_prev = (
                     betta2[n_s-1]
                     + np.pi*betta3[n_s-1]*(CCFV[0][n_CUT] + CCFV)
                     + (2/3)*np.pi**2*betta4[n_s-1] *
                     (CCFV[0][n_CUT]**2 + CCFV[0][n_CUT]*CCFV + CCFV**2)
                  )

                  B_acc[n_s,:] = (
                     1e24*term_ns*p.z_max_pos[n_s]
                     + B_acc_1[n_s-1,:]
                     + 1e24*term_prev*L_s[n_s-1]
                  )

                  B_acc_1[n_s,:] = (
                     B_acc_1[n_s-1,:]
                     + 1e24*term_prev*L_s[n_s-1]
                  )

         B_acc_abs = np.abs(B_acc)

         nn_ch = np.ones((N_s_max,1))*np.arange(1,N_c+1)

         KK_cut = (nn_ch == (n_CUT+1))

         # ---- SCI correction ----
         cor_f_CUT = (
               (1 + a[22]*(roll_off_vec_CUT_ex)**a[23]) *
               (a[8] + a[9]*(PHI_ex)**a[10]
               + a[11]*(PHI_ex)**a[12] *
               (1 + a[13]*R_s_ex**a[14]
               + a[15]*(B_acc_abs + a[16])**a[17]))
         )

         # ---- XCI correction ----
         cor_f_no_CUT = (
               (1 + a[18]*(roll_off_vec_CUT_ex)**a[19]
               + a[20]*(roll_off_vec_no_CUT_ex)**a[21]) *
               (a[0] + a[1]*(PHI_ex)**a[2]
               + a[3]*(PHI_ex)**a[4] *
               (1 + a[5]*(B_acc_abs + a[6])**a[7]))
         )

         # enable/disable corrections
         cor_f_CUT = (ro_SCI == 1) * cor_f_CUT + (ro_SCI == 0) * 1
         cor_f_no_CUT = (ro_XCI == 1) * cor_f_no_CUT + (ro_XCI == 0) * 1

         # merge
         cor_f = KK_cut * cor_f_CUT + (1 - KK_cut) * cor_f_no_CUT

         # ---------- degeneracy factor ----------
         Del_fact = 2 - (nn_ch == (n_CUT + 1))

         # ---------- effective beta2 ----------
         beta2_eff = (
             betta2 
             + np.pi * betta3 * (CCFV + CCFV[0][n_CUT]) 
             + (2/3) * np.pi**2 * betta4 * (CCFV[0][n_CUT]**2 + CCFV[0][n_CUT] * CCFV + CCFV**2)
         )

         # ---------- sai_1 ----------
         denom = 4 * np.pi * np.abs(beta2_eff) * (2 * (self.alfa_0 + self.alfa_1))

         arg1 = (
            np.pi**2 * np.abs(beta2_eff) *
            CH_BR_vec[0][n_CUT] *
            (f_e - CCFV[0][n_CUT])
            / (2*self.alfa_0 + 2*self.alfa_1))

         arg2 = (
            np.pi**2 * np.abs(beta2_eff) *
            CH_BR_vec[0][n_CUT] *
            (f_s - CCFV[0][n_CUT])
            / (2*self.alfa_0 + 2*self.alfa_1))

         sai_1 = (1 / denom) * (np.arcsinh(arg1) - np.arcsinh(arg2))
         sai_2=np.zeros((N_s_max,N_c))
         for m in range(int(M_p) + 1):
            factorial_term = 1 / factorial(m)
            power_term = (2 * alfa_1_temp / sigma_temp) ** m
            h_term = h_cons[:, :, m]
            denom_loss = 2 * self.alfa_0 + m * sigma_temp
            arg_upper = (
               np.pi**2
               * np.abs(beta2_eff)
               * CH_BR_vec[0][n_CUT]
               * (f_e - CCFV[0][n_CUT])
               / denom_loss)
            
            arg_lower = (
               np.pi**2
               * np.abs(beta2_eff)
               * CH_BR_vec[0][n_CUT]
               * (f_s - CCFV[0][n_CUT])
               / denom_loss)

            phase_term = (
               1 / (4 * np.pi * np.abs(beta2_eff))
            ) * (np.arcsinh(arg_upper) - np.arcsinh(arg_lower))

            sai_2 += (
               factorial_term
               * power_term
               * h_term
               * phase_term)
         # Final scaling of sai_2
         sai_2 = 2 * sai_2 * np.exp(-4 * alfa_1_temp / sigma_temp)
         sai = sai_2
         # Compute kk12 (SCI effective beta)
         kk12 = (betta2
            + np.pi * betta3 * (CCFV[0][n_CUT] + CCFV[0][n_CUT])
            + (2/3) * np.pi**2 * betta4 * 3 * (CCFV[0][n_CUT] ** 2))
         # Avoid division by zero
         kk12[kk12 == 0] = 1e-100
         gamma = (
            (2 * np.pi / self.p.c)
            * self.p.nuu_vec[n_CUT]
            * self.p.n2_vec
            / A_eff(self.p.nuu_vec[n_CUT], self.p.c, self.p.NA_vec, self.p.a_vec)).T
         
         gamma_sq = gamma**2
         loss_term = (1 / (2 * self.alfa_0[:, n_CUT]))**2

         sinint_arg = (
             np.pi**2
             * np.abs(kk12)
             * L_s
             * (CH_BR_vec[0][n_CUT]**2)
         )

         sinint_term = np.zeros_like(sinint_arg, dtype=float)

         for idx in range(len(sinint_arg)):
             sinint_term[idx] = sinint_1(sinint_arg[idx])

         denom = np.pi * np.abs(kk12) * L_s

         undersigma_extra_term_coh = (self.p.ro_coh
            * cor_f[:, n_CUT]* (G_tot_lin_minus_1[:, n_CUT]**2)
            * (G_ch[0][n_CUT]**3)* gamma_sq* loss_term
            * (2/np.pi)* sinint_term/ denom)
         
         A_cut = A_eff(self.p.nuu_vec[n_CUT], self.p.c, self.p.NA_vec, self.p.a_vec)
         A_all = A_eff(self.p.nuu_vec, self.p.c, self.p.NA_vec, self.p.a_vec)

         gamma_ext = (
            (2 * np.pi / self.p.c)
            * self.p.nuu_vec[n_CUT]
            * (self.p.n2_vec.reshape(-1,1))
            / (0.5 * A_cut + 0.5 * A_all)
         )

         
         for n_s in range(N_s_max):
            # === Incoherent GN sum ===
            term = (gamma_ext[:n_s+1, :]**2
               * G_ch_ext[:n_s+1, :]**2
               * G_tot_lin_minus_1[:n_s+1, :]**2
               * Del_fact[:n_s+1, :]
               * sai[:n_s+1, :]
               * cor_f[:n_s+1, :])

            gn_sum = np.sum(term)

            G_NLI[n_s, n_CUT] = ((16/27)
               * G_ch[0][n_CUT]
               * G_tot_lin[n_s, n_CUT]
               * gn_sum
            )

            # === Coherent accumulation ===
            if n_s > 0:
               undersigma_extra[n_s, 0] = (
                     undersigma_extra[n_s-1, 0]
                     + undersigma_extra_term_coh[n_s, 0]
               )
            else:
               undersigma_extra[n_s, 0] = undersigma_extra_term_coh[n_s, 0]

            # === Harmonic factor ===
            if n_s > 0:
               harmonic_val = np.sum(1 / np.arange(1, n_s+1))
            else:
               harmonic_val = 0

            coherence_factor = (1/(n_s+1)) - 1 + harmonic_val

            G_NLI[n_s, n_CUT] += ((16/27)
               * coherence_factor
               * G_tot_lin[n_s, n_CUT]
               * undersigma_extra[n_s, 0])

      BW_eff = np.ones((N_s_max,1))*self.p.CH_BR_vec
      P_NLI = G_NLI*BW_eff
      return P_NLI
   
   def solve_online(self):
      p = self.p
      
      N_c = p.N_c
      Beta2 = p.betta2_ps_squared_per_km*1e-27
      Beta3 = p.betta3_ps_cube_per_km*1e-39
      Beta4 = p.betta4_ps_4_per_km
      f_ref = p.bands[0].opt_params.f_ref
      frequencies = p.grid_center
      n2 = p.n2
      c = p.c
      A_eff = np.array([A_eff_2_D(f, 300000000, p.NA_vec[0], p.a_vec[0]) for f in frequencies])
      R = p.bands[0].opt_params.Rs_mat
      P = p.P_in
      roll_off = p.bands[0].opt_params.rof
      BW_ch = R * ( 1 + roll_off)
      PHI = p.PHI[0]
      
      # ---- model coefficients ----
      # 0-indexed in Python: a[0] = a1, ..., a[23] = a24
      a = np.array([
         +9.3143e-1, -7.7122e-1, +9.1090e-1, -1.4555e+1, # a1 to a4
         +8.5816e-1, -9.9415e-1, +1.0812e0,  +5.2247e-3, # a5 to a8
         +9.9313e-1, -1.8838e0,  +6.2974e-1, -1.1421e+1, # a9 to a12
         +6.7368e-1, -1.1759e0,  +6.4482e-3, +1.8738e+5, # a13 to a16
         +1.9527e+3, -2.0016e0,  -6.7997e-1, +2.0215e0,  # a17 to a20
         -2.9781e-1, +5.5130e-1, -3.6718e-1, +1.1486e0   # a21 to a24
      ])

      # order of series expansion
      M = np.zeros(N_c, dtype=int)
      for i in range(N_c):
         M[i] = int(np.fix(10 * np.abs(2 * self.alfa_1[0][i] / self.sigma[0][i]))) + 1

      P_NLI = np.zeros(N_c)

      for n in range(N_c):
         summ = 0
         for m in range(N_c):
               for j in range(2): # j = 0, 1
                  for k in range(M[n] + 1): # k = 0 to M[n]
                     for q in range(M[n] + 1): # q = 0 to M[n]
                           
                           if m == n:
                              beta2_acc = Beta2 * 1e24
                              ro = (1 + a[22] * (roll_off**a[23])) * \
                                    ( a[8] + a[9]*(PHI[m]**a[10]) + a[11]*(PHI[m]**a[12]) * \
                                    (1 + a[13]*((BW_ch/(1+roll_off))**a[14]) + a[15] * \
                                       ((abs(beta2_acc) + a[16])**a[17]) ) )
                           else:
                              beta2_acc = 0
                              ro = (1 + a[18]*(roll_off**a[19]) + a[20]*(roll_off**a[21])) * \
                                    ( a[0] + a[1]*(PHI[m]**a[2]) + a[3]*(PHI[m]**a[4]) * \
                                    (1 + a[5]*((abs(beta2_acc) + a[6])**a[7])) )

                           eff_dispersion = Beta2 + np.pi * Beta3 * (frequencies[m] + frequencies[n] - 2*f_ref) + \
                                          (2 * (np.pi**2) / 3) * Beta4 * \
                                          ( (frequencies[n] - f_ref)**2 + (frequencies[n] - f_ref)*(frequencies[m] - f_ref) + (frequencies[m] - f_ref)**2 )
                              
                           Gamma = (2 * np.pi * frequencies[n] * 2 * n2) / (c * (A_eff[n] + A_eff[m]))
                           
                           Si_arg = ((np.pi**2) * eff_dispersion * R * (frequencies[m] - frequencies[n] + (((-1)**j) * (R/2)))) / \
                                    (2 * self.alfa_0[0][m] + k * self.sigma[0][m])
                           Si = np.arcsinh(Si_arg)
                           
                           if m == n:
                              delta_func = 1
                           else:
                              delta_func = 0
                              
                           term1 = (((Gamma * P[m])**2) * ro * (2 - delta_func) * np.exp(-4 * self.alfa_1[0][m] / self.sigma[0][m]) * ((-1)**j)) / \
                                 (2 * np.pi * (R**2) * math.factorial(k) * math.factorial(q) * eff_dispersion * (4 * self.alfa_0[0][m] + (k + q) * self.sigma[0][m]))
                                 
                           term2 = (2 * self.alfa_1[0][m] / self.sigma[0][m]) ** (k + q)
                           
                           summ += term1 * term2 * Si
                           
         P_NLI[n] = (16.0 / 27.0) * P[n] * summ

      return P_NLI

class ASESolver:

   def __init__(self,params,gain_dB=None,loss_dB=None):

      self.p=params
      if gain_dB is None:
         gain_dB = self.p.Gain_dB_vec
      if loss_dB is None:
         loss_dB = self.p.Loss_dB
      self.gain=gain_dB
      self.loss=loss_dB

   def solve(self):

      N_s=self.p.N_ss
      N_c=self.p.N_c

      G_ASE=np.zeros((N_s,N_c))
      P_ASE=np.zeros((N_s,N_c))

      h=6.62607004e-34
      # --- BOOSTER NOISE CALCULATION ---
      # Ensure booster parameters exist (falling back to defaults if not)
      G_booster_dB = self.p.G_booster_dB
      F_booster_dB = self.p.F_booster_dB
      
      F_booster_lin = 10 ** (F_booster_dB / 10)
      G_booster_lin = 10 ** (G_booster_dB / 10)
      
      G_ASE_booster = h * self.p.nuu_vec * (G_booster_lin - 1) * F_booster_lin
      # ---------------------------------

      for n_s in range(N_s):

         if n_s==0:

               Loss_lin=10**(self.loss[n_s,:]/10)
               F_lin=10**(0.1*self.p.F_dB[n_s,:])
               G_lin=10**(0.1*self.gain[n_s,:])

               G_ASE[n_s,:]= (G_ASE_booster * (1 / Loss_lin) * G_lin) + h*self.p.nuu_vec*(G_lin-1)*F_lin

         else:

               Loss_lin=10**(self.loss[n_s,:]/10)

               F_lin=10**(0.1*self.p.F_dB[n_s,:])
               G_lin=10**(0.1*self.gain[n_s,:])

               G_ASE[n_s,:]=G_ASE[n_s-1,:]*(1/Loss_lin)*G_lin + \
                           h*self.p.nuu_vec*(G_lin-1)*F_lin

         P_ASE[n_s,:]=G_ASE[n_s,:]*self.p.CH_BR_vec

      return P_ASE

class OSNRCalculator:

   def __init__(self,params,P_ASE,P_NLI,G_tot_lin=None,P_in=None):

      self.p=params
      self.P_ASE=P_ASE
      self.P_NLI=P_NLI
      N_s_max = self.p.N_ss
      if G_tot_lin is None:
         G_tot_lin = 10**(0.1 * self.p.G_tot_dB)
      if P_in is None:
         P_CH_VEC1 = 10**(-3+0.1*self.p.P_in_dBm[0,:])
         P_in_W_ext = np.ones((N_s_max,1))*P_CH_VEC1
         P_in = P_in_W_ext

      self.G_tot_lin=G_tot_lin
      self.P_in=P_in

   def compute(self):

      OSNR_NLI_dB=10*np.log10(((self.P_in*self.G_tot_lin-1*self.P_NLI)*((self.P_in*self.G_tot_lin-self.P_NLI)>0)+1e-30*((self.P_in*self.G_tot_lin-self.P_NLI)<=0))/self.P_NLI)
   
      OSNR_ASE_dB=10*np.log10(((self.P_in*self.G_tot_lin-1*self.P_NLI)*((self.P_in*self.G_tot_lin-self.P_NLI)>0)+1e-30*((self.P_in*self.G_tot_lin-self.P_NLI)<=0))/self.P_ASE)

      OSNR_NLI_ASE_dB=10*np.log10(((self.P_in*self.G_tot_lin-1*self.P_NLI)*((self.P_in*self.G_tot_lin-self.P_NLI)>0)+1e-30*((self.P_in*self.G_tot_lin-self.P_NLI)<=0))/(self.P_ASE+self.P_NLI))

      return OSNR_NLI_dB,OSNR_ASE_dB,OSNR_NLI_ASE_dB

class OSNRCalculatorV2:

    def __init__(self, params, P_ASE, P_NLI, multicore_params=None,
                 G_tot_lin=None, P_in=None):

        self.p = params
        self.P_ASE = P_ASE
        self.P_NLI = P_NLI
        self.mc = multicore_params

        N_s_max = self.p.N_ss

        if G_tot_lin is None:
            G_tot_lin = 10**(0.1 * self.p.G_tot_dB)

        if P_in is None:
            P_CH_VEC1 = 10**(-3 + 0.1*self.p.P_in_dBm[0, :])
            P_in_W_ext = np.ones((N_s_max, 1)) * P_CH_VEC1
            P_in = P_in_W_ext

        self.G_tot_lin = G_tot_lin
        self.P_in = P_in


    def _signal_power(self):
        """Signal power after amplification with numerical protection."""
        S = self.P_in * self.G_tot_lin - self.P_NLI
        return (S * (S > 0) + 1e-30 * (S <= 0))


    def compute(self):

        S = self._signal_power()

        OSNR_NLI_dB = 10*np.log10(S / self.P_NLI)

        OSNR_ASE_dB = 10*np.log10(S / self.P_ASE)

        OSNR_NLI_ASE_dB = 10*np.log10(S / (self.P_ASE + self.P_NLI))

        results = {
            "OSNR_NLI_dB": OSNR_NLI_dB,
            "OSNR_ASE_dB": OSNR_ASE_dB,
            "OSNR_NLI_ASE_dB": OSNR_NLI_ASE_dB
        }

        # Multicore Crosstalk (if provided)
        if self.mc is not None:

            ic_xt = self.mc.ic_xt   # linear XT coefficient

            S_xt = self.P_in * self.G_tot_lin

            P_XC_span = ic_xt * (S_xt * (S_xt > 0) + 1e-30*(S_xt <= 0))

            OSNR_ICXT_dB = 10*np.log10(S / P_XC_span)

            OSNR_NLI_ASE_XT_dB = 10*np.log10(
                S / np.abs(self.P_ASE + self.P_NLI + P_XC_span)
            )

            results["P_XC_span"] = P_XC_span
            results["OSNR_ICXT_dB"] = OSNR_ICXT_dB
            results["OSNR_NLI_ASE_XT_dB"] = OSNR_NLI_ASE_XT_dB

        return results
    


class MultiSpanSystem:
    """
    Orchestrates the multi-span Closed-Form Model (CFM) simulation pipeline.
    Replicates the main MATLAB script by sequentially chaining the physical solvers.
    """
    
    def __init__(self, link, bands, P_in, grid_center, 
                 multicore_params=None, isrs_model='FLP', **kwargs):
        """
        Initializes the pipeline with the required configuration arrays.
        Accepts any additional **kwargs that ParameterBuilder requires.
        """
        self.params = ParameterBuilder(link, bands, P_in, grid_center, **kwargs)
        self.multicore_params = multicore_params
        self.isrs_model = isrs_model

    def set_input_power(self, P_in):
        """
        Updates the input power for power-sweep simulations without re-instantiating.
        """
        self.params.P_in = P_in
        self.params.reset()

    def run(self):
        """
        Executes the complete multi-span calculations sequence.
        Returns a dictionary containing the final OSNR arrays and intermediate results.
        """
        # Step 1: Solve ISRS for power evolution and attenuation parameters
        isrs = ISRSSolver(self.params, model=self.isrs_model)
        Pout_dBm_00, alfa_0, alfa_1, sigma = isrs.solve()
        
        if Pout_dBm_00 is None:
            raise ValueError(f"ISRS simulation failed for P_in = {self.params.P_in} dBm.")

        # Step 2: Calculate Nonlinear Interference (NLI) Noise
        nli = NLISolver(self.params, alfa_0, alfa_1, sigma)
        P_NLI = nli.solve()

        # Step 3: Calculate Amplified Spontaneous Emission (ASE) Noise
        ase = ASESolver(self.params)
        P_ASE = ase.solve()

        # Step 4: Calculate OSNR metrics
        osnr = OSNRCalculatorV2(self.params, P_ASE, P_NLI, multicore_params=self.multicore_params)
        results = osnr.compute()
        
        # Attach intermediate physical arrays for plotting or debugging
        results['P_NLI'] = P_NLI
        results['P_ASE'] = P_ASE
        results['alfa_0'] = alfa_0
        results['alfa_1'] = alfa_1
        results['sigma'] = sigma
        results['G_tot_dB'] = self.params.G_tot_dB
        
        return results





