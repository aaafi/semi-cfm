import numpy as np
from .qot_estimator import ParameterBuilder, ISRSSolver, NLISolver, ASESolver, OSNRCalculator, OSNRCalculatorV2

class BasePowerModel:
    def compute_start_power(self, link, span_lengths):
        raise NotImplementedError

    def generate_power_grid(self, start_power):
        raise NotImplementedError


class FRPPowerModel(BasePowerModel):

    def __init__(self, p_opt_input= [-1,  -0.5, -0.1, 0.2,  0.5,  0.7]):
        self.name = 'FRP'
        self.p_opt_input = p_opt_input

    def compute_start_power(self, link):
        Ls = link.length_span

        if Ls < 50:
            power_start = self.p_opt_input[0]
            start_power = power_start - 0.2 * Ls

        elif 50 <= Ls < 60:
            power_start = self.p_opt_input[0]
            start_power = power_start - 0.2 * Ls

        elif 60 <= Ls < 70:
            power_start = self.p_opt_input[1]
            start_power = power_start - 0.2 * Ls

        elif 70 <= Ls < 80:
            power_start = self.p_opt_input[2]
            start_power = power_start - 0.2 * Ls

        elif 80 <= Ls < 90:
            power_start = self.p_opt_input[3]
            start_power = power_start - 0.2 * Ls

        elif 90 <= Ls < 100:
            power_start = self.p_opt_input[4]
            start_power = power_start - 0.2 * Ls - 0.2

        else:
            power_start = self.p_opt_input[5]
            start_power = power_start - 0.2 * Ls - 0.6

        return start_power


    def generate_power_grid(self, start_power=-15, stop_power=0, step=0.1):
        return np.arange(start_power, stop_power + step, step)


class FLPPowerModel(BasePowerModel):

    def __init__(self, p_opt_input = [-1,  -0.5, -0.1, 0.2,  0.5,  0.7]):
        self.name = 'FLP'
        self.p_opt_input = p_opt_input

    def compute_start_power(self, link):
        L = link.length_span

        if L < 50:
            power_start = -1.7
        elif (L >= 50  and L < 60): 
            power_start = self.p_opt_input[0]

        elif (L >= 60  and L < 70): 
            power_start = self.p_opt_input[1]

        elif (L >= 70  and L < 80): 
            power_start = self.p_opt_input[2]

        elif (L >= 80  and L < 90): 
            power_start = self.p_opt_input[3]

        elif (L >= 90  and L < 100): 
            power_start = self.p_opt_input[4]

        elif L >= 100:  
            power_start = self.p_opt_input[5]
        
        return power_start

    def generate_power_grid(self, start_power=-2, stop_power=2, step=0.1):
        return np.arange(start_power, stop_power + step, step)

class PowerOptimizer:

    def __init__(self, link, bands, grid_center, model, alpha_dB_LCS=None):

        self.link = link
        self.bands = bands
        self.grid_center = grid_center
        self.model = model
        self.alpha_dB_LCS = alpha_dB_LCS
        if self.alpha_dB_LCS is None:
            self.alpha_dB_LCS = [0.2]

    def evaluate_power(self, P_in, model_name):

        self.model_name = model_name
        # build parameters
        params = ParameterBuilder(
            self.link,
            self.bands,
            P_in,
            self.grid_center,
            alpha_dB_LCS=self.alpha_dB_LCS
        )

        # ISRS
        alfa_0, alfa_1, sigma = ISRSSolver(params, self.model_name).solve()
        if alfa_0 is None:
            return None, None, None, params


        # NLI
        P_NLI = NLISolver(params, alfa_0, alfa_1, sigma).solve()

        # ASE
        P_ASE = ASESolver(params).solve()

        # OSNR
        OSNR_NLI_dB, OSNR_ASE_dB, OSNR_NLI_ASE_dB = OSNRCalculator(
            params, P_ASE, P_NLI
        ).compute()

        return OSNR_NLI_dB, OSNR_ASE_dB, OSNR_NLI_ASE_dB, params


    def compute_oct(self, OSNR):

        return 2 * 75 * 1e-3 * np.sum(np.log2(1 + 10 ** (OSNR / 10)))


    def optimize(self, span_lengths=None):

        start_power = self.model.compute_start_power(
            self.link
        )

        power_candidates = self.model.generate_power_grid(start_power)

        max_oct = -np.inf
        results = {}

        for p in power_candidates:

            OSNR_NLI_dB, OSNR_ASE_dB, OSNR_NLI_ASE_dB, params = self.evaluate_power(p, self.model.name)

            if OSNR_NLI_ASE_dB is None:
                OCT = 0
            else:
                OCT = self.compute_oct(OSNR_NLI_ASE_dB)

            if OCT > max_oct:
                max_oct = OCT

                results = {
                    "P_opt": p,
                    "gsnr": OSNR_NLI_ASE_dB,
                    "osnr": OSNR_ASE_dB,
                    "snr_nli": OSNR_NLI_dB,
                    "max_oct": OCT
                }
            else:
                break

        return results
    
class PowerOptimizerV2:

    def __init__(self, link, bands, grid_center, model, base, MCF_params=None, search_range=None, alpha_dB_LCS=None):

        self.link = link
        self.bands = bands
        self.grid_center = grid_center
        self.model = model
        self.base = base
        self.MCF_params = MCF_params
        self.alpha_dB_LCS = alpha_dB_LCS
        self.search_range = search_range
        if self.alpha_dB_LCS is None:
            self.alpha_dB_LCS = [0.2]

    def evaluate_power(self, P_in, model_name):

        self.model_name = model_name
        # build parameters
        params = ParameterBuilder(
            self.link,
            self.bands,
            P_in,
            self.grid_center,
            alpha_dB_LCS=self.alpha_dB_LCS
        )

        # ISRS
        alfa_0, alfa_1, sigma = ISRSSolver(params, self.model_name).solve()
        if alfa_0 is None:
            return None, params


        # NLI
        P_NLI = NLISolver(params, alfa_0, alfa_1, sigma).solve()

        # ASE
        P_ASE = ASESolver(params).solve()

        # OSNR
        OSNR= OSNRCalculatorV2(
            params, P_ASE, P_NLI, self.MCF_params
        ).compute()

        return OSNR, params


    def compute_oct(self, OSNR, base):

        return 2 * 75 * 1e-3 * np.sum(np.log2(1 + 10 ** (OSNR[base] / 10)))


    def optimize(self, span_lengths=None):

        if self.search_range is None:
            start_power = self.model.compute_start_power(
                self.link
            )

            power_candidates = self.model.generate_power_grid(start_power)
        else:
            power_candidates = self.search_range

        max_oct = -np.inf
        results = {}

        for p in power_candidates:

            OSNR, params = self.evaluate_power(p, self.model.name)

            if OSNR is None:
                OCT = -np.inf
            else:
                OCT = self.compute_oct(OSNR, self.base)

            if OCT > max_oct:
                max_oct = OCT

                results = {
                    "P_opt": p,
                    "gsnr": OSNR[self.base],
                    "osnr": OSNR['OSNR_ASE_dB'],
                    "snr_nli": OSNR['OSNR_NLI_dB'],
                    "max_oct": OCT
                }
            else:
                if self.search_range is not None:
                    continue
                elif max_oct == -np.inf:
                    raise "No valid power in search range"
                else:
                    break

        return results
    

class TopologyPowerOptimizer:

    def __init__(self, topology, bands, grid_center, model, alpha_dB_LCS=None):

        self.topology = topology
        self.bands = bands
        self.grid_center = grid_center
        self.model = model
        self.alpha_dB_LCS = alpha_dB_LCS

    def optimize(self):

        results = {}

        links = self.topology.get_links_list()

        for u, v, link in links:

            optimizer = PowerOptimizer(
                link=link,
                bands=self.bands,
                grid_center=self.grid_center,
                model=self.model,
                alpha_dB_LCS=self.alpha_dB_LCS
            )

            print("Optimizing for link", u, v, link.length)

            opt_result = optimizer.optimize()

            results[(u, v)] = {
                "start_node": u,
                "end_node": v,
                "weight": link.length,
                "num_span": link.num_span,
                "num_amp": link.num_amp,
                "span_length": link.length_span,
                "optimization": opt_result
            }


        return results
