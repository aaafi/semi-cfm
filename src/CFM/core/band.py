import numpy as np
import os
from dataclasses import dataclass, field
from scipy.special import erfcinv

@dataclass
class OpticalParameters:
    """
        A data class to store and compute fiber and system parameters for optical network modeling.
    
        Example:
        --------
        >>> from sixgman.core.band import OpticalParameters
        
        >>> # Define C-band parameters
        >>> c_band_params = OpticalParameters()
    """
    
    # Fundamental constants
    h_plank: float = 6.626e-34
    """Planck's constant (J·s)"""
    
    target_ber: float = 1e-2 
    """Target bit error rate"""

    # Modulation format penalty factors
    phi_MFL: np.ndarray = field(default_factory=lambda: -1 * np.array([1, 1, 2 / 3, 17 / 25, 69 / 100, 13 / 21]))
    """Modulation format penalty factors"""

    # System Parameters
    epsilon: int = 0
    """Auxiliary variable for modeling purposes"""
    
    beta_3: float = 0.14e-39
    """Third-order dispersion coefficient (s^3/m)""" 
    
    Cr: float = 0.028 / 1e3 / 1e12
    """Chromatic dispersion coefficient (1/(m·Hz²))"""

    # Fiber Parameters
    alpha_db: float = 0.2 
    """Fiber attenuation (dB/km)"""
    
    beta_2: float = -21.7e-27 
    """Second-order dispersion coefficient (s²/m)"""
    
    gama: float = 1.21e-3 
    """Nonlinear coefficient (1/(W·m))"""

    f_ref: float = 3*1e8/(1550*1e-9)

    # Noise figures (converted from dB to linear scale)
    F_C: float = field(default_factory=lambda: 4.5)  # Noise figure for C-band (6 dB)
    """Noise figure in linear scale for C-band."""
    
    F_L: float = field(default_factory=lambda: 5)   # Noise figure for L-band (6 dB)
    """Noise figure in linear scale for L-band"""

    F_S: float = field(default_factory=lambda: 6)   # Noise figure for S-band (6 dB)
    """Noise figure in linear scale for S-band"""

    F_E: float = field(default_factory=lambda: 5)   # Noise figure for S-band (6 dB)
    """Noise figure in linear scale for E-band"""

    F_G: float = field(default_factory=lambda: 1)   # Noise figure for S-band (6 dB)
    """Noise figure in linear scale for E-band"""

    # Symbol transmission
    Rs_mat: float = 40e9 
    """Symbol rate (Baud)"""
    
    MFL: np.ndarray = field(default_factory=lambda: np.arange(1, 7)) 
    """Available modulation format levels (1 to 6)"""
    
    rof: float = 0.05 
    """Roll-off factor"""

    sysymbol_rate: float = 120*1e9
    """"symbol rate"""

    # Computed attributes (set in __post_init__)
    alpha_norm: float = field(init=False)  # 1/m
    """Normalized attenuation (1/m)"""
    
    L_eff_a: float = field(init=False) # m
    """Effective fiber length (m)"""
    
    B_ch_mat: float = field(init=False) # Hz
    """Channel bandwidth (Hz)"""
    
    B_ch: float = field(init=False) # Hz (alias for B_ch_mat)
    """Channel bandwidth (Hz) [alias]"""
    
    target_SNR_dB: np.ndarray = field(init=False)
    """Target SNR of modulation formats to reach the target_ber (dB)"""

    def __post_init__(self):
        """
        Perform post-initialization computations for dependent parameters.
        """
        # Convert attenuation from dB/km to 1/m (natural units)
        self.alpha_norm = self.alpha_db / (10 * np.log10(np.exp(1)) * 1e3)

        # Compute effective fiber length
        self.L_eff_a = 1 / self.alpha_norm

        # Compute channel bandwidth
        self.B_ch_mat = self.Rs_mat * (1 + self.rof)
        self.B_ch = self.B_ch_mat  # alias
        
        # Compute fixed target SNR values based on target BER
        # Modulation Formats: 64-QAM, 32-QAM, 16-QAM, 8-QAM, QPSK, BPSK
        self.target_SNR_dB = np.array([
            10 * np.log10(2 * (erfcinv(np.log2(64) * self.target_ber / 2 / (1 - 1 / np.sqrt(64)))) ** 2 * (64 - 1) / 3), 
            10 * np.log10(2 * (erfcinv(np.log2(32) * self.target_ber / 2 / (1 - 1 / np.sqrt(32)))) ** 2 * (32 - 1) / 3),
            10 * np.log10(10 * (erfcinv((8 / 3) * self.target_ber)) ** 2),
            10 * np.log10((14 / 3) * (erfcinv(1.5 * self.target_ber)) ** 2),
            10 * np.log10(2 * (erfcinv(2 * self.target_ber)) ** 2),
            10 * np.log10(1 * (erfcinv(2 * self.target_ber)) ** 2), 
        ])

class Band:
    """
    Class representing an optical transmission band with its spectral characteristics.

    Provides frequency slot calculations and properties for multi-band optical 
    network planning (e.g., C, L, S, E, or G bands).

    Attributes:
    -----------
        name (str): 
            Band identifier (e.g., 'C', 'L', 'S', 'E', 'G').
        start_freq (float): 
            Start frequency of the transmission band in THz.
        end_freq (float): 
            End frequency of the transmission band in THz.
        channel_spacing (float): 
            Frequency spacing between adjacent channels in THz.
        opt_params (OpticalParameters): 
            Optical transmission and fiber parameters associated with the band.
        noise_figure (float): 
            Band-specific linear noise figure assigned from opt_params.
        spectrum (np.ndarray): 
            Array of optical carrier center frequencies in THz, ordered high to low.
        num_channels (int): 
            Total number of optical channels available in the band.
    """

    def __init__(self, 
                 name: str,
                 start_freq: float,
                 end_freq: float,
                 opt_params: OpticalParameters,
                 channel_spacing: float = 0.05):
        """
        Initialize an optical transmission Band instance.

        Args:
        ---------
            name (str): 
                Band name (e.g., 'C', 'L', 'S', 'E', 'G').
            start_freq (float): 
                Starting frequency edge of the band in THz.
            end_freq (float): 
                Ending frequency edge of the band in THz.
            opt_params (OpticalParameters): 
                Physical transmission and transceiver parameters.
            channel_spacing (float, optional): 
                Frequency separation between channels in THz. Default is 0.05 THz (50 GHz).

        Raises:
        -------
            ValueError: 
                If start_freq is greater than or equal to end_freq, if channel_spacing is 
                less than or equal to 0, or if an unrecognized band name is provided.

        Example:
        ---------
        >>> from CFM.core.band import Band, OpticalParameters
        >>> c_params = OpticalParameters()
        >>> c_band = Band(
        ...     name='C',
        ...     start_freq=191.3,
        ...     end_freq=196.1,
        ...     opt_params=c_params,
        ...     channel_spacing=0.05
        ... )
        """
        if start_freq >= end_freq:
            raise ValueError("start_freq must be less than end_freq")
        if channel_spacing <= 0:
            raise ValueError("channel_spacing must be positive")

        self.name = name
        self.start_freq = start_freq
        self.end_freq = end_freq
        self.channel_spacing = channel_spacing
        self.opt_params = opt_params

        n = self.name.lower()

        if n == 'c':
            self.noise_figure = self.opt_params.F_C
        elif n == 'l':
            self.noise_figure = self.opt_params.F_L
        elif n == 's':
            self.noise_figure = self.opt_params.F_S
        elif n == 'e':
            self.noise_figure = self.opt_params.F_E
        elif n == 'g':
            self.noise_figure = self.opt_params.F_G
        else:
            raise ValueError(f'Unknown band name: {self.name}')

        self.spectrum: np.ndarray = self.calc_spectrum()
        self.num_channels: int = len(self.spectrum)

    def calc_spectrum(self) -> np.ndarray:
        """
        Calculate the discrete channel center frequencies across the band.

        Returns:
        ---------
            np.ndarray: 
                Array of carrier center frequencies in THz, arranged descendingly.

        Example:
        ---------
        >>> spectrum = c_band.calc_spectrum()
        >>> num_channels = len(spectrum)
        """
        return np.flip(np.arange(self.start_freq, self.end_freq, step=self.channel_spacing))