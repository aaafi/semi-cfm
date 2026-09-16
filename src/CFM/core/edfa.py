from dataclasses import dataclass


@dataclass
class EdfaConfig:
    """
    Configuration profile for an optical amplifier within a fiber span or terminal.

    Specifies target optical gain, gain-flattening tilt targets, attenuation from 
    variable optical attenuators (VOA), and operational mode (inline or booster)[cite: 3].

    Attributes:
    -----------
        gain_target (float): 
            Nominal target operational gain in dB (e.g., set to compensate span loss)[cite: 3].
        tilt_target (float): 
            Gain tilt slope target in dB across the band to compensate spectral gain ripple. 
            Default is 0.0 dB[cite: 3].
        out_voa (float): 
            Output Variable Optical Attenuator (VOA) loss setting in dB. Default is 0.0 dB[cite: 3].
        is_booster (bool): 
            Flag indicating whether the amplifier operates as a post-transmitter 
            booster amplifier rather than an inline span amplifier. Default is False[cite: 3].

    Example:
    --------
    >>> from CFM.core.edfa import EdfaConfig
    >>> # Define an inline span EDFA
    >>> inline_edfa = EdfaConfig(gain_target=18.0, tilt_target=0.5, out_voa=1.0)
    >>> # Define a post-transmitter booster EDFA
    >>> booster_edfa = EdfaConfig(gain_target=15.0, is_booster=True)
    """

    gain_target: float
    tilt_target: float = 0.0
    out_voa: float = 0.0
    is_booster: bool = False