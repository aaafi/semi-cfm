from dataclasses import dataclass

@dataclass
class EdfaConfig:
    """Configuration for a single amplifier in a specific span."""
    gain_target: float  # Gain target in dB (e.g., 20.0 to compensate span loss)
    tilt_target: float = 0.0
    out_voa: float = 0.0
    is_booster: bool = False # Flag to identify if this acts as the initial booster