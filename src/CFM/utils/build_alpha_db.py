from scipy.interpolate import interp1d
import numpy as np

def load_alpha_reference():
    path = ".././data/alpha_reference.npz"
    data = np.load(path)
    
    freq_ref = data["freq_ref"]
    alpha_ref = data["alpha_ref"]

    # Sort if needed
    if not np.all(np.diff(freq_ref) > 0):
        idx = np.argsort(freq_ref)
        freq_ref = freq_ref[idx]
        alpha_ref = alpha_ref[idx]

    return alpha_ref, freq_ref

def build_alpha_for_band(alpha_ref, freq_ref, band, *, extrapolate=True):
    """
    alpha_ref: 1D array of alpha in dB. If None → load automatically.
    freq_ref: 1D array of frequencies (Hz). If None → load automatically.
    band: object with calc_spectrum() and channel_spacing attributes.
    extrapolate: if False, clamp outside reference range.
    """

    # 1. Load reference if missing
    if alpha_ref is None or freq_ref is None:
        alpha_loaded, freq_loaded = load_alpha_reference()
        
        if alpha_ref is None and freq_ref is None:
            alpha_ref, freq_ref = alpha_loaded, freq_loaded
        elif alpha_ref is None:
            alpha_ref = alpha_loaded
            # Force using loaded freq_ref if user-provided one mismatches
            if freq_ref.shape[0] != freq_loaded.shape[0] or not np.allclose(freq_ref, freq_loaded):
                freq_ref = freq_loaded
        elif freq_ref is None:
            freq_ref = freq_loaded
            if alpha_ref.shape[0] != alpha_loaded.shape[0]:
                alpha_ref = alpha_loaded

    # 2. Guarantee sorted frequencies
    if not np.all(np.diff(freq_ref) > 0):
        idx = np.argsort(freq_ref)
        freq_ref = freq_ref[idx]
        alpha_ref = alpha_ref[idx]

    # 3. Build interpolator
    if extrapolate:
        alpha_interp = interp1d(freq_ref, alpha_ref, kind="linear",
                                fill_value="extrapolate",
                                assume_sorted=True)
    else:
        alpha_interp = interp1d(freq_ref, alpha_ref, kind="linear",
                                bounds_error=False,
                                fill_value=(alpha_ref[0], alpha_ref[-1]),
                                assume_sorted=True)

    # 4. Compute band grid
    new_grid = band.calc_spectrum() + band.channel_spacing * 0.5
    new_grid = new_grid[::-1] * 1e12  # preserve your pipeline convention

    # 5. Evaluate interpolation
    return alpha_interp(new_grid)
