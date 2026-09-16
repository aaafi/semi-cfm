from importlib.resources import files, as_file
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from scipy.interpolate import interp1d
from scipy.io import loadmat
from CFM.core.band import Band


def get_data_file_path(filename: str) -> Path:
    """
    Locate a data file within the package directory or upward search path.

    Searches within the installed package resources (CFM.data) and falls back
    to upward filesystem traversal to find the file in root or local data folders.

    Args:
    ---------
        filename (str):
            Name of the data file to locate (e.g., 'alpha_reference.npz').

    Returns:
    ---------
        Path:
            Resolved Path object pointing to the existing file.

    Raises:
    -------
        FileNotFoundError:
            If the file cannot be located in package resources or parent directories.

    Example:
    ---------
    >>> from CFM.utils.build_alpha_db import get_data_file_path
    >>> file_path = get_data_file_path("alpha_reference.npz")
    """
    try:
        resource = files("CFM.data").joinpath(filename)
        with as_file(resource) as resolved_path:
            if resolved_path.is_file():
                return Path(resolved_path)
    except (ModuleNotFoundError, TypeError):
        pass

    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        candidate_root = parent / "data" / filename
        candidate_pkg = parent / "CFM" / "data" / filename
        if candidate_root.is_file():
            return candidate_root
        if candidate_pkg.is_file():
            return candidate_pkg

    raise FileNotFoundError(
        f"Could not locate '{filename}'. Ensure it exists in 'data/' or 'CFM/data/'."
    )


def load_alpha_reference() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load reference fiber attenuation and frequency arrays from storage.

    Returns:
    ---------
        Tuple[np.ndarray, np.ndarray]:
            - alpha_ref (np.ndarray): Fiber attenuation array in dB/km.
            - freq_ref (np.ndarray): Reference optical frequency array in Hz.

    Example:
    ---------
    >>> from CFM.utils.build_alpha_db import load_alpha_reference
    >>> alpha_ref, freq_ref = load_alpha_reference()
    """
    npz_path = get_data_file_path("alpha_reference.npz")
    data = np.load(str(npz_path))
    return data["alpha_ref"], data["freq_ref"]


def load_alpha_db_lcs(filename: str = "alpha_dB_LCS_268channels.mat") -> np.ndarray:
    """
    Load precomputed multi-channel fiber attenuation matrix from a MATLAB .mat file.

    Args:
    ---------
        filename (str, optional):
            Name of the MAT file containing the attenuation matrix.
            Default is 'alpha_dB_LCS_268channels.mat'.

    Returns:
    ---------
        np.ndarray:
            Attenuation coefficients array ('alpha_dB_LCS') in dB/km.

    Example:
    ---------
    >>> from CFM.utils.build_alpha_db import load_alpha_db_lcs
    >>> alpha_lcs = load_alpha_db_lcs()
    """
    mat_path = get_data_file_path(filename)
    mat_data = loadmat(str(mat_path))
    return mat_data["alpha_dB_LCS"]


def build_alpha_for_band(alpha_ref: Optional[np.ndarray] = None,
                         freq_ref: Optional[np.ndarray] = None,
                         band: Optional[Band] = None,
                         *,
                         extrapolate: bool = True) -> np.ndarray:
    """
    Interpolate fiber attenuation across the frequency grid of an optical transmission band.

    Computes channel-specific attenuation values by mapping reference fiber attenuation
    data onto the optical channel grid of the specified transmission band.

    Args:
    ---------
        alpha_ref (np.ndarray, optional):
            Reference fiber attenuation values in dB/km. If None, loaded automatically.
        freq_ref (np.ndarray, optional):
            Reference optical frequencies in Hz. If None, loaded automatically.
        band (Band):
            Optical transmission band instance defining the frequency grid and channel spacing.
        extrapolate (bool, optional):
            If True, extrapolate attenuation values outside the reference frequency range.
            If False, clamp values to the boundary values. Default is True.

    Returns:
    ---------
        np.ndarray:
            Interpolated attenuation array across the band channels in dB/km.

    Raises:
    -------
        ValueError:
            If the band instance is not provided.

    Example:
    ---------
    >>> from CFM.core.band import Band, OpticalParameters
    >>> from CFM.utils.build_alpha_db import build_alpha_for_band
    >>> band = Band(name='C', start_freq=191.3, end_freq=196.1, opt_params=OpticalParameters())
    >>> alpha_band = build_alpha_for_band(band=band)
    """
    if band is None:
        raise ValueError("A valid 'band' instance must be provided.")

    if alpha_ref is None or freq_ref is None:
        alpha_loaded, freq_loaded = load_alpha_reference()
        if alpha_ref is None and freq_ref is None:
            alpha_ref, freq_ref = alpha_loaded, freq_loaded
        elif alpha_ref is None:
            alpha_ref = alpha_loaded
            if freq_ref.shape[0] != freq_loaded.shape[0] or not np.allclose(freq_ref, freq_loaded):
                freq_ref = freq_loaded
        elif freq_ref is None:
            freq_ref = freq_loaded
            if alpha_ref.shape[0] != alpha_loaded.shape[0]:
                alpha_ref = alpha_loaded

    if not np.all(np.diff(freq_ref) > 0):
        idx = np.argsort(freq_ref)
        freq_ref = freq_ref[idx]
        alpha_ref = alpha_ref[idx]

    if extrapolate:
        alpha_interp = interp1d(
            freq_ref,
            alpha_ref,
            kind="linear",
            fill_value="extrapolate",
            assume_sorted=True,
        )
    else:
        alpha_interp = interp1d(
            freq_ref,
            alpha_ref,
            kind="linear",
            bounds_error=False,
            fill_value=(alpha_ref[0], alpha_ref[-1]),
            assume_sorted=True,
        )

    new_grid = band.calc_spectrum() + band.channel_spacing * 0.5
    new_grid = new_grid[::-1] * 1e12

    return alpha_interp(new_grid)