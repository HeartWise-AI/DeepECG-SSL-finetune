"""
Dataset-level ECG amplitude calibration.

Estimates a fixed scale factor to convert ECG signals from arbitrary ADC units
to millivolts, by comparing lead-0 mean power to a reference computed from
datasets with known ADC gains.

Unlike per-sample spectral normalization (which destroys inter-patient amplitude
differences), this computes ONE scale factor for an entire dataset or source,
preserving the amplitude variation that is diagnostically important for
conditions like LVH, chamber enlargement, etc.

Usage:
    from fairseq_signals.data.ecg.calibration import estimate_dataset_scale

    # From a consolidated NPY array
    scale = estimate_dataset_scale("/path/to/X_data.npy", n_samples=2000)

    # From a directory of individual NPY files
    scale = estimate_dataset_scale("/path/to/npy/dir/", n_samples=2000)

    # From an in-memory array of shape (N, samples, leads)
    scale = estimate_dataset_scale(data_array, n_samples=2000)
"""

import glob
import os

import numpy as np


# Median lead-0 (lead I) time-domain mean power of ECG signals in millivolts.
#
# Computed as the geometric mean of two independently verified references:
#   - MHI dataset (scale 0.00488 mV/ADC, MUSE GE): median power = 0.01779 mV^2
#   - MIMIC dataset (scale 0.001 mV/ADC):           median power = 0.01638 mV^2
# Each computed over 5000 random samples with seed=42.
#
# Geometric mean: sqrt(0.01779 * 0.01638) = 0.01707
#
# Validation:
#   MHI:   estimated_scale = 0.00478, known = 0.00488  (2.0% error)
#   MIMIC: estimated_scale = 0.00102, known = 0.001    (2.1% error)
REFERENCE_LEAD0_POWER_MV = 0.01707


def compute_lead_power(signal, lead=0):
    """
    Compute the mean time-domain power of a single lead: mean(x^2).

    Parameters
    ----------
    signal : np.ndarray
        ECG signal of shape (samples, leads), (leads, samples), or
        (samples, leads, 1).
    lead : int
        Lead index to use (default: 0 = lead I).

    Returns
    -------
    float
        Mean power: mean(x^2).
    """
    if signal.ndim == 3 and signal.shape[-1] == 1:
        signal = signal.squeeze(-1)

    if signal.ndim != 2:
        raise ValueError(f"Expected 2D signal, got shape {signal.shape}")

    # Determine orientation: (samples, leads) vs (leads, samples)
    # Heuristic: ECG typically has 12 leads, so the axis with 12 is leads
    if signal.shape[0] == 12 and signal.shape[1] != 12:
        x = signal[lead].astype(np.float64)
    elif signal.shape[1] == 12 and signal.shape[0] != 12:
        x = signal[:, lead].astype(np.float64)
    elif signal.shape[0] == 12:
        # Ambiguous (12, 12) - assume (leads, samples)
        x = signal[lead].astype(np.float64)
    else:
        # Neither axis is 12, use the longer axis as samples
        if signal.shape[0] >= signal.shape[1]:
            x = signal[:, lead].astype(np.float64)
        else:
            x = signal[lead].astype(np.float64)

    n = len(x)
    if n < 2:
        return 0.0

    return float(np.mean(x ** 2))


def _load_samples_from_npy_dir(directory, n_samples, seed):
    """Load a random subset of individual NPY files from a directory."""
    patterns = [
        os.path.join(directory, "**", "*.npy"),
        os.path.join(directory, "*.npy"),
    ]
    all_files = []
    for pat in patterns:
        all_files.extend(glob.glob(pat, recursive=True))
    all_files = sorted(set(all_files))

    if not all_files:
        raise FileNotFoundError(f"No .npy files found in {directory}")

    rng = np.random.RandomState(seed)
    n = min(n_samples, len(all_files))
    chosen = rng.choice(len(all_files), size=n, replace=False)

    signals = []
    for idx in chosen:
        try:
            sig = np.load(all_files[idx]).squeeze()
            if sig.ndim == 2:
                signals.append(sig)
        except Exception:
            continue

    if not signals:
        raise ValueError(f"Could not load any valid signals from {directory}")

    return signals


def _load_samples_from_consolidated(npy_path, n_samples, seed):
    """Load a random subset from a consolidated NPY file (N, samples, leads)."""
    data = np.lib.format.open_memmap(npy_path, mode="r")
    total = data.shape[0]

    rng = np.random.RandomState(seed)
    n = min(n_samples, total)
    indices = rng.choice(total, size=n, replace=False)
    indices.sort()  # sequential access for memmap efficiency

    return [data[i] for i in indices]


def estimate_dataset_scale(
    source,
    n_samples=2000,
    lead=0,
    seed=42,
    reference_power=REFERENCE_LEAD0_POWER_MV,
):
    """
    Estimate a fixed scale factor to convert a dataset's signals to millivolts.

    Computes the median lead-0 mean power across a random sample of the dataset,
    then returns the scale factor that would match this power to a millivolt
    reference derived from datasets with known ADC gains (MHI, MIMIC).

    The scale factor satisfies: raw_signal * scale ≈ signal_in_mV.

    Parameters
    ----------
    source : str or np.ndarray
        One of:
        - Path to a consolidated .npy file of shape (N, samples, leads)
        - Path to a directory containing individual .npy files
        - A numpy array of shape (N, samples, leads)
    n_samples : int
        Number of random samples to use for estimation (default: 2000).
    lead : int
        Lead index to use for power computation (default: 0 = lead I).
    seed : int
        Random seed for reproducible sampling.
    reference_power : float
        Target mean power in mV^2 (default: empirical reference from MHI+MIMIC).

    Returns
    -------
    float
        Fixed scale factor. Multiply raw signals by this to get millivolts.
    """
    # Load signals
    if isinstance(source, np.ndarray):
        rng = np.random.RandomState(seed)
        n = min(n_samples, len(source))
        indices = rng.choice(len(source), size=n, replace=False)
        signals = [source[i] for i in indices]
    elif isinstance(source, str):
        source = os.path.expanduser(source)
        if os.path.isfile(source) and source.endswith(".npy"):
            signals = _load_samples_from_consolidated(source, n_samples, seed)
        elif os.path.isdir(source):
            signals = _load_samples_from_npy_dir(source, n_samples, seed)
        else:
            raise ValueError(
                f"Source must be a .npy file or directory, got: {source}"
            )
    else:
        raise TypeError(f"source must be str or np.ndarray, got {type(source)}")

    # Compute per-sample mean power for the chosen lead
    powers = []
    for sig in signals:
        try:
            p = compute_lead_power(sig, lead=lead)
            if p > 0:
                powers.append(p)
        except (ValueError, IndexError):
            continue

    if not powers:
        raise ValueError("Could not compute power for any signals")

    median_power = np.median(powers)
    scale = np.sqrt(reference_power / median_power)

    return float(scale)
