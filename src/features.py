"""Time- and frequency-domain feature extraction for IMU windows.

Each 2-second window is summarised into a flat vector of 105 features:
12 time-domain stats + 3 FFT-based stats per sensor channel (6 channels),
plus the same 15 features on the acceleration magnitude.

The magnitude is orientation-invariant (norm of x/y/z), which makes
the classifier more robust to how the phone sits in the pocket.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.fft import rfft, rfftfreq

from . import config


def _time_features(x: np.ndarray) -> dict[str, float]:
    """Basic time-domain statistics for one channel."""
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "median": float(np.median(x)),
        "rms": float(np.sqrt(np.mean(x**2))),
        "mad": float(np.median(np.abs(x - np.median(x)))),
        "iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
        "skew": float(stats.skew(x)),
        "kurtosis": float(stats.kurtosis(x)),
        "energy": float(np.sum(x**2) / len(x)),
        "zcr": float(np.mean(np.abs(np.diff(np.sign(x))) > 0)),
    }


def _freq_features(x: np.ndarray, fs: int) -> dict[str, float]:
    """FFT-based features for one channel — dominant frequency, energy, entropy."""
    spectrum = np.abs(rfft(x))
    freqs = rfftfreq(len(x), d=1.0 / fs)
    power = spectrum**2
    total = float(np.sum(power)) + 1e-12
    norm = power / total

    return {
        "dom_freq": float(freqs[int(np.argmax(spectrum))]),
        "spec_energy": total,
        "spec_entropy": float(-np.sum(norm * np.log2(norm + 1e-12))),
    }


def extract_features(
    window: np.ndarray,
    fs: int = config.SAMPLING_RATE_HZ,
) -> np.ndarray:
    """Convert one IMU window into a flat feature vector (105 values).

    Args:
        window: Array of shape (window_size, n_channels).
                Channel order must follow config.SENSOR_CHANNELS.
        fs: Sampling rate in Hz.

    Returns:
        1-D float32 vector of length 105.
    """
    feats: list[float] = []

    for c in range(window.shape[1]):
        x = window[:, c]
        feats.extend(_time_features(x).values())
        feats.extend(_freq_features(x, fs).values())

    # Acceleration magnitude — less sensitive to phone orientation
    acc_mag = np.linalg.norm(window[:, 0:3], axis=1)
    feats.extend(_time_features(acc_mag).values())
    feats.extend(_freq_features(acc_mag, fs).values())

    return np.asarray(feats, dtype=np.float32)


def feature_names(fs: int = config.SAMPLING_RATE_HZ) -> list[str]:
    """Return feature names in the same order as extract_features().

    Useful for debugging and feature importance analysis.
    """
    dummy = np.zeros((config.WINDOW_SIZE, len(config.SENSOR_CHANNELS)))
    t_keys = list(_time_features(dummy[:, 0]).keys())
    f_keys = list(_freq_features(dummy[:, 0], fs).keys())

    names: list[str] = []
    for ch in config.SENSOR_CHANNELS:
        names += [f"{ch}_{k}" for k in t_keys]
        names += [f"{ch}_{k}" for k in f_keys]

    names += [f"acc_mag_{k}" for k in t_keys]
    names += [f"acc_mag_{k}" for k in f_keys]
    return names
