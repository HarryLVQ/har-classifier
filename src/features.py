"""Hand-crafted time- and frequency-domain features for HAR windows.

Each window of raw IMU data is summarised into a flat vector of
numerical features. Classical ML models (Random Forest, SVM, etc.)
expect this fixed-length vector as input rather than raw time series.

Feature families:
- Time domain : statistics that describe the signal's amplitude and
  shape (mean, std, RMS, skewness, etc.).
- Frequency domain : statistics derived from the FFT that capture
  periodicity and dominant motion frequency.
- Acceleration magnitude : orientation-invariant norm of the 3-axis
  accelerometer, robust to phone placement variation.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.fft import rfft, rfftfreq

from . import config


def _time_features(x: np.ndarray) -> dict[str, float]:
    """Compute time-domain statistics for a single channel.

    Args:
        x: 1-D array of signal values for one window.

    Returns:
        Dictionary mapping feature name to scalar value.
    """
    return {
        "mean":     float(np.mean(x)),
        "std":      float(np.std(x)),
        "min":      float(np.min(x)),
        "max":      float(np.max(x)),
        "median":   float(np.median(x)),
        "rms":      float(np.sqrt(np.mean(x ** 2))),
        "mad":      float(np.median(np.abs(x - np.median(x)))),
        "iqr":      float(np.percentile(x, 75) - np.percentile(x, 25)),
        "skew":     float(stats.skew(x)),
        "kurtosis": float(stats.kurtosis(x)),
        "energy":   float(np.sum(x ** 2) / len(x)),
        "zcr":      float(np.mean(np.abs(np.diff(np.sign(x))) > 0)),
    }


def _freq_features(x: np.ndarray, fs: int) -> dict[str, float]:
    """Compute frequency-domain features via FFT for a single channel.

    Args:
        x: 1-D array of signal values for one window.
        fs: Sampling rate in Hz.

    Returns:
        Dictionary mapping feature name to scalar value.
    """
    spectrum = np.abs(rfft(x))
    freqs    = rfftfreq(len(x), d=1.0 / fs)
    power    = spectrum ** 2
    total    = float(np.sum(power)) + 1e-12
    norm     = power / total

    return {
        "dom_freq":    float(freqs[int(np.argmax(spectrum))]),
        "spec_energy": total,
        "spec_entropy": float(-np.sum(norm * np.log2(norm + 1e-12))),
    }


def extract_features(
    window: np.ndarray,
    fs: int = config.SAMPLING_RATE_HZ,
) -> np.ndarray:
    """Turn one window into a flat feature vector.

    Computes time- and frequency-domain features for every sensor
    channel, plus the same features on the acceleration magnitude
    (orientation-invariant summary of the 3-axis accelerometer).

    Args:
        window: Array of shape (window_size, n_channels).
                Channel order must match config.SENSOR_CHANNELS.
        fs: Sampling rate in Hz.

    Returns:
        1-D float32 feature vector of fixed length.
    """
    feats: list[float] = []

    # Per-channel features (all 6 channels).
    for c in range(window.shape[1]):
        x = window[:, c]
        feats.extend(_time_features(x).values())
        feats.extend(_freq_features(x, fs).values())

    # Acceleration magnitude — robust to phone orientation in pocket.
    acc_mag = np.linalg.norm(window[:, 0:3], axis=1)
    feats.extend(_time_features(acc_mag).values())
    feats.extend(_freq_features(acc_mag, fs).values())

    return np.asarray(feats, dtype=np.float32)


def feature_names(fs: int = config.SAMPLING_RATE_HZ) -> list[str]:
    """Return the ordered list of feature names.

    Useful for feature importance plots and debugging.

    Args:
        fs: Sampling rate in Hz.

    Returns:
        List of feature name strings, same order as extract_features().
    """
    dummy = np.zeros((config.WINDOW_SIZE, len(config.SENSOR_CHANNELS)))
    names: list[str] = []

    t_keys = list(_time_features(dummy[:, 0]).keys())
    f_keys = list(_freq_features(dummy[:, 0], fs).keys())

    for ch in config.SENSOR_CHANNELS:
        names += [f"{ch}_{k}" for k in t_keys]
        names += [f"{ch}_{k}" for k in f_keys]

    names += [f"acc_mag_{k}" for k in t_keys]
    names += [f"acc_mag_{k}" for k in f_keys]

    return names