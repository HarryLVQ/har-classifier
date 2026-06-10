"""Signal cleaning, denoising, and sliding-window segmentation.

Raw IMU signals contain high-frequency sensor noise that is irrelevant
to human motion recognition. We apply a low-pass filter to remove it,
then segment the cleaned signal into fixed-size overlapping windows
ready for feature extraction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from . import config


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Linearly interpolate missing values, then fill remaining edges.

    IMU streams occasionally drop samples. Linear interpolation keeps
    the signal continuous without introducing artificial jumps. Edge
    gaps (start/end) are filled with the nearest valid value.

    Args:
        df: Raw session DataFrame.

    Returns:
        DataFrame with no NaN values.
    """
    return df.interpolate(method="linear").bfill().ffill()


def lowpass_filter(
    signal: np.ndarray,
    cutoff_hz: float = 20.0,
    fs: int = config.SAMPLING_RATE_HZ,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth low-pass filter.

    Human motion energy lies below ~15-20 Hz. A 20 Hz cutoff removes
    high-frequency sensor noise while preserving all motion-relevant
    content. ``filtfilt`` applies the filter twice (forward + backward)
    for zero phase shift — no time distortion.

    Args:
        signal: 1-D array representing one sensor channel.
        cutoff_hz: Cutoff frequency in Hz.
        fs: Sampling rate in Hz.
        order: Filter order (higher = steeper roll-off).

    Returns:
        Filtered signal, same shape as input.
    """
    nyquist = 0.5 * fs
    b, a = butter(order, cutoff_hz / nyquist, btype="low")
    return filtfilt(b, a, signal)


def denoise(df: pd.DataFrame) -> pd.DataFrame:
    """Apply low-pass filter to every sensor channel.

    The 'time' column is left untouched.

    Args:
        df: Session DataFrame with SENSOR_CHANNELS columns.

    Returns:
        Denoised DataFrame, same shape as input.
    """
    out = df.copy()
    for ch in config.SENSOR_CHANNELS:
        out[ch] = lowpass_filter(df[ch].to_numpy())
    return out


def sliding_windows(
    df: pd.DataFrame,
    window_size: int = config.WINDOW_SIZE,
    step: int = config.WINDOW_STEP,
) -> np.ndarray:
    """Cut a continuous recording into overlapping fixed-size windows.

    Each window is a snapshot of all sensor channels over ``window_size``
    samples. Consecutive windows overlap by (window_size - step) samples.

    Example with window_size=200, step=100 (50% overlap):
        window 1 → samples [0:200]
        window 2 → samples [100:300]
        window 3 → samples [200:400]
        ...

    Args:
        df: Cleaned session DataFrame containing SENSOR_CHANNELS.
        window_size: Number of samples per window (default: 200 = 2s).
        step: Hop size between windows (default: 100 = 1s).

    Returns:
        Array of shape (n_windows, window_size, n_channels).
    """
    data = df[list(config.SENSOR_CHANNELS)].to_numpy()
    n = len(data)
    windows = [
        data[start : start + window_size]
        for start in range(0, n - window_size + 1, step)
    ]
    return np.asarray(windows)
