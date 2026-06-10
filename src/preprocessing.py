"""Signal cleaning and sliding-window segmentation for IMU data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from . import config


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate missing samples, then fill any remaining edge gaps.

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
    """Zero-phase Butterworth low-pass filter for one sensor channel.

    Human motion sits below ~15-20 Hz, so a 20 Hz cutoff removes
    sensor noise without touching the actual movement signal.
    filtfilt runs forward + backward to avoid any phase shift.

    Args:
        signal: 1-D array for one channel.
        cutoff_hz: Cutoff frequency in Hz.
        fs: Sampling rate in Hz.
        order: Filter order.

    Returns:
        Filtered signal, same shape as input.
    """
    nyquist = 0.5 * fs
    b, a = butter(order, cutoff_hz / nyquist, btype="low")
    return filtfilt(b, a, signal)


def denoise(df: pd.DataFrame) -> pd.DataFrame:
    """Apply low-pass filter to every sensor channel (time column untouched).

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
    """Segment a continuous recording into overlapping fixed-size windows.

    With window_size=200 and step=100, consecutive windows overlap by
    100 samples (50%). This doubles the number of training examples and
    avoids cutting events across window boundaries.

    Args:
        df: Cleaned session DataFrame containing SENSOR_CHANNELS.
        window_size: Samples per window (200 = 2s at 100 Hz).
        step: Hop between windows (100 = 1s, i.e. 50% overlap).

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
