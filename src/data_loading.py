"""Load raw Phyphox CSV exports and merge accelerometer + gyroscope signals.

Phyphox records each sensor separately, with slightly irregular timestamps.
This module reads both files, drops Phyphox's pre-computed magnitude column,
and resamples everything onto a shared uniform time grid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# Column names as exported by Phyphox (French/English versions may differ)
_ACC_COLS = {
    "Time (s)": "time",
    "Linear Acceleration x (m/s^2)": "acc_x",
    "Linear Acceleration y (m/s^2)": "acc_y",
    "Linear Acceleration z (m/s^2)": "acc_z",
}

_GYR_COLS = {
    "Time (s)": "time",
    "Gyroscope x (rad/s)": "gyr_x",
    "Gyroscope y (rad/s)": "gyr_y",
    "Gyroscope z (rad/s)": "gyr_z",
}


def _read_acc(path: Path) -> pd.DataFrame:
    """Read accelerometer CSV, rename columns, drop the magnitude column."""
    df = pd.read_csv(path)
    df = df.rename(columns=_ACC_COLS)
    return df[["time", "acc_x", "acc_y", "acc_z"]].sort_values("time")


def _read_gyr(path: Path) -> pd.DataFrame:
    """Read gyroscope CSV, rename columns, drop the magnitude column."""
    df = pd.read_csv(path)
    df = df.rename(columns=_GYR_COLS)
    return df[["time", "gyr_x", "gyr_y", "gyr_z"]].sort_values("time")


def load_session(
    acc_path: Path,
    gyr_path: Path,
    target_rate_hz: int = config.SAMPLING_RATE_HZ,
) -> pd.DataFrame:
    """Merge one accelerometer + gyroscope recording pair onto a uniform grid.

    The two sensors don't share the same timestamps, and Phyphox timing
    is slightly irregular. We interpolate both onto a common uniform grid
    so downstream steps always see evenly-spaced samples.

    Args:
        acc_path: Accelerometer CSV exported from Phyphox.
        gyr_path: Gyroscope CSV exported from Phyphox.
        target_rate_hz: Target sampling rate in Hz.

    Returns:
        DataFrame with columns ['time', 'acc_x', 'acc_y', 'acc_z',
        'gyr_x', 'gyr_y', 'gyr_z'] at uniform ``target_rate_hz`` Hz.
    """
    acc = _read_acc(acc_path)
    gyr = _read_gyr(gyr_path)

    # Keep only the overlapping time range (both sensors must be active)
    t_start = max(acc["time"].iloc[0], gyr["time"].iloc[0])
    t_end = min(acc["time"].iloc[-1], gyr["time"].iloc[-1])

    n_samples = int((t_end - t_start) * target_rate_hz)
    grid = np.linspace(t_start, t_end, n_samples)

    out = pd.DataFrame({"time": grid})
    for col in ("acc_x", "acc_y", "acc_z"):
        out[col] = np.interp(grid, acc["time"].to_numpy(), acc[col].to_numpy())
    for col in ("gyr_x", "gyr_y", "gyr_z"):
        out[col] = np.interp(grid, gyr["time"].to_numpy(), gyr[col].to_numpy())

    return out
