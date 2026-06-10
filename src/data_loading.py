"""Load and harmonize raw Phyphox CSV exports into clean DataFrames.

Phyphox exports accelerometer and gyroscope data as separate CSV files.
This module reads both, drops the pre-computed magnitude column, and
resamples both sensors onto a shared uniform time grid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# --- Exact column names from Phyphox export ---
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
    """Read accelerometer CSV and return DataFrame with columns
    ['time', 'acc_x', 'acc_y', 'acc_z'].

    The 'Absolute acceleration' column exported by Phyphox is dropped —
    we recompute magnitude ourselves later for consistency.

    Args:
        path: Path to the accelerometer CSV file.

    Returns:
        DataFrame with renamed columns, sorted by time.
    """
    df = pd.read_csv(path)
    df = df.rename(columns=_ACC_COLS)
    return df[["time", "acc_x", "acc_y", "acc_z"]].sort_values("time")


def _read_gyr(path: Path) -> pd.DataFrame:
    """Read gyroscope CSV and return DataFrame with columns
    ['time', 'gyr_x', 'gyr_y', 'gyr_z'].

    The 'Absolute' column exported by Phyphox is dropped.

    Args:
        path: Path to the gyroscope CSV file.

    Returns:
        DataFrame with renamed columns, sorted by time.
    """
    df = pd.read_csv(path)
    df = df.rename(columns=_GYR_COLS)
    return df[["time", "gyr_x", "gyr_y", "gyr_z"]].sort_values("time")


def load_session(
    acc_path: Path,
    gyr_path: Path,
    target_rate_hz: int = config.SAMPLING_RATE_HZ,
) -> pd.DataFrame:
    """Load one session: merge accelerometer + gyroscope on a uniform grid.

    Phyphox timestamps are slightly irregular and the two sensors may run
    at different rates. We interpolate both onto a shared uniform time grid
    at ``target_rate_hz`` Hz so every downstream step sees evenly-spaced
    samples.

    Args:
        acc_path: Path to the accelerometer CSV.
        gyr_path: Path to the gyroscope CSV.
        target_rate_hz: Target sampling rate in Hz after resampling.

    Returns:
        DataFrame with columns:
        ['time', 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
    """
    acc = _read_acc(acc_path)
    gyr = _read_gyr(gyr_path)

    # Overlapping time window only (both sensors must be running).
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
