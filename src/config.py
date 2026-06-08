"""Centralized configuration for the HAR pipeline.

Keeping every magic number in one place makes experiments reproducible
and the rest of the codebase free of hardcoded values.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths ---
ROOT_DIR: Path = Path(__file__).resolve().parents[1]
RAW_DATA_DIR: Path = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR: Path = ROOT_DIR / "data" / "processed"
MODELS_DIR: Path = ROOT_DIR / "models"
REPORTS_DIR: Path = ROOT_DIR / "reports"

# --- Signal parameters ---
SAMPLING_RATE_HZ: int = 100          # target rate after resampling
WINDOW_SECONDS: float = 2.0          # window length
WINDOW_OVERLAP: float = 0.5          # 50% overlap
WINDOW_SIZE: int = int(WINDOW_SECONDS * SAMPLING_RATE_HZ)        # 200 samples
WINDOW_STEP: int = int(WINDOW_SIZE * (1 - WINDOW_OVERLAP))        # 100 samples (hop)

# --- Sensor channels (order matters and must stay consistent everywhere) ---
SENSOR_CHANNELS: tuple[str, ...] = ("acc_x", "acc_y", "acc_z",
                                    "gyr_x", "gyr_y", "gyr_z")

# --- Labels ---
LABELS: dict[int, str] = {0: "walking", 1: "sitting", 2: "running", 3: "falling"}
LABEL_TO_ID: dict[str, int] = {name: idx for idx, name in LABELS.items()}

# --- Reproducibility ---
RANDOM_SEED: int = 42