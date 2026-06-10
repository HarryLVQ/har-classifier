"""Pipeline configuration — all parameters in one place.

Centralizing constants here avoids scattered magic numbers and makes
it easy to re-run experiments with different settings.
"""

from __future__ import annotations

from pathlib import Path

# Paths
ROOT_DIR: Path = Path(__file__).resolve().parents[1]
RAW_DATA_DIR: Path = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR: Path = ROOT_DIR / "data" / "processed"
MODELS_DIR: Path = ROOT_DIR / "models"
REPORTS_DIR: Path = ROOT_DIR / "reports"

# Signal — target rate after resampling (Phyphox is slightly irregular)
SAMPLING_RATE_HZ: int = 100
WINDOW_SECONDS: float = 2.0
WINDOW_OVERLAP: float = 0.5
WINDOW_SIZE: int = int(WINDOW_SECONDS * SAMPLING_RATE_HZ)  # 200 samples
WINDOW_STEP: int = int(WINDOW_SIZE * (1 - WINDOW_OVERLAP))  # 100 samples

# Channel order must stay consistent across the whole pipeline
SENSOR_CHANNELS: tuple[str, ...] = (
    "acc_x", "acc_y", "acc_z",
    "gyr_x", "gyr_y", "gyr_z",
)

# Labels
LABELS: dict[int, str] = {0: "walking", 1: "sitting", 2: "running", 3: "falling"}
LABEL_TO_ID: dict[str, int] = {name: idx for idx, name in LABELS.items()}

RANDOM_SEED: int = 42
