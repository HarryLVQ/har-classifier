# HAR Classifier — Human Activity Recognition

Lightweight classifier detecting **walking / sitting / running / falling**
from smartphone IMU data (accelerometer + gyroscope), exported to ONNX
for real-time on-device inference.

Built as a technical assessment for Reev — powered knee orthosis company.

---

## Results

| Metric             | Value             |
|--------------------|-------------------|
| Test accuracy      | 99%               |
| Macro F1           | 0.981             |
| ONNX model size    | 40.4 KB           |
| Inference latency  | 0.032 ms / window |
| Real-time budget   | 1000 ms / window  |
| Real-time headroom | 31 354x           |

---

## Dataset

- **Sensors** : accelerometer + gyroscope (3-axis each), sampled at 100 Hz
- **Phone placement** : front pants pocket
- **Recording app** : [Phyphox](https://phyphox.org/) (RWTH Aachen)
- **Activities** : walking, sitting, running, falling
- **Sessions** : 3 sessions per activity (walking, sitting, running), 1 session for falling
- **Split strategy** : session-level split to avoid data leakage from overlapping windows

| Activity | Sessions | Total windows (train/val/test) |
|----------|----------|--------------------------------|
| Walking  |    3     |         125 / 64 / 64          |
| Sitting  |    3     |         59  / 64 / 60          |
| Running  |    3     |         120 / 62 / 65          |
| Falling  |    1     |         24  / 8  / 9           |

---

## Project structure

har-classifier/
├── data/raw/              # Raw Phyphox CSV exports (one folder per activity)
├── models/                # Trained model (.joblib) and ONNX export (.onnx)
├── reports/figures/       # Confusion matrix and other plots
├── scripts/               # Standalone scripts (latency benchmark)
├── src/                   # Main Python package
│   ├── config.py          # Centralized parameters
│   ├── data_loading.py    # Phyphox CSV loading + sensor fusion
│   ├── preprocessing.py   # Noise filtering + sliding windows
│   ├── features.py        # Time & frequency domain feature extraction
│   ├── dataset.py         # Dataset assembly + leakage-free split
│   ├── train.py           # Model training
│   ├── evaluate.py        # Metrics + confusion matrix
│   └── export_onnx.py     # ONNX conversion + verification
└── scripts/
└── benchmark_latency.py   # Inference latency benchmark

---

## Setup

**Requirements** : Python 3.11+, [uv](https://github.com/astral-sh/uv)

```bash
# Clone the repository
git clone https://github.com/HarryLVQ/har-classifier.git
cd har-classifier

# Install dependencies (reproducible via uv.lock)
uv sync
```

---

## Run the full pipeline

```bash
# 1. Train the model
uv run python -m src.train

# 2. Evaluate on test set (metrics + confusion matrix)
uv run python -m src.evaluate

# 3. Export to ONNX + verify predictions match sklearn
uv run python -m src.export_onnx

# 4. Benchmark inference latency
uv run python scripts/benchmark_latency.py
```

---

## Pipeline overview

data/raw/
└── [activity]/sessionX_acc.csv + sessionX_gyr.csv
│
▼
data_loading.py       → merge acc + gyr on uniform 100 Hz grid
│
▼
preprocessing.py      → fill missing + low-pass filter (20 Hz) + sliding windows (2s, 50% overlap)
│
▼
features.py           → 105 features per window (time + frequency domain)
│
▼
dataset.py            → session-level train / val / test split
│
▼
train.py              → Random Forest (100 trees, max_depth=12, balanced weights)
│
▼
evaluate.py           → accuracy, per-class F1, confusion matrix
│
▼
export_onnx.py        → model.onnx (40.4 KB) — real-time inference on CPU

---

## Key design decisions

**Session-level split** : overlapping windows (50%) mean adjacent windows
share 100 samples. A random window-level split would leak near-identical
windows into both train and test, inflating accuracy artificially. Splitting
by session guarantees a clean boundary.

**Random Forest over deep learning** : small dataset (~700 windows total),
real-time embedded target, and interpretable feature importances make a
classical approach the right trade-off here.

**Low-pass filter at 20 Hz** : human motion energy lies below 15-20 Hz.
Filtering removes high-frequency sensor noise while preserving all
motion-relevant signal content.

**class_weight='balanced'** : the falling class is heavily under-represented
(24 training windows vs 120+ for other classes). Balanced weights prevent
the model from ignoring rare but critical events.

---

## Notes

- Falling class : simulated controlled falls onto a mattress (few repetitions).
  Due to the limited number of samples (9 test windows), metrics for this
  class should be interpreted with caution.
- All training and evaluation is performed on personally collected data.


