# Human Activity Recognition on Smartphone
**Technical Assessment — Reev** | *[Van Quang LUU]* | *[10/06/2026]*

---

## 1. Dataset Description

IMU data was collected using the Phyphox app (RWTH Aachen) on a
personal smartphone (Samsung A54) placed in the front pants pocket.
Both accelerometer and gyroscope were recorded as separate CSV files at
approximately 100 Hz, then merged in post-processing onto a shared
uniform time grid. 

Recording conditions were kept consistent across
sessions: same pocket, same walking/running route, same seated position.
Falls were simulated as controlled forward and lateral drops onto a mattress.

| Activity | Sessions | Total duration                      |
|----------|----------|-------------------------------------|
| Walking  | 3        | ~4 min                              |
| Sitting  | 3        | ~4 min                              |
| Running  | 3        | ~4 min                              |
| Falling  | 1        | ~1 min (repeated falls on mattress) |

Accelerometer and gyroscope were recorded as separate CSV files and 
merged in post-processing onto a shared uniform time grid.

---

## 2. Preprocessing & Feature Engineering

**Resampling** : Phyphox timestamps are slightly irregular and the two
sensors don't share the same time grid. Both signals are linearly
interpolated onto a uniform 100 Hz grid before any further processing.

**Noise filtering** : A zero-phase 4th-order Butterworth low-pass filter
(cutoff: 20 Hz) is applied to each channel via filtfilt (forward +
backward pass, no phase shift). Human motion energy lies below 15–20 Hz,
so this removes sensor noise without affecting the motion signal.

**Windowing** : The signal is cut into 2-second windows (200 samples)
with 50% overlap (hop = 1 s). Two seconds captures 2–3 full gait cycles
for walking/running, and comfortably encompasses a fall (~1 s). The 50%
overlap doubles the number of available windows and avoids cutting an
event across two boundaries.

**Features** :  105 features are extracted per window, 12 time-domain
statistics (mean, std, RMS, energy, kurtosis, zero-crossing rate, etc.)
and 3 frequency-domain statistics (dominant frequency, spectral energy,
spectral entropy) for each of the 6 sensor channels, plus the same 15
features computed on the acceleration magnitude (orientation-invariant,
robust to phone placement variation in the pocket).

**Split strategy** :  The split is done at the session level, not the
window level. With 50% overlap, adjacent windows share 100 samples,
a random window split would leak near-identical data into both train and
test, artificially inflating accuracy. For the falling class (single
session only), a contiguous 60/20/20 block split is used instead.

| Split | Windows |
|-------|---------|
| Train | 328     |
| Val   | 198     |
| Test  | 198     |

---

## 3. Model Architecture & Training

A **Random Forest** was chosen over deep learning approaches (1D-CNN,
LSTM) for three practical reasons: the dataset is small (~700 windows
total), the model must be lightweight for embedded deployment, and
hand-crafted features on 2-second windows already carry enough
discriminative information. No feature scaling is needed, and feature
importances come for free.

| Hyperparameter     | Value    | Justification                                   |
|--------------------|----------|-------------------------------------------------|
| `n_estimators`     | 100      | Stable predictions, lightweight                 |
| `max_depth`        | 12       | Limits overfitting on small dataset             |
| `min_samples_leaf` | 3        | Avoids memorising noisy windows                 |
| `class_weight`     | balanced | Compensates for under-represented falling class |
| `random_state`     | 42       | Reproducibility                                 |


The model takes 105 features as input (6 channels × 15 features + 15 on
acceleration magnitude). The validation set was used to confirm
hyperparameter choices. The test set was used once, only for final
reporting.

---

## 4. Results

| Class         | Precision | Recall   | F1       | Support |
|---------------|-----------|----------|----------|---------|
| Walking       | 0.98      | 0.98     | 0.98     | 64      |
| Sitting       | 1.00      | 1.00     | 1.00     | 60      |
| Running       | 1.00      | 0.98     | 0.99     | 65      |
| Falling       | 0.90      | 1.00     | 0.95     | 9       |
| **Macro avg** | **0.97**  | **0.99** | **0.98** | 198     |

**Test accuracy : 99% — Macro F1 : 0.981**

![Confusion matrix](reports/figures/confusion_matrix.png)

**Failure mode analysis** : Sitting is perfectly classified (F1 = 1.00)
as expected — the signal is nearly flat. The main confusion risk is
between walking and running at intermediate pace, though this only
affected 1–2 windows in practice (F1 = 0.98–0.99).

The falling class has recall = 1.00, no fall was missed, which is the
critical metric in a medical context. Precision = 0.90 means one window
from another activity was incorrectly flagged as a fall. With only 9 test
windows, a single error shifts precision by ~10 points; results for this
class should be confirmed on a larger fall dataset.

---

## 5. On-Device Performance

The model is exported to **ONNX** (40.4 KB) and benchmarked using
`onnxruntime` on a standard laptop CPU (1000 inference calls, after
20 warm-up runs).

| Metric                        | Value             |
|-------------------------------|-------------------|
| ONNX model size               | 40.4 KB           |
| Mean inference latency        | 0.032 ms / window |
| Real-time budget (window hop) | 1000 ms           |
| Real-time headroom            | **31 354×**       |

With a 2s window and 50% overlap, a new prediction is needed every 1
second. At 0.032 ms per window, the model runs ~31 000× faster than
required, leaving very comfortable margin even for embedded targets
running at a fraction of a laptop's clock speed. ONNX and sklearn
predictions were verified to match exactly (agreement = 1.0000).

---

## 6. Limitations & Next Steps

The main limitation of this work is the single-subject dataset, all
recordings come from one person, which likely explains the high accuracy
but limits generalization. A real deployment would need data from multiple
subjects with varying morphologies, walking speeds, and phone positions.

The falling class in particular would benefit from more diverse data:
only controlled mattress falls were recorded, which may not represent
real-world fall dynamics. For a production system, I would augment this
with a public fall dataset (e.g. SisFall) and apply data augmentation
(signal jittering, time-warping) to improve robustness.