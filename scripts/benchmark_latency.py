"""Measure single-window inference latency of the ONNX model on CPU.

Real-time constraint:
    With a 2s window and 50% overlap, a new window arrives every 1s
    (the hop = 100 samples at 100 Hz = 1000 ms).
    The model must classify one window in well under 1000 ms to keep up.

This script measures the average latency over 1000 runs and reports
the real-time headroom (budget / latency).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time

import numpy as np
import onnxruntime as ort

from src import config


def main() -> None:
    """Benchmark ONNX inference latency on CPU."""
    sess       = ort.InferenceSession(str(config.MODELS_DIR / "model.onnx"))
    input_name = sess.get_inputs()[0].name
    n_features = sess.get_inputs()[0].shape[1]

    # Single window = one feature vector.
    dummy = np.random.rand(1, n_features).astype(np.float32)

    # Warm-up : first calls are slower due to JIT / cache effects.
    print("Warming up...")
    for _ in range(20):
        sess.run(None, {input_name: dummy})

    # Benchmark.
    n_runs = 1000
    print(f"Benchmarking {n_runs} inference calls...\n")
    start = time.perf_counter()
    for _ in range(n_runs):
        sess.run(None, {input_name: dummy})
    elapsed = time.perf_counter() - start

    mean_ms  = elapsed / n_runs * 1000
    hop_ms   = config.WINDOW_STEP / config.SAMPLING_RATE_HZ * 1000
    headroom = hop_ms / mean_ms

    print(f"Mean latency       : {mean_ms:.3f} ms / window")
    print(f"Real-time budget   : {hop_ms:.0f} ms (= window hop at {config.SAMPLING_RATE_HZ} Hz)")
    print(f"Real-time headroom : {headroom:.0f}x")
    print()
    if headroom >= 10:
        print("✓ Well within real-time constraints.")
    else:
        print("⚠ Latency too high for real-time use.")


if __name__ == "__main__":
    main()