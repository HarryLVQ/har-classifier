"""Export the trained sklearn model to ONNX for portable inference.

ONNX removes the scikit-learn dependency at runtime, which matters
for embedded targets where only onnxruntime is available.
"""

from __future__ import annotations

import joblib
import numpy as np
import onnxruntime as ort
from skl2onnx import to_onnx

from . import config


def main() -> None:
    """Convert to ONNX, save, and verify predictions match sklearn."""
    model = joblib.load(config.MODELS_DIR / "model.joblib")
    data  = joblib.load(config.MODELS_DIR / "dataset.joblib")

    sample = data.X_train[:1].astype(np.float32)

    print("Converting to ONNX...")
    onnx_model = to_onnx(model, sample, target_opset=12)

    onnx_path = config.MODELS_DIR / "model.onnx"
    onnx_path.write_bytes(onnx_model.SerializeToString())
    print(f"ONNX model saved → {onnx_path}")
    print(f"File size        : {onnx_path.stat().st_size / 1024:.1f} KB")

    # Make sure ONNX and sklearn agree on every test prediction
    print("\nVerifying ONNX predictions match sklearn...")
    sess       = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name
    onnx_pred  = sess.run(None, {input_name: data.X_test.astype(np.float32)})[0]
    skl_pred   = model.predict(data.X_test)

    agreement = float(np.mean(onnx_pred.ravel() == skl_pred))
    print(f"Agreement sklearn vs ONNX : {agreement:.4f}")

    if agreement == 1.0:
        print("✓ Perfect match — export successful.")
    else:
        n_diff = int(np.sum(onnx_pred.ravel() != skl_pred))
        print(f"⚠ {n_diff} predictions differ — check skl2onnx version.")


if __name__ == "__main__":
    main()
