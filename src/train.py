"""Train and save the HAR Random Forest classifier.

Random Forest was chosen over deep learning for a few practical reasons:
the dataset is small (~700 windows), no feature scaling is needed,
inference is fast, and the serialized model stays lightweight.
class_weight='balanced' handles the under-represented falling class
without manual oversampling.
"""

from __future__ import annotations

import joblib
from sklearn.ensemble import RandomForestClassifier

from . import config, dataset


def build_model() -> RandomForestClassifier:
    """Return a configured (untrained) Random Forest.

    max_depth=12 and min_samples_leaf=3 prevent overfitting on
    the small dataset while keeping inference fast.
    """
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=3,
        n_jobs=-1,
        class_weight="balanced",
        random_state=config.RANDOM_SEED,
    )


def main() -> None:
    """Build dataset, train, and persist model + dataset to disk."""
    print("Building dataset...")
    data = dataset.build_dataset()
    data.summary()

    print("\nTraining Random Forest...")
    model = build_model()
    model.fit(data.X_train, data.y_train)

    val_acc = model.score(data.X_val, data.y_val)
    print(f"\nValidation accuracy : {val_acc:.3f}")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODELS_DIR / "model.joblib")
    joblib.dump(data, config.MODELS_DIR / "dataset.joblib")
    print(f"Model saved → {config.MODELS_DIR / 'model.joblib'}")


if __name__ == "__main__":
    main()
