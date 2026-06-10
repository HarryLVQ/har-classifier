"""Train the HAR classifier and persist it to disk.

Model choice — Random Forest:
    - No feature scaling required (tree-based model).
    - Fast training and inference, small serialized size.
    - Naturally handles mixed-scale features.
    - ``class_weight='balanced'`` compensates for the under-represented
      'falling' class without manual oversampling.
    - Feature importances available for free (useful for analysis).
"""

from __future__ import annotations

import joblib
from sklearn.ensemble import RandomForestClassifier

from . import config, dataset


def build_model() -> RandomForestClassifier:
    """Instantiate the Random Forest classifier.

    Hyperparameter choices:
        n_estimators=100  : enough trees for stable predictions while
                            keeping the model lightweight.
        max_depth=12      : limits tree size → faster inference, less
                            overfitting on the small dataset.
        min_samples_leaf=3: avoids memorising individual noisy windows.
        class_weight='balanced': up-weights rare classes (falling)
                            so the model doesn't ignore them.

    Returns:
        Untrained RandomForestClassifier instance.
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
    """Build dataset, train model, and save to disk."""
    print("Building dataset...")
    data = dataset.build_dataset()
    data.summary()

    print("\nTraining Random Forest...")
    model = build_model()
    model.fit(data.X_train, data.y_train)

    # Quick validation check (never used for final reporting).
    val_acc = model.score(data.X_val, data.y_val)
    print(f"\nValidation accuracy : {val_acc:.3f}")

    # Persist model and dataset for downstream scripts.
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODELS_DIR / "model.joblib")
    joblib.dump(data, config.MODELS_DIR / "dataset.joblib")
    print(f"Model saved → {config.MODELS_DIR / 'model.joblib'}")


if __name__ == "__main__":
    main()
