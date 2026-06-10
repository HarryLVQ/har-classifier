"""Evaluate the trained model on the held-out test set."""

from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from . import config


def main() -> None:
    """Print per-class metrics and save confusion matrix figure."""
    model = joblib.load(config.MODELS_DIR / "model.joblib")
    data = joblib.load(config.MODELS_DIR / "dataset.joblib")

    y_pred = model.predict(data.X_test)
    labels = list(config.LABELS.keys())
    names = [config.LABELS[i] for i in labels]

    print("=" * 50)
    print(classification_report(data.y_test, y_pred, target_names=names))
    print(f"Macro F1 : {f1_score(data.y_test, y_pred, average='macro'):.3f}")
    print("=" * 50)

    cm = confusion_matrix(data.y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=names,
        yticklabels=names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix — test set")
    fig.tight_layout()

    out = config.REPORTS_DIR / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    print(f"\nConfusion matrix saved → {out / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
