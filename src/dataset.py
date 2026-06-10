"""Build the (X, y) dataset with a leakage-free train/val/test split.

Split is done at the session level, not the window level.
Adjacent windows share 100 samples (50% overlap), so a random
window split would leak near-identical data across the boundary.
Assigning full sessions to each split avoids this entirely.

For activities with 3 sessions: sessions 1-2 → train, session 3 → val/test.
For falling (single session): contiguous 60/20/20 block split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config, data_loading, features, preprocessing


@dataclass
class Dataset:
    """Train / val / test split container.

    Attributes:
        X_train: Feature matrix, shape (n, n_features).
        y_train: Labels, shape (n,).
        X_val:   Validation features.
        y_val:   Validation labels.
        X_test:  Test features — evaluated once, at the very end.
        y_test:  Test labels.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val:   np.ndarray
    y_val:   np.ndarray
    X_test:  np.ndarray
    y_test:  np.ndarray

    def summary(self) -> None:
        """Print split sizes and per-class window counts."""
        for name, _, y in [
            ("train", self.X_train, self.y_train),
            ("val",   self.X_val,   self.y_val),
            ("test",  self.X_test,  self.y_test),
        ]:
            counts = {
                config.LABELS[i]: int(np.sum(y == i))
                for i in sorted(config.LABELS)
            }
            print(f"{name:5s} → {len(y):4d} windows | {counts}")


def _process_session(acc_path: Path, gyr_path: Path) -> np.ndarray:
    """Load one session and return its feature matrix."""
    df = data_loading.load_session(acc_path, gyr_path)
    df = preprocessing.fill_missing(df)
    df = preprocessing.denoise(df)
    windows = preprocessing.sliding_windows(df)
    return np.asarray([features.extract_features(w) for w in windows])


def _split_single_session(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """Split one session into contiguous train/val/test blocks.

    Used for falling (only one session available). Contiguous blocks
    avoid leakage from overlapping windows.
    """
    n      = len(X)
    i_val  = int(n * train_ratio)
    i_test = int(n * (train_ratio + val_ratio))

    return (
        (X[:i_val],        y[:i_val]),
        (X[i_val:i_test],  y[i_val:i_test]),
        (X[i_test:],       y[i_test:]),
    )


def build_dataset(raw_dir: Path = config.RAW_DATA_DIR) -> Dataset:
    """Build the full dataset from raw Phyphox CSV files.

    Args:
        raw_dir: Root folder with one subfolder per activity label.

    Returns:
        Dataset with train, val, and test splits.
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    parts: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "train": [], "val": [], "test": [],
    }

    for label_id, name in config.LABELS.items():
        activity_dir = raw_dir / name
        sessions     = sorted(activity_dir.glob("*_acc.csv"))

        if not sessions:
            print(f"[WARN] No sessions found for '{name}', skipping.")
            continue

        if len(sessions) == 1:
            acc_path = sessions[0]
            gyr_path = acc_path.with_name(acc_path.name.replace("_acc", "_gyr"))
            X = _process_session(acc_path, gyr_path)
            y = np.full(len(X), label_id)
            (Xtr, ytr), (Xv, yv), (Xte, yte) = _split_single_session(X, y)
            parts["train"].append((Xtr, ytr))
            parts["val"].append((Xv, yv))
            parts["test"].append((Xte, yte))

        else:
            # Last session → test, second-to-last → val, the rest → train
            for i, acc_path in enumerate(sessions):
                gyr_path = acc_path.with_name(acc_path.name.replace("_acc", "_gyr"))
                X = _process_session(acc_path, gyr_path)
                y = np.full(len(X), label_id)

                if i == len(sessions) - 1:
                    split = "test"
                elif i == len(sessions) - 2:
                    split = "val"
                else:
                    split = "train"

                parts[split].append((X, y))

    def stack_and_shuffle(key: str) -> tuple[np.ndarray, np.ndarray]:
        Xs  = np.concatenate([x for x, _ in parts[key]])
        ys  = np.concatenate([y for _, y in parts[key]])
        idx = rng.permutation(len(Xs))
        return Xs[idx], ys[idx]

    X_train, y_train = stack_and_shuffle("train")
    X_val,   y_val   = stack_and_shuffle("val")
    X_test,  y_test  = stack_and_shuffle("test")

    return Dataset(X_train, y_train, X_val, y_val, X_test, y_test)
