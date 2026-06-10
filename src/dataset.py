"""Assemble the full (X, y) dataset with a leakage-free train/val/test split.

WHY SESSION-LEVEL SPLIT:
    Windows overlap by 50%, so two adjacent windows share 100 samples.
    A random window-level split would place near-identical windows on
    both sides of the train/test boundary → data leakage → artificially
    inflated accuracy. Splitting at the session level guarantees that no
    window in the test set shares samples with any training window.

SPLIT STRATEGY:
    For activities with 3 sessions:
        session 1, 2  → train
        session 3     → val   (first half) / test (second half)
    For falling (1 session only):
        first 60%  → train
        next 20%   → val
        last 20%   → test
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config, data_loading, features, preprocessing


@dataclass
class Dataset:
    """Container for the train / val / test split.

    Attributes:
        X_train: Feature matrix for training, shape (n, n_features).
        y_train: Label vector for training, shape (n,).
        X_val:   Feature matrix for validation.
        y_val:   Label vector for validation.
        X_test:  Feature matrix for test (used ONCE at the very end).
        y_test:  Label vector for test.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    def summary(self) -> None:
        """Print a quick overview of split sizes and class distribution."""
        for name, X, y in [
            ("train", self.X_train, self.y_train),
            ("val", self.X_val, self.y_val),
            ("test", self.X_test, self.y_test),
        ]:
            counts = {
                config.LABELS[i]: int(np.sum(y == i)) for i in sorted(config.LABELS)
            }
            print(f"{name:5s} → {len(y):4d} windows | {counts}")


def _process_session(acc_path: Path, gyr_path: Path) -> np.ndarray:
    """Load, clean, and extract features from one session.

    Args:
        acc_path: Path to the accelerometer CSV.
        gyr_path: Path to the gyroscope CSV.

    Returns:
        Feature matrix of shape (n_windows, n_features).
    """
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
    """Split a single session into train/val/test by contiguous blocks.

    Used for the 'falling' class which has only one session.
    Contiguous (non-random) split avoids leakage from overlapping windows.

    Args:
        X: Feature matrix for the session.
        y: Label vector for the session.
        train_ratio: Fraction of windows assigned to train.
        val_ratio: Fraction of windows assigned to val.

    Returns:
        Three (X, y) tuples for train, val, and test.
    """
    n = len(X)
    i_val = int(n * train_ratio)
    i_test = int(n * (train_ratio + val_ratio))

    return (
        (X[:i_val], y[:i_val]),
        (X[i_val:i_test], y[i_val:i_test]),
        (X[i_test:], y[i_test:]),
    )


def build_dataset(raw_dir: Path = config.RAW_DATA_DIR) -> Dataset:
    """Build the full dataset from raw CSV files.

    Iterates over every activity folder, loads all sessions, and
    assigns them to train / val / test using a leakage-free strategy.

    Args:
        raw_dir: Root directory containing one subfolder per activity.

    Returns:
        Dataset with populated train, val, and test splits.
    """
    rng = np.random.default_rng(config.RANDOM_SEED)

    parts: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for label_id, name in config.LABELS.items():
        activity_dir = raw_dir / name
        sessions = sorted(activity_dir.glob("*_acc.csv"))

        if len(sessions) == 0:
            print(f"[WARN] No sessions found for '{name}', skipping.")
            continue

        if len(sessions) == 1:
            # Only one session (e.g. falling) → contiguous block split.
            acc_path = sessions[0]
            gyr_path = acc_path.with_name(acc_path.name.replace("_acc", "_gyr"))
            X = _process_session(acc_path, gyr_path)
            y = np.full(len(X), label_id)
            (Xtr, ytr), (Xv, yv), (Xte, yte) = _split_single_session(X, y)
            parts["train"].append((Xtr, ytr))
            parts["val"].append((Xv, yv))
            parts["test"].append((Xte, yte))

        else:
            # Multiple sessions → last = test, second-to-last = val,
            # the rest = train.
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

    def stack_and_shuffle(
        key: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Concatenate all (X, y) pairs for a split and shuffle."""
        Xs = np.concatenate([x for x, _ in parts[key]])
        ys = np.concatenate([y for _, y in parts[key]])
        idx = rng.permutation(len(Xs))
        return Xs[idx], ys[idx]

    X_train, y_train = stack_and_shuffle("train")
    X_val, y_val = stack_and_shuffle("val")
    X_test, y_test = stack_and_shuffle("test")

    return Dataset(X_train, y_train, X_val, y_val, X_test, y_test)
