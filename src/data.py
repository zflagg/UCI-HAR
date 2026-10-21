"""Data loading, windowing, subject-wise splitting, and normalization.

Arrays use the shape (N, C, T): N windows, C sensor channels, T time steps.
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

HAR_SIGNALS = [f"{kind}_{axis}" for kind in ("body_acc", "body_gyro", "total_acc") for axis in "xyz"]
HAR_LABELS = ["walking", "walking_upstairs", "walking_downstairs", "sitting", "standing", "laying"]


# ---------------------------------------------------------------- loaders

def load_har(root, split):
    """Load one split ("train" or "test") of the UCI HAR raw inertial signals.

    `root` is the unzipped "UCI HAR Dataset" folder. Windows are already
    2.56 s long (128 samples at 50 Hz) with 50% overlap.
    """
    d = Path(root) / split
    X = np.stack(
        [np.loadtxt(d / "Inertial Signals" / f"{s}_{split}.txt") for s in HAR_SIGNALS],
        axis=1,
    ).astype(np.float32)
    y = np.loadtxt(d / f"y_{split}.txt", dtype=int) - 1  # labels 1..6 -> 0..5
    subjects = np.loadtxt(d / f"subject_{split}.txt", dtype=int)
    return X, y, subjects


def make_synthetic(n_subjects=12, per_subject=80, n_channels=9, T=128, n_classes=6, seed=0):
    """Fake sensor data for smoke-testing the pipeline before real data is downloaded.

    Each class is a noisy sine at a different frequency, with a per-subject
    gain and offset so that subject-wise splitting actually matters.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2.56, T, dtype=np.float32)
    X, y, subj = [], [], []
    for s in range(n_subjects):
        gain, offset = rng.uniform(0.7, 1.3), rng.normal(0, 0.3)
        for _ in range(per_subject):
            k = rng.integers(n_classes)
            phase = rng.uniform(0, 2 * np.pi, size=(n_channels, 1))
            sig = gain * np.sin(2 * np.pi * (k + 1) * t + phase) + offset
            X.append(sig + rng.normal(0, 0.5, size=sig.shape))
            y.append(k)
            subj.append(s)
    return np.asarray(X, np.float32), np.asarray(y), np.asarray(subj)


def load_dataset(name, root=None):
    """Return (train, test, label_names) where train/test are (X, y, subjects)."""
    if name == "har":
        if root is None:
            raise ValueError("--data-root is required for the HAR dataset")
        return load_har(root, "train"), load_har(root, "test"), HAR_LABELS
    if name == "synthetic":
        X, y, s = make_synthetic()
        test_mask = s >= 9  # hold out the last 3 subjects entirely
        return (X[~test_mask], y[~test_mask], s[~test_mask]), \
               (X[test_mask], y[test_mask], s[test_mask]), \
               [f"class_{i}" for i in range(6)]
    raise ValueError(f"unknown dataset: {name}")


# ---------------------------------------------------------------- windowing

def make_windows(signal, window, stride):
    """Cut a continuous recording of shape (T_total, C) into windows (N, C, window).

    Use this for datasets that are not pre-windowed (PAMAP2, your own CSVs).
    """
    signal = np.asarray(signal, dtype=np.float32)
    starts = range(0, len(signal) - window + 1, stride)
    return np.stack([signal[i:i + window].T for i in starts]) if len(starts) else \
        np.empty((0, signal.shape[1], window), np.float32)


# ---------------------------------------------------------------- splitting

def subject_split(subjects, val_frac=0.2, seed=0):
    """Boolean masks (train, val) that keep every subject on one side only.

    Random window-level splits leak: overlapping windows from the same person
    land in both sets and inflate accuracy.
    """
    rng = np.random.default_rng(seed)
    unique = rng.permutation(np.unique(subjects))
    n_val = max(1, int(round(len(unique) * val_frac)))
    val_mask = np.isin(subjects, unique[:n_val])
    return ~val_mask, val_mask


# ---------------------------------------------------------------- normalization

def fit_norm(X):
    """Per-channel mean/std computed on TRAINING data only."""
    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


def apply_norm(X, mean, std):
    return (X - mean) / std


class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
