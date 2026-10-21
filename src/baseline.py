"""Non-deep baselines the deep model has to beat.

Example:
    python -m src.baseline --dataset har --data-root "data/UCI HAR Dataset"
"""
import argparse

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from .data import load_dataset


def window_features(X):
    """Hand-crafted features per channel: summary stats plus spectral energy in 3 bands."""
    stats = [X.mean(-1), X.std(-1), X.min(-1), X.max(-1),
             np.percentile(X, 25, axis=-1), np.percentile(X, 75, axis=-1)]
    spectrum = np.abs(np.fft.rfft(X, axis=-1)) ** 2
    bands = np.array_split(spectrum[..., 1:], 3, axis=-1)  # skip the DC component
    energy = [b.sum(-1) for b in bands]
    return np.concatenate(stats + energy, axis=1)  # (N, C * 9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["har", "synthetic"], default="synthetic")
    p.add_argument("--data-root", default=None)
    args = p.parse_args()

    (X_tr, y_tr, _), (X_te, y_te, _), _ = load_dataset(args.dataset, args.data_root)
    F_tr, F_te = window_features(X_tr), window_features(X_te)

    for name, clf in [
        ("majority class", DummyClassifier(strategy="most_frequent")),
        ("random forest (features)", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0)),
    ]:
        pred = clf.fit(F_tr, y_tr).predict(F_te)
        print(f"{name:<26} accuracy {accuracy_score(y_te, pred):.4f}  "
              f"macro-F1 {f1_score(y_te, pred, average='macro'):.4f}")


if __name__ == "__main__":
    main()
