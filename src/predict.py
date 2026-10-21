"""Run a trained model on a raw sensor recording.

The CSV must have one column per channel, in the same order used for training,
and one row per time step at the training sampling rate (50 Hz for UCI HAR).

Example:
    python -m src.predict --checkpoint runs/<run>/model.pt --csv recording.csv --out predictions.csv
"""
import argparse

import numpy as np
import torch

from .data import apply_norm, make_windows
from .models import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--stride", type=int, default=None, help="default: half a window")
    p.add_argument("--out", default="predictions.csv")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model"], ckpt["in_channels"], ckpt["n_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    signal = np.loadtxt(args.csv, delimiter=",", skiprows=1, dtype=np.float32)
    if signal.ndim == 1:
        signal = signal[:, None]
    if signal.shape[1] != ckpt["in_channels"]:
        raise ValueError(f"CSV has {signal.shape[1]} channels, model expects {ckpt['in_channels']}")

    window = ckpt["window"]
    stride = args.stride or window // 2
    X = make_windows(signal, window, stride)
    if len(X) == 0:
        raise ValueError(f"recording is shorter than one window ({window} samples)")

    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(apply_norm(X, ckpt["mean"], ckpt["std"]))), dim=1).numpy()

    labels = ckpt["labels"]
    with open(args.out, "w") as f:
        f.write("window,start_sample,label,confidence\n")
        for i, row in enumerate(probs):
            k = int(row.argmax())
            f.write(f"{i},{i * stride},{labels[k]},{row[k]:.4f}\n")
    print(f"wrote {len(probs)} window predictions to {args.out}")


if __name__ == "__main__":
    main()
