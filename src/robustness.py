"""Evaluate a trained checkpoint on the test set under simulated sensor failures.

Example:
    python -m src.robustness --checkpoint runs/<run>/model.pt --dataset har --data-root "data/UCI HAR Dataset"

Compare a run trained with and without --augment to see whether augmentation buys robustness.
"""
import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader

from .data import WindowDataset, apply_norm, load_dataset
from .engine import get_device, metrics, predict
from .models import build_model
from .perturb import PERTURBATIONS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", choices=["har", "synthetic"], default="synthetic")
    p.add_argument("--data-root", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(ckpt["model"], ckpt["in_channels"], ckpt["n_classes"]).to(device)
    model.load_state_dict(ckpt["state_dict"])

    _, (X_te, y_te, _), _ = load_dataset(args.dataset, args.data_root)
    test_dl = DataLoader(WindowDataset(apply_norm(X_te, ckpt["mean"], ckpt["std"]), y_te), batch_size=256)

    print(f"{'perturbation':<15}{'level':>7}{'macro-F1':>11}")
    for name, (fn, levels) in PERTURBATIONS.items():
        for level in levels:
            torch.manual_seed(args.seed)  # same corruption across models for a fair comparison
            m = metrics(*predict(model, test_dl, device, partial(fn, **_kw(name, level))), ckpt["n_classes"])
            print(f"{name:<15}{level:>7.2f}{m['macro_f1']:>11.4f}")


def _kw(name, level):
    return {"noise": {"sigma": level}, "drop_channels": {"p": level}, "drift": {"max_slope": level}}[name]


if __name__ == "__main__":
    main()
