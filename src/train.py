"""Train a deep model on windowed sensor data.

Example:
    python -m src.train --dataset synthetic --epochs 5
    python -m src.train --dataset har --data-root "data/UCI HAR Dataset" --model cnn --augment
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import WindowDataset, apply_norm, fit_norm, load_dataset, subject_split
from .engine import format_confusion, get_device, metrics, predict, train_one_epoch
from .models import build_model
from .perturb import augment


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["har", "synthetic"], default="synthetic")
    p.add_argument("--data-root", default=None)
    p.add_argument("--model", choices=["cnn", "gru"], default="cnn")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=7, help="early-stop after this many epochs without val improvement")
    p.add_argument("--augment", action="store_true", help="train with noise / channel dropout / drift")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()

    (X_tr, y_tr, s_tr), (X_te, y_te, _), labels = load_dataset(args.dataset, args.data_root)
    n_classes, in_channels, T = len(labels), X_tr.shape[1], X_tr.shape[2]

    # Validation subjects are carved out of the training subjects; test subjects are never touched.
    tr_mask, val_mask = subject_split(s_tr, val_frac=0.2, seed=args.seed)
    mean, std = fit_norm(X_tr[tr_mask])  # fit on train split only
    norm = lambda X: apply_norm(X, mean, std)

    train_dl = DataLoader(WindowDataset(norm(X_tr[tr_mask]), y_tr[tr_mask]), batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(WindowDataset(norm(X_tr[val_mask]), y_tr[val_mask]), batch_size=256)
    test_dl = DataLoader(WindowDataset(norm(X_te), y_te), batch_size=256)
    print(f"train {tr_mask.sum()} | val {val_mask.sum()} | test {len(y_te)} windows; "
          f"{in_channels} channels x {T} steps; device {device}")

    model = build_model(args.model, in_channels, n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    run_dir = Path(args.out) / f"{args.dataset}_{args.model}{'_aug' if args.augment else ''}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "model.pt"

    best_f1, bad_epochs, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_dl, optimizer, loss_fn, device, augment if args.augment else None)
        val = metrics(*predict(model, val_dl, device), n_classes)
        history.append({"epoch": epoch, "train_loss": loss, "val_macro_f1": val["macro_f1"]})
        print(f"epoch {epoch:3d}  loss {loss:.4f}  val macro-F1 {val['macro_f1']:.4f}")

        if val["macro_f1"] > best_f1:
            best_f1, bad_epochs = val["macro_f1"], 0
            torch.save({
                "state_dict": model.state_dict(), "model": args.model,
                "in_channels": in_channels, "n_classes": n_classes, "window": T,
                "mean": mean, "std": std, "labels": labels,
            }, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(ckpt_path, weights_only=False)["state_dict"])
    test = metrics(*predict(model, test_dl, device), n_classes)
    print(f"\nTEST  accuracy {test['accuracy']:.4f}  macro-F1 {test['macro_f1']:.4f}")
    print(format_confusion(test["confusion_matrix"], labels))

    with open(run_dir / "results.json", "w") as f:
        json.dump({"args": vars(args), "best_val_macro_f1": best_f1, "test": test, "history": history}, f, indent=2)
    print(f"\nsaved checkpoint and results to {run_dir}")


if __name__ == "__main__":
    main()
