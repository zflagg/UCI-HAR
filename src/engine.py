"""Training and evaluation loops, shared by train.py and robustness.py."""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def train_one_epoch(model, loader, optimizer, loss_fn, device, augment_fn=None):
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if augment_fn is not None:
            xb = augment_fn(xb)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(yb)
        n += len(yb)
    return total / n


@torch.no_grad()
def predict(model, loader, device, perturb_fn=None):
    model.eval()
    preds, targets = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        if perturb_fn is not None:
            xb = perturb_fn(xb)
        preds.append(model(xb).argmax(dim=1).cpu().numpy())
        targets.append(yb.numpy())
    return np.concatenate(targets), np.concatenate(preds)


def metrics(y_true, y_pred, n_classes):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).tolist(),
    }


def format_confusion(cm, labels):
    width = max(len(l) for l in labels)
    header = " " * (width + 2) + " ".join(f"{i:>5}" for i in range(len(labels)))
    rows = [f"{l:>{width}}  " + " ".join(f"{v:>5}" for v in row) + f"   ({i})"
            for i, (l, row) in enumerate(zip(labels, cm))]
    return "rows = true, columns = predicted\n" + header + "\n" + "\n".join(rows)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple silicon
        return torch.device("mps")
    return torch.device("cpu")
