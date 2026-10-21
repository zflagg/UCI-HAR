"""Models that take input of shape (batch, channels, time) and return class logits."""
import torch
from torch import nn


class CNN1D(nn.Module):
    """Three conv blocks, then global average pooling over time."""

    def __init__(self, in_channels, n_classes, width=64, dropout=0.3):
        super().__init__()

        def block(c_in, c_out, k):
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, k, padding=k // 2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(),
            )

        self.features = nn.Sequential(
            block(in_channels, width, 7), nn.MaxPool1d(2),
            block(width, width * 2, 5), nn.MaxPool1d(2),
            block(width * 2, width * 2, 3),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width * 2, n_classes))

    def forward(self, x):
        x = self.features(x)          # (B, 2W, T/4)
        x = x.mean(dim=-1)            # global average pool over time
        return self.head(x)


class GRUClassifier(nn.Module):
    """Two-layer GRU over time; classifies from the final hidden state."""

    def __init__(self, in_channels, n_classes, hidden=64, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(in_channels, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):
        _, h = self.gru(x.transpose(1, 2))   # GRU wants (B, T, C)
        return self.head(h[-1])


MODELS = {"cnn": CNN1D, "gru": GRUClassifier}


def build_model(name, in_channels, n_classes):
    return MODELS[name](in_channels, n_classes)
