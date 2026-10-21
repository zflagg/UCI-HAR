"""Sensor failure modes, used both as training augmentation and as robustness tests.

All functions take and return a tensor of shape (B, C, T) in normalized units.
"""
import torch


def add_noise(x, sigma):
    """Gaussian noise, like a noisier sensor or a worse ADC."""
    return x + sigma * torch.randn_like(x)


def drop_channels(x, p):
    """Zero out each channel with probability p, like a dead or disconnected sensor."""
    keep = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) >= p).float()
    return x * keep


def add_drift(x, max_slope):
    """Linear drift over the window with a random slope per channel, like calibration drift."""
    ramp = torch.linspace(0, 1, x.shape[-1], device=x.device)
    slope = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) * 2 - 1) * max_slope
    return x + slope * ramp


PERTURBATIONS = {
    "noise": (add_noise, [0.0, 0.25, 0.5, 1.0]),
    "drop_channels": (drop_channels, [0.0, 0.1, 0.25, 0.5]),
    "drift": (add_drift, [0.0, 0.5, 1.0, 2.0]),
}


def augment(x, noise=0.1, drop_p=0.1, drift=0.25):
    """Mild random corruption applied to training batches when --augment is on."""
    return add_drift(drop_channels(add_noise(x, noise), drop_p), drift)
