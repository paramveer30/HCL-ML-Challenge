"""Loss weights per class, so rare emotions are not drowned out by neutral."""
import torch


def class_weights(counts, scheme):
    """counts: number of training examples per class. Returns a float tensor with mean 1."""
    counts = torch.tensor(counts, dtype=torch.float32)
    if scheme == "none":
        weights = torch.ones_like(counts)
    elif scheme == "inv":
        weights = 1.0 / counts
    elif scheme == "sqrt_inv":
        weights = 1.0 / counts.sqrt()
    else:
        raise ValueError(f"unknown class weighting: {scheme}")
    return weights / weights.mean()
