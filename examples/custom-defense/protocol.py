"""A preprocessing example, not a recommended or guaranteed effective defense."""
import torch
import torch.nn.functional as F


def fit(model, images, labels, *, seed, config):
    # Optional: implement your training algorithm here using ONLY supplied
    # training images/labels. Update model parameters in place. The organizer
    # starts each of five runs from its original checkpoint and a fresh process.
    # Return JSON-serializable preprocessing state; no pickle or model objects.
    # This example needs no training and leaves model weights unchanged.
    return {}


def preprocess(images, *, state, config):
    # Replace this illustrative smoothing operation with your defense layer.
    # This runs BEFORE the model's fixed preprocessing and quantum encoding.
    # Keep shape (N,3,32,32), float64, finite values, and range [0,1].
    strength = float(config.get('strength', 0.1))
    if not 0 <= strength <= 1:
        raise ValueError('strength must be in [0,1]')
    smoothed = F.avg_pool2d(F.pad(images, (1, 1, 1, 1), mode='replicate'), 3, stride=1)
    return (1 - strength) * images + strength * smoothed
