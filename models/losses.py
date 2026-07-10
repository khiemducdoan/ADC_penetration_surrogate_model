"""Loss functions available for surrogate training, selected via `training.loss`."""
from __future__ import annotations

from torch import nn

_LOSSES = {
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
    "smooth_l1": nn.SmoothL1Loss,
}


def get_loss(name: str = "mse") -> nn.Module:
    try:
        return _LOSSES[name]()
    except KeyError as e:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(_LOSSES)}") from e
