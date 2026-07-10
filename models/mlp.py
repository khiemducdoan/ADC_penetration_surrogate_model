"""MLP surrogate: (log10 c0, log10 D, log10 r, t) -> C(x, t) profile."""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU}


class MLPSurrogate(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 100,
        hidden_dims: Sequence[int] = (256, 256, 256),
        activation: str = "gelu",
    ):
        super().__init__()
        act_cls = _ACTIVATIONS[activation]

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act_cls())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
