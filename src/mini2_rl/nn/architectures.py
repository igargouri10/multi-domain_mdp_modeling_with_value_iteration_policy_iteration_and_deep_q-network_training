r"""Neural network architectures used by the deep RL agents (V2 requirement).

The directory is deliberately minimal at this stage: the grid-world foundation
environment has small observations (one-hot cell position or a small grid
image), so we only really need an MLP and a shallow CNN. When the course moves
to richer foundation environments (Atari-style pixel inputs, continuous
control), this module should grow to house the architectures in their own
files (``dueling.py``, ``resnet.py``, ``actor_critic.py``, ...).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    r"""A generic fully-connected network, suitable for DQN / actor-critic."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden: Sequence[int] = (128, 128),
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvQNet(nn.Module):
    r"""A shallow CNN used when grid observations are passed as ``(1, H, W)``."""

    def __init__(self, in_channels: int, out_dim: int, h: int, w: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Linear(32 * h * w, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)
        return self.head(z.flatten(start_dim=1))
