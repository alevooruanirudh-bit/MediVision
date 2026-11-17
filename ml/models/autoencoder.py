"""DnCNN-style residual denoiser (Zhang et al. 2017), trained by us on simulated low-dose noise."""

from __future__ import annotations

import torch
import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, depth: int = 10, width: int = 48) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(1, width, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(width, width, 3, padding=1, bias=False), nn.BatchNorm2d(width),
                       nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(width, 1, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        """Input/output in [0,1]; the network predicts the noise residual."""
        return (noisy - self.net(noisy)).clamp(0, 1)
