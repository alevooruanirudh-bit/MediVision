"""Small SRGAN-style x2 super-resolution: residual generator + PatchGAN discriminator (ours).

The generator predicts a residual on top of bicubic upsampling, so with no training it is exactly
bicubic; the adversarial term is weighted lightly to sharpen texture without inventing anatomy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Res(nn.Module):
    def __init__(self, w: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(w, w, 3, padding=1), nn.PReLU(w), nn.Conv2d(w, w, 3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.2 * self.body(x)


class SRGenerator(nn.Module):
    """(B,1,h,w) in [0,1] -> (B,1,s*h,s*w) in [0,1]."""

    def __init__(self, width: int = 48, blocks: int = 6, scale: int = 2) -> None:
        super().__init__()
        self.scale = scale
        self.head = nn.Sequential(nn.Conv2d(1, width, 3, padding=1), nn.PReLU(width))
        self.body = nn.Sequential(*[_Res(width) for _ in range(blocks)])
        self.up = nn.Sequential(
            nn.Conv2d(width, width * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.PReLU(width),
            nn.Conv2d(width, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        f = self.head(x)
        return (base + self.up(f + self.body(f))).clamp(0, 1)


class PatchDiscriminator(nn.Module):
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        w = width

        def blk(i: int, o: int, s: int) -> list[nn.Module]:
            return [nn.Conv2d(i, o, 3, s, 1), nn.BatchNorm2d(o), nn.LeakyReLU(0.2, True)]

        self.net = nn.Sequential(
            nn.Conv2d(1, w, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            *blk(w, w, 2),
            *blk(w, 2 * w, 1),
            *blk(2 * w, 2 * w, 2),
            *blk(2 * w, 4 * w, 1),
            nn.Conv2d(4 * w, 1, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
