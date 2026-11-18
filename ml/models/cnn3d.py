"""Our compact 3-D CNN for chest-CT nodule patches (64^3), trained from scratch."""

from __future__ import annotations

import torch
import torch.nn as nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm3d(cout),
        nn.ReLU(inplace=True),
        nn.Conv3d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm3d(cout),
        nn.ReLU(inplace=True),
    )


class SmallCNN3D(nn.Module):
    """4 conv stages (w, 2w, 4w, 8w) with max-pooling -> GAP -> dropout -> linear.

    `features` is the last spatial block (8^3 for a 64^3 input) and is the Grad-CAM layer.
    """

    def __init__(self, num_classes: int = 1, width: int = 16, dropout: float = 0.3) -> None:
        super().__init__()
        w = width
        self.stem = nn.Sequential(_block(1, w), nn.MaxPool3d(2))
        self.s2 = nn.Sequential(_block(w, 2 * w), nn.MaxPool3d(2))
        self.s3 = nn.Sequential(_block(2 * w, 4 * w), nn.MaxPool3d(2))
        self.features = _block(4 * w, 8 * w)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(8 * w, num_classes))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(self.s3(self.s2(self.stem(x)))).mean((2, 3, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))

    def cam_layer(self) -> nn.Module:
        return self.features
