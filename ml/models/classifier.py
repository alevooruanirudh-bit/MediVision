"""Image classifiers: generic backbone (timm, optional ImageNet init) + our MC-dropout head."""

from __future__ import annotations

import torch
import torch.nn as nn

from ml.common import pretrained_backbone

TIMM_NAMES = {
    "densenet121": "densenet121",
    "efficientnet_b2": "efficientnet_b2",
    "efficientnet_b0": "efficientnet_b0",
    "convnext_tiny": "convnext_tiny",
    "resnet18": "resnet18",
}


class ClassifierNet(nn.Module):
    """backbone -> global pool -> [Dropout -> Linear(hidden) -> GELU -> Dropout -> Linear(classes)].

    `forward_features` returns the pooled embedding so inference can run MC-dropout over the head
    cheaply (in numpy, on top of the ONNX backbone) without re-running the backbone.
    """

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        dropout: float = 0.3,
        hidden: int = 512,
        pretrained: bool | None = None,
        in_chans: int = 3,
    ) -> None:
        super().__init__()
        import timm

        pre = pretrained_backbone() if pretrained is None else pretrained
        self.backbone_name = backbone
        self.backbone = timm.create_model(
            TIMM_NAMES.get(backbone, backbone), pretrained=pre, num_classes=0, in_chans=in_chans
        )
        feat = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
        self.num_features = feat

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def cam_layer(self) -> nn.Module:
        """Last spatial feature layer for Grad-CAM."""
        b = self.backbone
        name = self.backbone_name
        if name.startswith("densenet"):
            return b.features
        if name.startswith("efficientnet"):
            return b.bn2
        if name.startswith("convnext"):
            return b.stages[-1]
        if name.startswith("resnet"):
            return b.layer4
        raise ValueError(name)


class SmallCNN(nn.Module):
    """Our own compact CNN for the modality router (trained from scratch, ~0.4M params)."""

    def __init__(self, num_classes: int, width: int = 32, dropout: float = 0.3) -> None:
        super().__init__()

        def block(i: int, o: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1, bias=False),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        w = width
        self.features = nn.Sequential(
            block(3, w), block(w, 2 * w), block(2 * w, 4 * w), block(4 * w, 8 * w)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(8 * w, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x)))

    def cam_layer(self) -> nn.Module:
        return self.features[-1]
