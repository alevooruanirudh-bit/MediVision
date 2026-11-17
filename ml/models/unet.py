"""Our U-Net and Attention U-Net (Oktay et al. 2018) implementations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, i: int, o: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1, bias=False),
            nn.BatchNorm2d(o),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout else nn.Identity(),
            nn.Conv2d(o, o, 3, padding=1, bias=False),
            nn.BatchNorm2d(o),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionGate(nn.Module):
    """Additive attention gate: the decoder signal g re-weights the skip connection x."""

    def __init__(self, g_ch: int, x_ch: int, inter: int) -> None:
        super().__init__()
        self.wg = nn.Sequential(nn.Conv2d(g_ch, inter, 1, bias=False), nn.BatchNorm2d(inter))
        self.wx = nn.Sequential(nn.Conv2d(x_ch, inter, 1, bias=False), nn.BatchNorm2d(inter))
        self.psi = nn.Sequential(nn.Conv2d(inter, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        a = F.relu(self.wg(g) + self.wx(x), inplace=True)
        return x * self.psi(a)


class UNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        base: int = 32,
        depth: int = 4,
        attention: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        chs = [base * 2**i for i in range(depth + 1)]
        self.downs = nn.ModuleList()
        prev = in_ch
        for c in chs[:-1]:
            self.downs.append(ConvBlock(prev, c))
            prev = c
        self.bottleneck = ConvBlock(chs[-2], chs[-1], dropout)
        self.ups = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.decs = nn.ModuleList()
        for c in reversed(chs[:-1]):
            self.ups.append(nn.ConvTranspose2d(c * 2, c, 2, stride=2))
            self.gates.append(AttentionGate(c, c, max(c // 2, 8)) if attention else nn.Identity())
            self.decs.append(ConvBlock(c * 2, c))
        self.out = nn.Conv2d(chs[0], out_ch, 1)
        self.attention = attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for d in self.downs:
            x = d(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, gate, dec, skip in zip(self.ups, self.gates, self.decs, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            s = gate(x, skip) if self.attention else skip
            x = dec(torch.cat([x, s], 1))
        return self.out(x)


def build_segmenter(
    arch: str,
    in_ch: int = 1,
    base: int = 32,
    depth: int = 4,
    encoder: str | None = None,
    pretrained: bool = True,
) -> nn.Module:
    if encoder:  # optional ResNet encoder via segmentation-models-pytorch (ImageNet or random init)
        import segmentation_models_pytorch as smp

        return smp.Unet(
            encoder,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_ch,
            classes=1,
        )
    return UNet(in_ch, 1, base, depth, attention=(arch == "attention_unet"))
