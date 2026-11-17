"""Loss functions for class imbalance (ASL / weighted BCE) and segmentation (BCE + Dice)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """Asymmetric loss for multi-label classification (Ridnik et al., ICCV 2021)."""

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0, clip: float = 0.05) -> None:
        super().__init__()
        self.gn, self.gp, self.clip = gamma_neg, gamma_pos, clip

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        p_neg = (1 - p + self.clip).clamp(max=1) if self.clip else 1 - p
        lp = y * torch.log(p.clamp(min=1e-8))
        ln = (1 - y) * torch.log(p_neg.clamp(min=1e-8))
        w = torch.pow(1 - p, self.gp) * y + torch.pow(1 - p_neg, self.gn) * (1 - y)
        return -((lp + ln) * w).sum(1).mean()


class WeightedBCE(nn.Module):
    def __init__(self, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.register_buffer("pw", pos_weight if pos_weight is not None else torch.ones(1))

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pw)


class BCEDice(nn.Module):
    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, y)
        p = torch.sigmoid(logits)
        inter = (p * y).sum((1, 2, 3))
        dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
        return bce + dice.mean()


def build_loss(cfg: dict, pos_weight: torch.Tensor | None = None,
               class_weight: torch.Tensor | None = None) -> nn.Module:
    lc = cfg.get("loss", {})
    t = lc.get("type", "ce")
    if t == "asl":
        return AsymmetricLoss(lc.get("gamma_neg", 4), lc.get("gamma_pos", 0), lc.get("clip", 0.05))
    if t == "weighted_bce":
        return WeightedBCE(pos_weight)
    if t == "bce_dice":
        return BCEDice()
    return nn.CrossEntropyLoss(weight=class_weight, label_smoothing=lc.get("label_smoothing", 0.0))
