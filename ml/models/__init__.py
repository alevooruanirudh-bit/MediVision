"""Model factory: build any MediVision architecture from its config."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from ml.common import (
    BRAIN_CLASSES,
    NIH_LABELS,
    RETINA_GRADES,
    ROUTER_CLASSES,
    ROUTER_CLASSES_V1,
    ROUTER_CLASSES_V2,
)

LABEL_SETS = {
    "nih14": NIH_LABELS,
    "brain4": BRAIN_CLASSES,
    "router5": ROUTER_CLASSES_V1,
    "router8": ROUTER_CLASSES_V2,
    "router9": ROUTER_CLASSES,
    "retina5": RETINA_GRADES,
}


def labels_for(cfg: dict[str, Any]) -> list[str]:
    if cfg["task"] == "binary":
        return [cfg.get("label_name", "Fracture")]
    return LABEL_SETS.get(cfg.get("labels", ""), [])


def build_model(cfg: dict[str, Any], pretrained: bool | None = None) -> nn.Module:
    m = cfg["model"]
    arch = m["arch"]
    if arch == "classifier":
        from ml.models.classifier import ClassifierNet

        n = len(labels_for(cfg)) or 1
        return ClassifierNet(
            m["backbone"], n, m.get("dropout", 0.3), m.get("hidden", 512), pretrained
        )
    if arch == "cnn3d":
        from ml.models.cnn3d import SmallCNN3D

        return SmallCNN3D(len(labels_for(cfg)) or 1, m.get("width", 16), m.get("dropout", 0.3))
    if arch == "srgan":
        from ml.models.srgan import SRGenerator

        return SRGenerator(m.get("width", 48), m.get("blocks", 6), m.get("scale", 2))
    if arch == "smallcnn":
        from ml.models.classifier import SmallCNN

        return SmallCNN(len(labels_for(cfg)), m.get("width", 32), m.get("dropout", 0.3))
    if arch in ("unet", "attention_unet"):
        from ml.common import pretrained_backbone
        from ml.models.unet import build_segmenter

        pre = pretrained_backbone() if pretrained is None else pretrained
        return build_segmenter(
            arch, m.get("in_ch", 1), m.get("base", 32), m.get("depth", 4), m.get("encoder"), pre
        )
    if arch == "fasterrcnn_mobilenet":
        from ml.models.detector import build_detector

        return build_detector(2, cfg["img_size"], cfg["img_size"], pretrained)
    if arch == "dncnn":
        from ml.models.autoencoder import DnCNN

        return DnCNN(m.get("depth", 10), m.get("width", 48))
    if arch == "report_gen":
        from ml.models.report_gen import ReportGenerator

        return ReportGenerator(
            m["vocab_size"], m["d_model"], m["layers"], m["heads"], m["ffn"], m["max_len"] + 2
        )
    raise ValueError(f"unknown arch {arch}")
