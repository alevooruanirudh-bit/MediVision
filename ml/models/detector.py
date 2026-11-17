"""Faster R-CNN (MobileNetV3-Large FPN) for fracture / pneumonia boxes, trained by us.

Init: generic COCO detection weights (everyday objects) when PRETRAINED_BACKBONE=true, else random.
"""

from __future__ import annotations

import torch.nn as nn

from ml.common import pretrained_backbone


def build_detector(
    num_classes: int = 2, min_size: int = 512, max_size: int = 512, pretrained: bool | None = None
) -> nn.Module:
    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_fpn,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    pre = pretrained_backbone() if pretrained is None else pretrained
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1 if pre else None,
        weights_backbone=None,
        min_size=min_size,
        max_size=max_size,
        box_detections_per_img=20,
    )
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes)
    return model
