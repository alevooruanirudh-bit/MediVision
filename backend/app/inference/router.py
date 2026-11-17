"""Modality router (our small CNN) + doubtful-input warnings."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.inference.registry import registry
from ml.common import ROUTER_CLASSES
from ml.data.transforms import preprocess_rgb

PIPELINE_FOR = {"chest_xray": "chest", "bone_xray": "fracture", "brain_mri": "brain"}
# Radiographs and MRI are grayscale (colourfulness ~0, JPEG noise < 3). Real chroma => not a scan.
COLOR_LIMIT = 6.0
HINTS = {"chest": "chest_xray", "fracture": "bone_xray", "bone": "bone_xray", "brain": "brain_mri"}


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def colorfulness(img: np.ndarray) -> float:
    if img.ndim == 2:
        return 0.0
    r, g, b = (img[..., i].astype(np.float32) for i in range(3))
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(
        np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    )


def route(img: np.ndarray, hint: str | None = None) -> dict[str, Any]:
    h = registry.get("router")
    color = colorfulness(img)
    if h is None:
        label = HINTS.get(hint or "", "chest_xray")
        return {
            "label": label,
            "probs": {},
            "confidence": None,
            "pipeline": PIPELINE_FOR.get(label),
            "ood": False,
            "warning": "Modality router not trained — using the upload hint.",
            "colorfulness": color,
        }
    t = float(h.metrics.get("test", {}).get("temperature", 1.0) or 1.0)
    logits = h.run(preprocess_rgb(img, h.img_size))[0][0]
    p = softmax(logits / t)
    k = int(p.argmax())
    label = ROUTER_CLASSES[k]
    conf = float(p[k])
    warning = None
    ood = False
    if color > COLOR_LIMIT and label != "not_medical":
        ood = True
        warning = (
            "Colour photograph detected — this does not look like a medical image "
            f"(router: {label.replace('_', ' ')} {conf:.0%}). Analysis was skipped."
        )
        label = "not_medical"
    elif label == "not_medical":
        ood = True
        warning = (
            "This does not look like a medical image (router: not medical, "
            f"{conf:.0%}). Analysis was skipped; results would be meaningless."
        )
    elif label == "other_medical":
        ood = True
        warning = (
            "Medical image of an unsupported type (e.g. CT, ultrasound). "
            "Supported: chest X-ray, bone X-ray, brain MRI."
        )
    elif conf < 0.6:
        warning = f"Modality uncertain ({label.replace('_', ' ')} {conf:.0%}). Please verify."
    if hint and HINTS.get(hint) and HINTS[hint] != label and not ood:
        warning = (
            f"Upload was marked as '{hint}' but the router sees "
            f"{label.replace('_', ' ')} ({conf:.0%})."
        )
    return {
        "label": label,
        "probs": {c: float(v) for c, v in zip(ROUTER_CLASSES, p)},
        "confidence": conf,
        "pipeline": None if ood else PIPELINE_FOR.get(label),
        "ood": ood,
        "warning": warning,
        "colorfulness": round(color, 1),
    }
