"""Fracture (classifier + detector) and brain MRI (classifier + tumour segmentation) pipelines."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.inference.registry import registry
from ml.common import BRAIN_CLASSES
from ml.data.transforms import preprocess

BRAIN_DISPLAY = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No tumour",
    "pituitary": "Pituitary tumour",
}


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def analyze_fracture(img: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {"findings": [], "detections": []}
    h = registry.get("fracture_cls")
    if h is not None:
        t = h.metrics.get("test", {})
        temp = float(t.get("temperature", 1.0) or 1.0)
        thr = float(t.get("threshold", 0.5) or 0.5)
        logit = float(h.run(preprocess(img, h.img_size))[0][0, 0])
        p = float(1 / (1 + np.exp(-logit / temp)))
        from app.inference.chest import certainty

        out["findings"].append(
            {
                "label": "Fracture",
                "key": "fracture",
                "probability": round(p, 4),
                "threshold": thr,
                "positive": p >= thr,
                "certainty": certainty(p, thr),
                "uncertainty": 0.0,
                "ratio_to_threshold": round(p / thr, 3),
                "kind": "classification",
            }
        )
    d = registry.get("fracture_det")
    if d is not None:
        import torch

        from ml.data.datasets import letterbox

        model = d.torch_model()
        size = d.img_size
        g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        lb, _ = letterbox(g, np.zeros((0, 4), np.float32), size)
        s = size / max(g.shape)
        with d.lock, torch.no_grad():
            res = model([torch.from_numpy(np.repeat(lb[None].astype(np.float32) / 255, 3, 0))])[0]
        thr = float(d.metrics.get("test", {}).get("score_threshold", 0.3) or 0.3)
        H, W = g.shape
        for box, sc in zip(res["boxes"].numpy(), res["scores"].numpy()):
            if sc < thr:
                continue
            x0, y0, x1, y1 = (box / s).tolist()
            out["detections"].append(
                {
                    "label": "Fracture",
                    "score": round(float(sc), 3),
                    "box": [x0 / W, y0 / H, x1 / W, y1 / H],
                }
            )
        out["detections"] = out["detections"][:5]
        if out["findings"] and out["detections"]:
            f = out["findings"][0]
            f["box"] = out["detections"][0]["box"]
            f["detector_score"] = out["detections"][0]["score"]
    return out


def analyze_brain(img: np.ndarray) -> dict[str, Any]:
    h = registry.get("brain_cls")
    if h is None:
        return {"findings": [], "error": "brain classifier not trained"}
    temp = float(h.metrics.get("test", {}).get("temperature", 1.0) or 1.0)
    logits = h.run(preprocess(img, h.img_size))[0][0]
    p = _softmax(logits / temp)
    findings = []
    k = int(p.argmax())
    for i, c in enumerate(BRAIN_CLASSES):
        if c == "notumor":
            continue
        pi = float(p[i])
        positive = i == k
        findings.append(
            {
                "label": BRAIN_DISPLAY[c],
                "key": c,
                "probability": round(pi, 4),
                "threshold": 0.5,
                "positive": positive,
                "certainty": (
                    "Likely" if positive and pi >= 0.7 else "Possible" if positive else "Unlikely"
                ),
                "uncertainty": round(float(1 - p.max()), 4),
                "ratio_to_threshold": round(pi / 0.5, 3),
                "kind": "classification",
            }
        )
    findings.sort(key=lambda f: -f["probability"])
    return {
        "findings": findings,
        "class_probs": {BRAIN_DISPLAY[c]: round(float(v), 4) for c, v in zip(BRAIN_CLASSES, p)},
        "predicted": BRAIN_DISPLAY[BRAIN_CLASSES[k]],
        "no_tumour_prob": round(float(p[2]), 4),
    }
