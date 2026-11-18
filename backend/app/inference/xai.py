"""Explainability: Grad-CAM, Grad-CAM++ (pytorch-grad-cam on our PyTorch models) and occlusion
sensitivity (model-agnostic, ONNX). Produces transparent PNG overlays + hottest-region box."""

from __future__ import annotations

import threading
from typing import Any

import cv2
import numpy as np

from app.inference.registry import ModelHandle
from ml.data.transforms import (
    LETTERBOX_DATASETS,
    letterbox_square,
    preprocess,
    preprocess_rgb,
    unletterbox,
)


def _prep(handle: ModelHandle, img: np.ndarray, size: int | None = None) -> np.ndarray:
    size = size or handle.img_size
    if handle.cfg.get("dataset") == "retina":
        return preprocess_rgb(img if img.ndim == 3 else np.stack([img] * 3, -1), size)
    if handle.cfg.get("dataset") in LETTERBOX_DATASETS:
        return preprocess(letterbox_square(img, size)[0], size)
    return preprocess(img, size)


def _to_image(m: np.ndarray, handle: ModelHandle, img: np.ndarray) -> np.ndarray:
    """Model-space map -> image resolution (undoing the letterbox when the model used one)."""
    if handle.cfg.get("dataset") in LETTERBOX_DATASETS:
        size = handle.img_size
        return unletterbox(m, letterbox_square(img, size)[1], size, img.shape[:2])
    return cv2.resize(
        m.astype(np.float32), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC
    )


_lock = threading.Lock()

COLORMAPS = {
    "inferno": cv2.COLORMAP_INFERNO,
    "jet": cv2.COLORMAP_JET,
    "turbo": cv2.COLORMAP_TURBO,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "magma": cv2.COLORMAP_MAGMA,
}


def cam(
    handle: ModelHandle,
    img: np.ndarray,
    class_idx: int,
    method: str = "gradcam",
    binary_single: bool = False,
) -> np.ndarray:
    """Class activation map in [0,1] at the input image's resolution."""
    import torch
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    model = handle.torch_model()
    layer = model.cam_layer()
    x = torch.from_numpy(_prep(handle, img))
    cls = GradCAMPlusPlus if method == "gradcampp" else GradCAM
    with _lock, cls(model=model, target_layers=[layer]) as c:
        g = c(input_tensor=x, targets=[ClassifierOutputTarget(class_idx)])[0]
    g = _to_image(g, handle, img)
    g = cv2.GaussianBlur(g, (0, 0), max(1.0, img.shape[0] / 150))
    g -= g.min()
    return g / g.max() if g.max() > 0 else g


def occlusion(
    handle: ModelHandle,
    img: np.ndarray,
    class_idx: int,
    grid: int = 10,
    activation: str = "sigmoid",
) -> np.ndarray:
    """Probability drop when each patch is occluded (batched through ONNX/PyTorch)."""
    size = handle.img_size
    base_in = _prep(handle, img, size)
    patch = size // grid
    batch, coords = [], []
    for gy in range(grid):
        for gx in range(grid):
            x = base_in.copy()
            x[
                ...,
                gy * patch : (gy + 1) * patch + patch // 2,
                gx * patch : (gx + 1) * patch + patch // 2,
            ] = 0
            batch.append(x[0])
            coords.append((gy, gx))
    X = np.stack(batch)

    def prob(z: np.ndarray) -> np.ndarray:
        if activation == "softmax":
            e = np.exp(z - z.max(1, keepdims=True))
            return (e / e.sum(1, keepdims=True))[:, class_idx]
        return 1 / (1 + np.exp(-z[:, class_idx]))

    base = prob(handle.run(base_in)[0])[0]
    drops = np.concatenate([prob(handle.run(X[i : i + 32])[0]) for i in range(0, len(X), 32)])
    heat = np.zeros((grid, grid), np.float32)
    for (gy, gx), p in zip(coords, drops):
        heat[gy, gx] = max(0.0, base - p)
    heat = np.clip(_to_image(heat, handle, img), 0, None)
    return heat / heat.max() if heat.max() > 0 else heat


def overlay(heat: np.ndarray, cmap: str = "inferno", min_alpha: float = 0.15) -> np.ndarray:
    """RGBA overlay: colour from the colormap, alpha grows with activation (transparent background)."""
    h8 = (np.clip(heat, 0, 1) * 255).astype(np.uint8)
    rgb = cv2.cvtColor(
        cv2.applyColorMap(h8, COLORMAPS.get(cmap, cv2.COLORMAP_INFERNO)), cv2.COLOR_BGR2RGB
    )
    a = np.clip((heat - min_alpha) / (1 - min_alpha), 0, 1) ** 0.8
    return np.dstack([rgb, (a * 230).astype(np.uint8)])


def hottest(heat: np.ndarray, frac: float = 0.6) -> dict[str, Any]:
    h, w = heat.shape
    y, x = np.unravel_index(int(np.argmax(heat)), heat.shape)
    m = (heat >= frac * heat.max()).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    comp = lab[y, x] if n > 1 else 0
    if comp:
        bx, by, bw, bh = stats[comp, :4]
    else:
        bx, by, bw, bh = max(0, x - w // 10), max(0, y - h // 10), w // 5, h // 5
    return {
        "peak": [float(x / w), float(y / h)],
        "peak_px": [int(x), int(y)],
        "box": [float(bx / w), float(by / h), float((bx + bw) / w), float((by + bh) / h)],
    }


def diff_overlay(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Diverging overlay for heatmap change (b - a): red = increased, blue = decreased."""
    d = np.clip(b - a, -1, 1)
    rgba = np.zeros((*d.shape, 4), np.uint8)
    pos, neg = d > 0.1, d < -0.1
    rgba[pos] = (239, 68, 68, 0)
    rgba[neg] = (59, 130, 246, 0)
    rgba[..., 3] = (np.abs(d) * 220).astype(np.uint8) * (pos | neg)
    return rgba
