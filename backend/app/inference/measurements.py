"""Segmentation-based measurements: lung masks, cardiothoracic ratio (CTR), lung areas, symmetry,
lung-zone mapping of heatmap peaks, and brain-tumour area.

Conventions: frontal chest X-rays are displayed in radiological orientation, so the patient's RIGHT
lung appears on the image LEFT. Coordinates returned to the UI are normalised to [0,1].
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.inference.registry import registry
from ml.data.transforms import preprocess


def segment(name: str, img: np.ndarray) -> np.ndarray | None:
    """Binary mask at the image's resolution, or None if the model is unavailable."""
    h = registry.get(name)
    if h is None:
        return None
    x = preprocess(img, h.img_size, channels=1)
    logit = h.run(x)[0][0, 0]
    prob = cv2.resize(1 / (1 + np.exp(-logit)), (img.shape[1], img.shape[0]))
    return (prob > 0.5).astype(np.uint8)


def _components(mask: np.ndarray, k: int = 2) -> list[np.ndarray]:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA])[:k] + 1
    min_area = mask.size * 0.01
    return [(lab == i).astype(np.uint8) for i in order if stats[i, cv2.CC_STAT_AREA] > min_area]


def lung_measurements(img: np.ndarray, spacing_mm: float | None = None) -> dict[str, Any]:
    mask = segment("lung_unet", img)
    if mask is None:
        return {"available": False}
    h, w = mask.shape
    comps = _components(mask, 2)
    if len(comps) < 2:
        return {
            "available": True,
            "mask": mask,
            "quality": "poor",
            "note": "Could not separate two lungs; measurements unavailable.",
        }
    cents = [np.argwhere(c)[:, 1].mean() for c in comps]
    right, left = (comps[0], comps[1]) if cents[0] < cents[1] else (comps[1], comps[0])
    both = ((right + left) > 0).astype(np.uint8)
    ys = np.where(both.any(1))[0]
    top, bot = int(ys.min()), int(ys.max())
    # Thoracic width: widest outer lung-to-lung distance.
    thorax_w, thorax_row, t_x0, t_x1 = 0, top, 0, 0
    for y in range(top, bot + 1):
        xs = np.where(both[y])[0]
        if len(xs) and xs.max() - xs.min() > thorax_w:
            thorax_w, thorax_row, t_x0, t_x1 = (
                int(xs.max() - xs.min()),
                y,
                int(xs.min()),
                int(xs.max()),
            )
    # Heart width estimate: medial gap between the lungs in the lower half, using only rows where BOTH
    # lungs are substantially present (>= 35% of their own max width), robust 90th percentile instead of
    # the max so a lung partly hidden by effusion/consolidation does not blow the gap up.
    rw, lw = right.sum(1), left.sum(1)
    rows = []
    for y in range(top + (bot - top) // 2, bot + 1):
        if rw[y] >= 0.35 * rw.max() and lw[y] >= 0.35 * lw.max():
            rx, lx = np.where(right[y])[0], np.where(left[y])[0]
            gap = int(lx.min() - rx.max())
            if gap > 0:
                rows.append((gap, y, int(rx.max()), int(lx.min())))
    heart_w, heart_row, h_x0, h_x1 = 0, bot, 0, 0
    if rows:
        gaps = np.array([r[0] for r in rows])
        target = np.percentile(gaps, 90)
        heart_w, heart_row, h_x0, h_x1 = min(rows, key=lambda r: abs(r[0] - target))
    ctr = heart_w / thorax_w if thorax_w else None
    ra, la = int(right.sum()), int(left.sum())
    px_area = (spacing_mm**2) if spacing_mm else None
    zones = {}
    for side, m in (("right", right), ("left", left)):
        yy = np.where(m.any(1))[0]
        zones[side] = {
            "top": int(yy.min()),
            "bottom": int(yy.max()),
            "x0": int(np.where(m.any(0))[0].min()),
            "x1": int(np.where(m.any(0))[0].max()),
        }
    return {
        "available": True,
        "mask": mask,
        "right_mask": right,
        "left_mask": left,
        "zones": zones,
        "ctr": round(ctr, 3) if ctr else None,
        "ctr_flag": bool(ctr and ctr > 0.5),
        "heart_width_px": heart_w,
        "thorax_width_px": thorax_w,
        "right_lung_area_px": ra,
        "left_lung_area_px": la,
        "right_lung_area_mm2": round(ra * px_area, 1) if px_area else None,
        "left_lung_area_mm2": round(la * px_area, 1) if px_area else None,
        "symmetry_index": round(min(ra, la) / max(ra, la), 3) if max(ra, la) else None,
        "lines": {
            "thorax": [[t_x0 / w, thorax_row / h], [t_x1 / w, thorax_row / h]],
            "heart": [[h_x0 / w, heart_row / h], [h_x1 / w, heart_row / h]],
        },
        "quality": "ok",
    }


def zone_of(x: float, y: float, lung: dict[str, Any] | None, w: int, h: int) -> str:
    """Map an image point (pixels) to 'right/left upper/middle/lower zone' (patient side)."""
    if not lung or not lung.get("zones"):
        side = "right" if x < w / 2 else "left"
        third = min(2, int(3 * y / h))
        return f"{side} {['upper', 'middle', 'lower'][third]} zone"
    xi, yi = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
    rm, lm = lung.get("right_mask"), lung.get("left_mask")
    side = None
    if rm is not None and rm[yi, xi]:
        side = "right"
    elif lm is not None and lm[yi, xi]:
        side = "left"
    if side is None:  # outside both lungs: nearest lung by horizontal distance, else mediastinum
        z = lung["zones"]
        if z["right"]["x1"] < xi < z["left"]["x0"]:
            return "cardiomediastinal region"
        side = "right" if abs(xi - z["right"]["x1"]) < abs(xi - z["left"]["x0"]) else "left"
    zz = lung["zones"][side]
    frac = (yi - zz["top"]) / max(zz["bottom"] - zz["top"], 1)
    third = ["upper", "middle", "lower"][int(np.clip(frac * 3, 0, 2))]
    return f"{side} {third} zone"


def tumour_measurements(img: np.ndarray, spacing_mm: float | None = None) -> dict[str, Any]:
    mask = segment("brain_seg", img)
    if mask is None:
        return {"available": False}
    comps = _components(mask, 3)
    area = int(mask.sum())
    out: dict[str, Any] = {
        "available": True,
        "mask": mask,
        "tumour_area_px": area,
        "tumour_fraction": round(area / mask.size, 4),
        "tumour_area_mm2": round(area * spacing_mm**2, 1) if spacing_mm else None,
        "components": len(comps),
    }
    if area:
        ys, xs = np.nonzero(mask)
        h, w = mask.shape
        out["box"] = [xs.min() / w, ys.min() / h, xs.max() / w, ys.max() / h]
        out["centroid"] = [float(xs.mean() / w), float(ys.mean() / h)]
        out["max_diameter_px"] = int(max(xs.max() - xs.min(), ys.max() - ys.min()))
    return out


def mask_overlay(
    mask: np.ndarray, color: tuple[int, int, int], alpha: int = 90, outline: bool = True
) -> np.ndarray:
    rgba = np.zeros((*mask.shape, 4), np.uint8)
    rgba[mask > 0] = (*color, alpha)
    if outline:
        cnts, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(rgba, cnts, -1, (*color, 255), max(1, mask.shape[0] // 300))
    return rgba
