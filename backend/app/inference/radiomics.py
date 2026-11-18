"""Imaging biomarkers / radiomics (IBSI-style subset, implemented here with numpy + scikit-image).

Shape (area, perimeter, circularity, eccentricity, solidity, equivalent diameter), first-order
(mean, SD, skewness, kurtosis, entropy, percentiles) and GLCM texture (contrast, homogeneity,
energy, correlation, dissimilarity; 4 directions, 32 grey levels) inside a region of interest.
Values are descriptive only — they are not validated prognostic biomarkers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

LEVELS = 32


def first_order(vals: np.ndarray) -> dict[str, float]:
    v = vals.astype(np.float64)
    if v.size < 2:
        return {}
    mu, sd = float(v.mean()), float(v.std())
    z = (v - mu) / (sd + 1e-9)
    hist, _ = np.histogram(v, bins=LEVELS, range=(0, 256))
    p = hist / max(hist.sum(), 1)
    ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    return {
        "mean": round(mu, 2),
        "std": round(sd, 2),
        "skewness": round(float((z**3).mean()), 3),
        "kurtosis": round(float((z**4).mean() - 3), 3),
        "entropy_bits": round(ent, 3),
        "p10": round(float(np.percentile(v, 10)), 1),
        "p90": round(float(np.percentile(v, 90)), 1),
    }


def shape(mask: np.ndarray, spacing_mm: float | None = None) -> dict[str, Any]:
    from skimage.measure import label, regionprops

    lab = label(mask.astype(np.uint8))
    props = regionprops(lab)
    if not props:
        return {}
    area = sum(p.area for p in props)
    perim = sum(p.perimeter for p in props)
    big = max(props, key=lambda p: p.area)
    s = spacing_mm or 1.0
    out = {
        "components": len(props),
        "area_px": int(area),
        "perimeter_px": round(float(perim), 1),
        "circularity": round(float(4 * np.pi * area / max(perim, 1) ** 2), 3),
        "eccentricity": round(float(big.eccentricity), 3),
        "solidity": round(float(big.solidity), 3),
        "equivalent_diameter_px": round(float(big.equivalent_diameter_area), 1),
    }
    if spacing_mm:
        out["area_mm2"] = round(area * s * s, 1)
        out["equivalent_diameter_mm"] = round(float(big.equivalent_diameter_area) * s, 1)
    return out


def glcm(img: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    from skimage.feature import graycomatrix, graycoprops

    ys, xs = np.where(mask)
    if len(ys) < 16:
        return {}
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = (img[y0:y1, x0:x1].astype(np.int32) * LEVELS // 256).clip(0, LEVELS - 1)
    m = mask[y0:y1, x0:x1]
    # pixels outside the ROI get an extra level that is dropped from the matrix
    q = np.where(m, crop, LEVELS).astype(np.uint8)
    g = graycomatrix(q, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], LEVELS + 1, symmetric=True)[
        :LEVELS, :LEVELS
    ]
    g = g.astype(np.float64)
    g /= np.maximum(g.sum((0, 1), keepdims=True), 1)
    return {
        k: round(float(graycoprops(g, k).mean()), 4)
        for k in ("contrast", "homogeneity", "energy", "correlation", "dissimilarity")
    }


def features(
    img: np.ndarray, mask: np.ndarray | None, roi: str, spacing_mm: float | None = None
) -> dict[str, Any]:
    g = img if img.ndim == 2 else img.mean(-1).astype(np.uint8)
    if mask is None or mask.sum() < 16:
        mask = np.ones_like(g, bool)
        roi = f"{roi} (whole image — no segmentation)"
    if mask.shape != g.shape:
        import cv2

        mask = cv2.resize(mask.astype(np.uint8), (g.shape[1], g.shape[0]), cv2.INTER_NEAREST) > 0
    mask = mask.astype(bool)
    return {
        "roi": roi,
        "shape": shape(mask, spacing_mm),
        "first_order": first_order(g[mask]),
        "texture_glcm": glcm(g, mask),
        "note": "Descriptive radiomics (IBSI-style subset). Not a validated biomarker.",
    }


def features_3d(vol: np.ndarray, mask: np.ndarray, spacing: tuple[float, ...]) -> dict[str, Any]:
    """3-D shape + first-order features and 2-D GLCM on the axial slice with most ROI."""
    from skimage.measure import marching_cubes, mesh_surface_area

    out: dict[str, Any] = {"roi": "lung nodule (3-D)", "first_order": {}, "shape": {}}
    if mask.sum() < 8:
        return out
    out["first_order"] = first_order(vol[mask])
    try:
        verts, faces, *_ = marching_cubes(mask.astype(np.float32), 0.5, spacing=spacing)
        sa = float(mesh_surface_area(verts, faces))
        v = float(mask.sum() * np.prod(spacing))
        out["shape"] = {
            "volume_mm3": round(v, 1),
            "surface_area_mm2": round(sa, 1),
            "sphericity": round(float(np.pi ** (1 / 3) * (6 * v) ** (2 / 3) / max(sa, 1e-6)), 3),
        }
    except (ValueError, RuntimeError):
        pass
    z = int(mask.sum((1, 2)).argmax())
    out["texture_glcm"] = glcm(vol[z], mask[z])
    out["note"] = "Descriptive radiomics (IBSI-style subset). Not a validated biomarker."
    return out
