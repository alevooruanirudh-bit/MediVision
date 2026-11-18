"""New modality pipelines (D-21): head-CT hemorrhage, breast ultrasound, fundus DR grading and
chest-CT nodule volumes. All models are ours (see ml/configs/*.yaml)."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.inference.chest import certainty, mc_dropout_head, sigmoid
from app.inference.registry import ModelHandle, registry
from ml.common import RETINA_GRADES
from ml.data.transforms import LETTERBOX_DATASETS, letterbox_square, preprocess, preprocess_rgb

PIPELINE_MODEL = {
    "head_ct": "head_ct_cls",
    "breast_us": "breast_us",
    "fundus": "retina_dr",
    "mammo": "mammo_cls",
    "ct_volume": "nodule3d",
    "ct_chest_scan": "lung_nodule_det",
    "petct": "petct_seg",
}


def prep(h: ModelHandle, img: np.ndarray) -> np.ndarray:
    """Model-specific pre-processing (colour for the fundus grader, grayscale otherwise)."""
    if h.cfg.get("dataset") == "retina":
        rgb = img if img.ndim == 3 else np.stack([img] * 3, -1)
        return preprocess_rgb(rgb, h.img_size)
    if h.cfg.get("dataset") in LETTERBOX_DATASETS:
        return preprocess(letterbox_square(img, h.img_size)[0], h.img_size)
    return preprocess(img, h.img_size)


def _binary(
    name: str, img: np.ndarray, label: str, key: str, samples: int = 8
) -> dict[str, Any] | None:
    h = registry.get(name)
    if h is None:
        return None
    t = h.metrics.get("test", {})
    temp = float(t.get("temperature", 1.0) or 1.0)
    thr = float(t.get("threshold", 0.5) or 0.5)
    logits, *rest = h.run(prep(h, img))
    p = float(sigmoid(float(logits[0, 0]) / temp))
    unc = 0.0
    if rest and h.head is not None and samples:
        mc = mc_dropout_head(rest[0][0], h.head, samples, np.random.default_rng(0))
        unc = float(sigmoid(mc[:, 0] / temp).std())
    return {
        "label": label,
        "key": key,
        "probability": round(p, 4),
        "threshold": round(thr, 4),
        "positive": p >= thr,
        "certainty": certainty(p, thr),
        "uncertainty": round(unc, 4),
        "ratio_to_threshold": round(p / thr, 3),
        "kind": "classification",
        "model": name,
    }


def analyze_head_ct(img: np.ndarray, samples: int = 8) -> dict[str, Any]:
    f = _binary("head_ct_cls", img, "Intracranial hemorrhage", "hemorrhage", samples)
    return {"findings": [f] if f else [], "error": None if f else "head-CT model not trained"}


def analyze_breast_us(img: np.ndarray, samples: int = 8) -> dict[str, Any]:
    f = _binary("breast_us", img, "Suspicious breast lesion", "malignant", samples)
    if f:
        f["risk_category"] = (
            "high suspicion (BI-RADS-like 4C-5 range)"
            if f["probability"] >= 0.8
            else (
                "intermediate suspicion (4A-4B range)"
                if f["positive"]
                else "low suspicion (2-3 range)"
            )
        )
    return {"findings": [f] if f else [], "error": None if f else "breast-US model not trained"}


def analyze_mammo(img: np.ndarray, samples: int = 8) -> dict[str, Any]:
    """Single screening view. Mini-DDSM labels are per exam, so this is a view-level suspicion score
    (the model card reports exam-level metrics where the 4 views are averaged)."""
    f = _binary(
        "mammo_cls", img, "Suspicious for breast cancer (mammography)", "breast_cancer", samples
    )
    if f:
        f["risk_category"] = (
            "high suspicion - recall for diagnostic work-up"
            if f["probability"] >= 0.8
            else ("suspicious - recall suggested" if f["positive"] else "no AI-flagged suspicion")
        )
        f["view_level"] = True
    return {"findings": [f] if f else [], "error": None if f else "mammography model not trained"}


def analyze_fundus(img: np.ndarray) -> dict[str, Any]:
    h = registry.get("retina_dr")
    if h is None:
        return {"findings": [], "error": "fundus model not trained"}
    temp = float(h.metrics.get("test", {}).get("temperature", 1.0) or 1.0)
    z = h.run(prep(h, img))[0][0] / temp
    p = np.exp(z - z.max())
    p = p / p.sum()
    grade = int(p.argmax())
    referable = float(p[2:].sum())  # ICDR >= moderate NPDR = referable DR
    expected = float((p * np.arange(len(p))).sum())
    f = {
        "label": "Referable diabetic retinopathy",
        "key": "referable_dr",
        "probability": round(referable, 4),
        "threshold": 0.5,
        "positive": referable >= 0.5,
        "certainty": certainty(referable, 0.5),
        "uncertainty": round(float(1 - p.max()), 4),
        "ratio_to_threshold": round(referable / 0.5, 3),
        "kind": "classification",
        "model": "retina_dr",
        "grade": RETINA_GRADES[grade],
        "grade_index": grade,
        "expected_grade": round(expected, 2),
    }
    return {
        "findings": [f],
        "grade_probs": {g: round(float(v), 4) for g, v in zip(RETINA_GRADES, p)},
        "predicted_grade": RETINA_GRADES[grade],
    }


# ---------------------------------------------------------------------------------- CT volumes
def normalise_volume(vol: np.ndarray) -> np.ndarray:
    """Any 3-D array -> uint8 (D,H,W). HU-like input is windowed to the lung window."""
    v = np.asarray(vol, dtype=np.float32)
    if v.min() < -200:  # Hounsfield units
        v = np.clip((v + 1350) / 1500, 0, 1) * 255  # lung window W1500 / L-600
    elif v.max() <= 1.0:
        v = v * 255
    return np.clip(v, 0, 255).astype(np.uint8)


def resample(vol: np.ndarray, size: int = 64) -> np.ndarray:
    from scipy.ndimage import zoom

    if vol.shape == (size, size, size):
        return vol
    f = [size / s for s in vol.shape]
    return zoom(vol.astype(np.float32), f, order=1).clip(0, 255).astype(np.uint8)


def nodule_segment(vol: np.ndarray) -> np.ndarray:
    """Classical nodule segmentation: Otsu threshold, keep the component nearest the centre."""
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    v = ndimage.gaussian_filter(vol.astype(np.float32), 1.0)
    try:
        t = threshold_otsu(v)
    except ValueError:
        return np.zeros_like(vol, bool)
    m = v > t
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    c = np.array(vol.shape) / 2
    best, bd = 0, 1e9
    for i, com in enumerate(ndimage.center_of_mass(m, lab, range(1, n + 1)), start=1):
        d = float(np.linalg.norm(np.array(com) - c))
        if d < bd:
            best, bd = i, d
    return ndimage.binary_fill_holes(lab == best)


def nodule_volumetrics(mask: np.ndarray, spacing_mm: tuple[float, float, float]) -> dict[str, Any]:
    vox = float(np.prod(spacing_mm))
    n = int(mask.sum())
    vol = n * vox
    d_eq = float((6 * vol / np.pi) ** (1 / 3)) if vol > 0 else 0.0
    zs = np.where(mask.any((1, 2)))[0]
    return {
        "available": n > 0,
        "voxels": n,
        "volume_mm3": round(vol, 1),
        "equivalent_diameter_mm": round(d_eq, 1),
        "slices_with_nodule": int(len(zs)),
        "spacing_mm": list(spacing_mm),
        "method": "Otsu threshold + central connected component (classical, not learned)",
    }


def gradcam3d(h: ModelHandle, x: np.ndarray) -> np.ndarray:
    """Grad-CAM for our 3-D CNN: weights = mean gradient per channel of the last block."""
    import torch

    model = h.torch_model()
    layer = model.cam_layer()
    store: dict[str, torch.Tensor] = {}
    fh = layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    bh = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("g", go[0]))
    try:
        with h.lock:
            t = torch.from_numpy(x).requires_grad_(True)
            out = model(t)
            model.zero_grad()
            out[0, 0].backward()
    finally:
        fh.remove()
        bh.remove()
    a, g = store["a"][0].detach(), store["g"][0].detach()
    cam = torch.relu((g.mean((1, 2, 3))[:, None, None, None] * a).sum(0))
    cam = torch.nn.functional.interpolate(
        cam[None, None], size=x.shape[2:], mode="trilinear", align_corners=False
    )[0, 0].numpy()
    cam -= cam.min()
    return cam / cam.max() if cam.max() > 0 else cam


def analyze_volume(
    vol: np.ndarray, spacing_mm: tuple[float, float, float] | None = None
) -> dict[str, Any]:
    v8 = resample(normalise_volume(vol), 64)
    sp = spacing_mm or (1.0, 1.0, 1.0)
    # spacing of the 64^3 grid after resampling
    sp64 = tuple(float(s * o / 64) for s, o in zip(sp, vol.shape))
    mask = nodule_segment(v8)
    vm = nodule_volumetrics(mask, sp64)  # type: ignore[arg-type]
    vm["spacing_assumed"] = spacing_mm is None
    out: dict[str, Any] = {"findings": [], "volume": v8, "mask": mask, "volumetrics": vm}
    h = registry.get("nodule3d")
    if h is None:
        out["error"] = "nodule model not trained"
        return out
    x = ((v8.astype(np.float32) / 255 - 0.5) / 0.25)[None, None]
    t = h.metrics.get("test", {})
    temp = float(t.get("temperature", 1.0) or 1.0)
    thr = float(t.get("threshold", 0.5) or 0.5)
    p = float(sigmoid(float(h.run(x)[0][0, 0]) / temp))
    out["findings"] = [
        {
            "label": "Malignancy-suspicious lung nodule",
            "key": "malignant_nodule",
            "probability": round(p, 4),
            "threshold": round(thr, 4),
            "positive": p >= thr,
            "certainty": certainty(p, thr),
            "uncertainty": 0.0,
            "ratio_to_threshold": round(p / thr, 3),
            "kind": "classification",
            "model": "nodule3d",
            "volume_mm3": vm["volume_mm3"],
        }
    ]
    try:
        out["cam"] = gradcam3d(h, x)
    except Exception:  # pragma: no cover
        out["cam"] = None
    return out


def mpr_slices(vol: np.ndarray) -> dict[str, np.ndarray]:
    d, h, w = vol.shape
    return {"axial": vol[d // 2], "coronal": vol[:, h // 2, :], "sagittal": vol[:, :, w // 2]}


def mip_frames(vol: np.ndarray, mask: np.ndarray | None, n: int = 24) -> list[np.ndarray]:
    """Rotating maximum-intensity projections (3-D view) with the nodule mask tinted."""
    from scipy.ndimage import rotate

    frames = []
    v = vol.astype(np.float32)
    m = mask.astype(np.float32) if mask is not None else None
    for k in range(n):
        ang = 360 * k / n
        r = rotate(v, ang, axes=(1, 2), reshape=False, order=1)
        mip = r.max(1)
        rgb = np.stack([mip] * 3, -1)
        if m is not None:
            rm = rotate(m, ang, axes=(1, 2), reshape=False, order=0).max(1) > 0.5
            rgb[rm] = 0.55 * rgb[rm] + 0.45 * np.array([167, 139, 250])
        frames.append(np.clip(rgb, 0, 255).astype(np.uint8))
    return frames
