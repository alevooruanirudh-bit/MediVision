"""Whole-scan chest-CT nodule detection (M5, D-36) and PET-CT fusion analysis (C7, D-34).

Both reuse the exact candidate / normalisation code used in training (ml/data/luna.py,
ml/petct_fusion.py). All models are ours.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.inference.chest import certainty
from app.inference.modalities import analyze_volume
from app.inference.registry import registry

WHOLE_SCAN_MIN_AXIS = (
    96  # volumes larger than this (any axis) are treated as scans, not nodule cubes
)


def is_whole_scan(vol: np.ndarray) -> bool:
    return max(vol.shape) > WHOLE_SCAN_MIN_AXIS


def _hu(vol: np.ndarray) -> np.ndarray:
    v = np.asarray(vol, np.float32)
    if v.min() >= 0 and v.max() <= 4096 and v.max() > 255:  # raw CT numbers stored with +1024
        v = v - 1024
    return v


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def _render_axial(
    vol_hu: np.ndarray, z: int, marks: list[tuple[int, int, float, bool]]
) -> np.ndarray:
    """Lung-window axial slice with detection circles (orange = above threshold, gray = below)."""
    sl = np.clip((vol_hu[z] + 1350) / 1500, 0, 1) * 255
    rgb = np.ascontiguousarray(np.stack([sl] * 3, -1).astype(np.uint8))
    for y, x, r, pos in marks:
        cv2.circle(
            rgb, (int(x), int(y)), int(max(r, 6)), (251, 146, 60) if pos else (148, 163, 184), 2
        )
    return rgb


def _mip_frames(
    vol_hu: np.ndarray, pts: list[tuple[np.ndarray, bool]], n: int = 18, size: int = 128
) -> list[np.ndarray]:
    """Rotating MIPs of a downsampled lung-windowed volume with detections marked."""
    from scipy.ndimage import rotate, zoom

    f = size / max(vol_hu.shape)
    small = zoom(np.clip((vol_hu + 1000) / 1400, 0, 1).astype(np.float32), f, order=1)
    marker = np.zeros_like(small)
    for c, pos in pts:
        cz, cy, cx = np.clip(np.round(np.asarray(c) * f).astype(int), 0, np.array(small.shape) - 1)
        marker[max(cz - 2, 0) : cz + 3, max(cy - 2, 0) : cy + 3, max(cx - 2, 0) : cx + 3] = (
            2 if pos else 1
        )
    frames = []
    for k in range(n):
        ang = 360 * k / n
        r = rotate(small, ang, axes=(1, 2), reshape=False, order=1).max(1)
        mk = rotate(marker, ang, axes=(1, 2), reshape=False, order=0).max(1)
        rgb = np.stack([r * 255] * 3, -1)
        rgb[mk >= 2] = [251, 146, 60]
        rgb[(mk >= 1) & (mk < 2)] = [148, 163, 184]
        frames.append(np.clip(rgb[::-1], 0, 255).astype(np.uint8))  # head up
    return frames


def local_nodule_size(iso: np.ndarray, c: np.ndarray, cube: int = 40) -> dict[str, Any]:
    """Size of the soft-tissue blob at a detection (1 mm grid).

    HU > -500 inside a 40 mm cube. Erode 1..3 voxels until the component at the centre no longer
    reaches the cube border (detaches vessels / pleura), then grow it back by the same amount inside
    the original tissue mask. If it never separates, the size is reported as indeterminate.
    """
    from scipy import ndimage as ndi

    from ml.data import luna as L

    v = L.extract_patch(iso, c, cube)
    tissue = v > -500
    h = cube // 2
    st = ndi.generate_binary_structure(3, 1)
    for k in (1, 2, 3):
        er = ndi.binary_erosion(tissue, st, iterations=k)
        lab, n = ndi.label(er)
        if n == 0:
            break
        win = lab[h - 3 : h + 4, h - 3 : h + 4, h - 3 : h + 4]
        ids = [i for i in np.unique(win) if i]
        if not ids:
            continue
        comp = lab == ids[np.argmax([np.sum(win == i) for i in ids])]
        if comp[[0, -1]].any() or comp[:, [0, -1]].any() or comp[:, :, [0, -1]].any():
            continue
        grown = ndi.binary_dilation(comp, st, iterations=k) & tissue
        vol = float(grown.sum())
        d_eq = (6 * vol / np.pi) ** (1 / 3)
        if (
            d_eq > 30
        ):  # a nodule is <= 30 mm by definition: this is wall / mediastinum, not the lesion
            break
        return {"equivalent_diameter_mm": round(d_eq, 1), "volume_mm3": round(vol, 0), "note": None}
    return {
        "equivalent_diameter_mm": None,
        "volume_mm3": None,
        "note": "not separable from wall / vessel - size indeterminate",
    }


def analyze_chest_scan(
    vol: np.ndarray, spacing_mm: tuple[float, float, float] | None
) -> dict[str, Any]:
    from ml.data import luna as L

    h = registry.get("lung_nodule_det")
    out: dict[str, Any] = {"findings": [], "detections": [], "measurements": {}}
    if h is None:
        out["error"] = "whole-scan nodule detector not trained"
        return out
    sp = np.asarray(spacing_mm if spacing_mm else (1.0, 1.0, 1.0), float)
    iso = L.resample_iso(_hu(vol).astype(np.int16), sp)
    mask = L.lung_mask(iso)
    cands = L.candidates(iso, mask)
    probs = np.zeros(0)
    if len(cands):
        X = np.stack([L.norm_patch(L.extract_patch(iso, c)) for c in cands])[:, None]
        probs = np.concatenate(
            [_sigmoid(h.run(X[i : i + 64])[0][:, 0]) for i in range(0, len(X), 64)]
        )
    thr = float(h.metrics.get("operating_threshold", 0.5) or 0.5)
    order = np.argsort(-probs)[:5]
    dets = []
    for rank, i in enumerate(order):
        c = cands[i]
        p = float(probs[i])
        crop = L.extract_patch(iso, c, 64)
        cv = analyze_volume(crop, (1.0, 1.0, 1.0))
        vm = local_nodule_size(iso, c)
        mal = cv["findings"][0] if cv.get("findings") else None
        dets.append(
            {
                "rank": rank + 1,
                "centre_mm_zyx": [int(v) for v in c],
                "probability": round(p, 4),
                "positive": p >= thr,
                "diameter_mm": vm.get("equivalent_diameter_mm"),
                "volume_mm3": vm.get("volume_mm3"),
                "malignancy_probability": mal["probability"] if mal else None,
                "malignancy_threshold": mal["threshold"] if mal else None,
                "size_note": vm.get("note"),
                "_crop": cv["volume"],
                "_mask": cv["mask"],
            }
        )
    pmax = float(probs.max()) if len(probs) else 0.0
    n_pos = int(sum(d["positive"] for d in dets))
    top = dets[0] if dets else None
    findings = [
        {
            "label": "Lung nodule (whole-scan detection)",
            "key": "lung_nodule_detected",
            "probability": round(pmax, 4),
            "threshold": round(thr, 4),
            "positive": pmax >= thr,
            "certainty": certainty(pmax, thr),
            "uncertainty": 0.0,
            "ratio_to_threshold": round(pmax / thr, 3) if thr else 0.0,
            "kind": "detection",
            "model": "lung_nodule_det",
            "n_detections": n_pos,
            "diameter_mm": top["diameter_mm"] if top else None,
            "zone": (
                f"z {top['centre_mm_zyx'][0]} mm, y {top['centre_mm_zyx'][1]} mm, x {top['centre_mm_zyx'][2]} mm"
                if top
                else None
            ),
        }
    ]
    if top and top["positive"] and top["malignancy_probability"] is not None:
        pm, tm = float(top["malignancy_probability"]), float(top["malignancy_threshold"] or 0.5)
        findings.append(
            {
                "label": "Malignancy-suspicious lung nodule",
                "key": "malignant_nodule",
                "probability": round(pm, 4),
                "threshold": round(tm, 4),
                "positive": pm >= tm,
                "certainty": certainty(pm, tm),
                "uncertainty": 0.0,
                "ratio_to_threshold": round(pm / tm, 3),
                "kind": "classification",
                "model": "nodule3d",
                "volume_mm3": top["volume_mm3"],
            }
        )
    lung_ml = float(mask.sum()) / 1000.0
    out.update(
        findings=findings,
        detections=[{k: v for k, v in d.items() if not k.startswith("_")} for d in dets],
        measurements={
            "candidates_scored": int(len(cands)),
            "detections_above_threshold": n_pos,
            "operating_point": h.metrics.get("operating_point"),
            "lung_volume_ml": round(lung_ml, 0),
            "resampled_shape": list(iso.shape),
            "spacing_mm": [float(v) for v in sp],
            "spacing_assumed": spacing_mm is None,
            "top_nodule": {k: v for k, v in top.items() if not k.startswith("_")} if top else None,
        },
        iso=iso,
        top_crop=(top["_crop"], top["_mask"]) if top else None,
    )
    z = int(top["centre_mm_zyx"][0]) if top else iso.shape[0] // 2
    marks = [
        (
            d["centre_mm_zyx"][1],
            d["centre_mm_zyx"][2],
            (d["diameter_mm"] or 8) / 2 + 4,
            d["positive"],
        )
        for d in dets
        if abs(d["centre_mm_zyx"][0] - z) <= 6
    ]
    out["axial"] = _render_axial(iso, z, marks)
    # lung MIP: only voxels inside the (dilated) lung mask, so vessels and nodules show, not the body
    from scipy import ndimage as ndi

    lung_only = np.where(ndi.binary_dilation(mask, iterations=3), iso, -1000).astype(np.int16)
    out["mip_frames"] = _mip_frames(
        lung_only, [(np.array(d["centre_mm_zyx"]), d["positive"]) for d in dets]
    )
    out["mpr"] = _mpr(iso, top["centre_mm_zyx"] if top else None)
    return out


def _mpr(iso: np.ndarray, c: list[int] | None) -> dict[str, np.ndarray]:
    zc, yc, xc = c if c else [s // 2 for s in iso.shape]
    w = np.clip((iso + 1350) / 1500, 0, 1) * 255
    to = lambda a: np.stack([a.astype(np.uint8)] * 3, -1)  # noqa: E731
    return {"axial": to(w[zc]), "coronal": to(w[:, yc, :][::-1]), "sagittal": to(w[:, :, xc][::-1])}


# --------------------------------------------------------------------------------------- PET-CT
def analyze_petct(
    ct: np.ndarray, pet: np.ndarray, spacing_mm: tuple[float, float, float] | None
) -> dict[str, Any]:
    from ml.petct_fusion import PAD, norm_ct, norm_pet, stack_input

    h = registry.get("petct_seg")
    out: dict[str, Any] = {"findings": [], "measurements": {}}
    if ct.shape != pet.shape:
        out["error"] = f"CT {ct.shape} and PET {pet.shape} must be co-registered on the same grid"
        return out
    if h is None:
        out["error"] = "PET-CT segmenter not trained"
        return out
    ct = _hu(ct)
    pet = np.clip(np.asarray(pet, np.float32), 0, None)
    hh, ww = ct.shape[1:]
    if max(hh, ww) > PAD:
        f = PAD / max(hh, ww)
        ct = np.stack(
            [
                cv2.resize(s, (round(ww * f), round(hh * f)), interpolation=cv2.INTER_AREA)
                for s in ct
            ]
        )
        pet = np.stack(
            [
                cv2.resize(s, (round(ww * f), round(hh * f)), interpolation=cv2.INTER_AREA)
                for s in pet
            ]
        )
        sp_in = list(spacing_mm) if spacing_mm else [5.0, 5.0, 5.0]
        spacing_mm = (sp_in[0], sp_in[1] / f, sp_in[2] / f)
        hh, ww = ct.shape[1:]
    X = stack_input(ct, pet, "petct")
    prob = np.concatenate([_sigmoid(h.run(X[i : i + 32])[0][:, 0]) for i in range(0, len(X), 32)])[
        :, :hh, :ww
    ]
    mask = prob > 0.5
    from scipy import ndimage as ndi

    lab, n = ndi.label(mask)
    if n:
        sizes = ndi.sum(mask, lab, range(1, n + 1))
        mask = lab == (int(np.argmax(sizes)) + 1)  # primary lesion = largest component
    sp = np.asarray(spacing_mm if spacing_mm else (5.0, 5.0, 5.0), float)
    vox_ml = float(np.prod(sp)) / 1000.0
    v = pet[mask]
    mtv = float(mask.sum()) * vox_ml
    meas = {
        "suv_max": round(float(v.max()), 2) if v.size else 0.0,
        "suv_mean": round(float(v.mean()), 2) if v.size else 0.0,
        "mtv_ml": round(mtv, 1),
        "tlg": round(float(v.mean()) * mtv, 1) if v.size else 0.0,
        "suv_max_whole_body": round(float(pet.max()), 2),
        "spacing_mm": [float(s) for s in sp],
        "spacing_assumed": spacing_mm is None,
        "method": "our 2.5-D PET+CT U-Net mask (largest component); SUV values as supplied",
    }
    pmax = float(prob.max()) if prob.size else 0.0
    positive = bool(mask.sum() > 0 and mtv >= 1.0)
    out["findings"] = [
        {
            "label": "FDG-avid tumour (PET-CT)",
            "key": "fdg_avid_tumour",
            "probability": round(pmax, 4),
            "threshold": 0.5,
            "positive": positive,
            "certainty": certainty(pmax, 0.5),
            "uncertainty": 0.0,
            "ratio_to_threshold": round(pmax / 0.5, 3),
            "kind": "segmentation",
            "model": "petct_seg",
            "suv_max": meas["suv_max"],
            "mtv_ml": meas["mtv_ml"],
        }
    ]
    out["measurements"] = meas
    z = int(np.argmax(mask.sum((1, 2)))) if mask.any() else int(np.argmax(pet.max((1, 2))))
    out["fused_axial"] = fuse_slice(norm_ct(ct[z]), norm_pet(pet[z]), mask[z])
    out["pet_mip"] = _pet_mip(norm_pet(pet), mask)
    out["mpr"] = {  # fused first: the viewer opens on the first plane
        "fused": out["fused_axial"],
        "ct": np.stack([(norm_ct(ct[z]) * 255).astype(np.uint8)] * 3, -1),
        "pet": _hot(norm_pet(pet[z])),
    }
    out["slice"] = z
    out["mask"] = mask
    return out


def _hot(x01: np.ndarray) -> np.ndarray:
    u8 = (np.clip(x01, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_HOT), cv2.COLOR_BGR2RGB)


def fuse_slice(
    ct01: np.ndarray, pet01: np.ndarray, mask: np.ndarray | None = None, alpha: float = 0.6
) -> np.ndarray:
    """CT grayscale + PET hot colour map (alpha grows with uptake) + tumour contour (cyan)."""
    base = np.stack([ct01 * 255] * 3, -1).astype(np.float32)
    hot = _hot(pet01).astype(np.float32)
    a = (np.clip(pet01, 0, 1) ** 0.7 * alpha)[..., None]
    rgb = np.clip(base * (1 - a) + hot * a, 0, 255).astype(np.uint8)
    if mask is not None and mask.any():
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        rgb = np.ascontiguousarray(rgb)
        cv2.drawContours(rgb, cnts, -1, (34, 211, 238), 1)
    return rgb


def _pet_mip(pet01: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Coronal PET MIP (the classic whole-body view) with the lesion tinted."""
    mip = pet01.max(1)[::-1]
    rgb = 255 - np.stack([np.clip(mip, 0, 1) * 255] * 3, -1)  # inverted, as read clinically
    mk = mask.max(1)[::-1]
    rgb[mk] = 0.5 * rgb[mk] + 0.5 * np.array([34, 211, 238])
    return np.clip(rgb, 0, 255).astype(np.uint8)
