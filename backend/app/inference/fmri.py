"""Task-fMRI activation maps (C7 functional imaging, D-35): renders the GLM t-maps computed by
ml/fmri_glm.py over the subject's mean EPI. Statistics, not a learned model."""

from __future__ import annotations

import base64
import json
from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from app.config import get_settings

UNCORRECTED_T = 3.1  # ~p < 0.001 uncorrected, shown faintly; Bonferroni-significant voxels bright


def _dir() -> Any:
    root = get_settings().artifacts_dir / "fmri_glm"
    for prof in ("full", "starter"):
        if (root / prof / "metrics.json").exists():
            return root / prof
    return None


def _png(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def render(mean: np.ndarray, t: np.ndarray, k: int, thr: float, scale: int = 5) -> np.ndarray:
    """Axial slice k: mean EPI (gray) + t > 3.1 (dim orange) + t > Bonferroni (bright yellow)."""
    m = np.rot90(mean[:, :, k].astype(np.float32))
    tt = np.rot90(t[:, :, k].astype(np.float32))
    lo, hi = np.percentile(m[m > 0], [2, 99]) if (m > 0).any() else (0, 1)
    g = np.clip((m - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.stack([g * 255] * 3, -1)
    weak = (tt > UNCORRECTED_T) & (tt <= thr)
    strong = tt > thr
    rgb[weak] = 0.45 * rgb[weak] + 0.55 * np.array([234, 88, 12])
    rgb[strong] = 0.2 * rgb[strong] + 0.8 * np.array([250, 204, 21])
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return cv2.resize(
        rgb, (rgb.shape[1] * scale, rgb.shape[0] * scale), interpolation=cv2.INTER_NEAREST
    )


@lru_cache(maxsize=1)
def summary(_mtime: float = 0.0) -> dict[str, Any] | None:
    d = _dir()
    if d is None:
        return None
    metrics = json.loads((d / "metrics.json").read_text())
    subjects = {}
    for sub, s in metrics.get("subjects", {}).items():
        f = d / f"maps_{sub}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        inv = np.linalg.inv(z["affine"])
        thr = float(z["t_thr"])
        images = {}
        for name, p in s["peaks"].items():
            key = name.split(" ")[0].lower()
            k = int(round((inv @ np.array([*p["cerebral_mni"], 1.0]))[2]))
            images[name] = _png(render(z["mean"], z[key], k, thr))
        subjects[sub] = {**s, "images": images}
    return {
        "method": metrics.get("method"),
        "dataset": metrics.get("dataset"),
        "caveat": metrics.get("caveat"),
        "somatotopy_pass_all": metrics.get("somatotopy_pass_all"),
        "uncorrected_t": UNCORRECTED_T,
        "subjects": subjects,
    }


def get_summary() -> dict[str, Any] | None:
    d = _dir()
    mt = (d / "metrics.json").stat().st_mtime if d is not None else 0.0
    return summary(mt)
