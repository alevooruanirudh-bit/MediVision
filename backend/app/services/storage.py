"""File storage for de-identified images, overlays and exports (served under /files)."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from app.config import get_settings

MAX_SIDE = 1024


def study_dir(study_id: str) -> Path:
    d = get_settings().storage_dir / "studies" / study_id
    (d / "overlays").mkdir(parents=True, exist_ok=True)
    return d


def rel(path: Path) -> str:
    return str(path.relative_to(get_settings().storage_dir))


def url(relpath: str | None) -> str | None:
    return f"/files/{relpath}" if relpath else None


def abs_path(relpath: str) -> Path:
    return get_settings().storage_dir / relpath


def save_image(study_id: str, pixels: np.ndarray) -> tuple[str, int, int, float]:
    """Save display image (max side 1024). Returns (relpath, w, h, scale vs original)."""
    h, w = pixels.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    if scale < 1:
        pixels = cv2.resize(
            pixels, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
        )
    p = study_dir(study_id) / "image.png"
    out = pixels if pixels.ndim == 2 else cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(p), out)
    return rel(p), pixels.shape[1], pixels.shape[0], scale


def load_image(relpath: str, gray: bool = True) -> np.ndarray:
    p = abs_path(relpath)
    if gray:
        return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_rgba(study_id: str, name: str, rgba: np.ndarray) -> str:
    p = study_dir(study_id) / "overlays" / f"{name}.png"
    cv2.imwrite(str(p), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    return rel(p)


def delete_study_files(study_id: str) -> None:
    shutil.rmtree(get_settings().storage_dir / "studies" / study_id, ignore_errors=True)
