"""LUNA16 whole-chest CT helpers: MetaImage reader, 1 mm resampling, lung mask, candidate generator.

Shared by training (ml/lung_detect.py) and the backend whole-scan pipeline, so the candidates the
classifier sees at inference are produced exactly as in training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from ml.common import DATA_DIR

LUNA = DATA_DIR / "raw" / "luna16"
PATCH = 32  # mm (= voxels at 1 mm)


def read_mhd(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (volume zyx int16 HU, spacing zyx mm, origin zyx mm). LUNA scans are axis-aligned."""
    hdr: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            hdr[k.strip()] = v.strip()
    dims = [int(v) for v in hdr["DimSize"].split()]  # x y z
    spacing = np.array([float(v) for v in hdr["ElementSpacing"].split()])[::-1]
    origin = np.array([float(v) for v in hdr["Offset"].split()])[::-1]
    raw = Path(path).with_name(hdr["ElementDataFile"])
    dtype = {"MET_SHORT": np.int16, "MET_USHORT": np.uint16, "MET_FLOAT": np.float32}[
        hdr["ElementType"]
    ]
    vol = np.fromfile(raw, dtype=dtype).reshape(dims[::-1])
    return vol.astype(np.int16), spacing, origin


def resample_iso(vol: np.ndarray, spacing: np.ndarray, new: float = 1.0) -> np.ndarray:
    zoom = np.asarray(spacing, float) / new
    return ndi.zoom(vol.astype(np.float32), zoom, order=1).astype(np.int16)


def lung_mask(vol_hu: np.ndarray, work_mm: float = 2.0) -> np.ndarray:
    """Lung field (both lungs) at the input grid, computed at `work_mm` for speed (vol is 1 mm).

    Air (< -320 HU) not connected to the in-plane border, two largest components, then a closing so
    juxtapleural nodules and vessels stay inside.
    """
    f = int(round(work_mm))
    small = vol_hu[::f, ::f, ::f]
    air = small < -320
    lab, n = ndi.label(air)
    if n == 0:
        return np.zeros(vol_hu.shape, bool)
    # only the in-plane edges count as "outside": the trachea legitimately leaves through the top slice
    border = set(
        np.unique(
            np.concatenate(
                [lab[:, 0].ravel(), lab[:, -1].ravel(), lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]
            )
        )
    )
    sizes = ndi.sum(air, lab, range(1, n + 1))
    order = [i + 1 for i in np.argsort(sizes)[::-1] if (i + 1) not in border]
    keep = np.zeros_like(air)
    min_size = 0.01 * air.size / 8
    for lbl in order[:2]:
        if sizes[lbl - 1] >= min_size:
            keep |= lab == lbl
    keep = ndi.binary_closing(keep, ndi.generate_binary_structure(3, 1), iterations=5)
    keep = np.stack([ndi.binary_fill_holes(s) for s in keep])
    up = np.repeat(np.repeat(np.repeat(keep, f, 0), f, 1), f, 2)
    out = np.zeros(vol_hu.shape, bool)
    sz = tuple(min(a, b) for a, b in zip(up.shape, vol_hu.shape, strict=True))
    out[: sz[0], : sz[1], : sz[2]] = up[: sz[0], : sz[1], : sz[2]]
    return out


def candidates(vol_hu: np.ndarray, mask: np.ndarray, max_n: int = 600) -> np.ndarray:
    """Blob candidates (zyx voxel coords, 1 mm grid) inside the lung mask.

    Multi-scale scale-normalised LoG (sigma 1.5 / 2.5 / 4 / 6 mm ~ 4-20 mm diameters) on the lung-windowed
    volume, local maxima (6 mm apart) ranked by response. Larger masses are still caught by the
    smaller scales at their surface / centre.
    """
    from skimage.feature import peak_local_max

    zz, yy, xx = np.where(mask)
    if len(zz) == 0:
        return np.zeros((0, 3), int)
    lo = np.maximum([zz.min() - 4, yy.min() - 4, xx.min() - 4], 0)
    hi = np.minimum([zz.max() + 5, yy.max() + 5, xx.max() + 5], vol_hu.shape)
    sub = vol_hu[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]].astype(np.float32)
    m = mask[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    x = np.clip((sub + 1000.0) / 1000.0, 0, 1.2)  # [-1000, 200] HU
    resp = np.zeros_like(x)
    for s in (1.5, 2.5, 4.0, 6.0):
        np.maximum(resp, -(s**2) * ndi.gaussian_laplace(x, s), out=resp)
    resp[~m] = 0
    pk = peak_local_max(
        resp, min_distance=6, threshold_abs=0.08, num_peaks=max_n, exclude_border=False
    )
    return pk + lo


def extract_patch(vol: np.ndarray, c: Any, size: int = PATCH, fill: int = -1000) -> np.ndarray:
    """Cube of `size` voxels centred on c (zyx), padded with air."""
    c = np.round(np.asarray(c)).astype(int)
    h = size // 2
    out = np.full((size, size, size), fill, np.int16)
    src_lo = c - h
    src_hi = c + h
    a = np.maximum(src_lo, 0)
    b = np.minimum(src_hi, vol.shape)
    if np.any(b <= a):
        return out
    dst = a - src_lo
    out[
        dst[0] : dst[0] + b[0] - a[0], dst[1] : dst[1] + b[1] - a[1], dst[2] : dst[2] + b[2] - a[2]
    ] = vol[a[0] : b[0], a[1] : b[1], a[2] : b[2]]
    return out


def norm_patch(p: np.ndarray) -> np.ndarray:
    """HU -> [0, 1] over the lung window [-1000, 400]."""
    return np.clip((p.astype(np.float32) + 1000.0) / 1400.0, 0, 1)


def load_scan_iso(uid: str) -> tuple[np.ndarray, np.ndarray]:
    """(1 mm volume, origin zyx) for a LUNA series uid."""
    vol, spacing, origin = read_mhd(LUNA / "scans" / f"{uid}.mhd")
    return resample_iso(vol, spacing), origin


def annotations() -> dict[str, list[tuple[np.ndarray, float]]]:
    """uid -> [(world zyx mm, diameter mm)]."""
    import csv

    out: dict[str, list[tuple[np.ndarray, float]]] = {}
    with (LUNA / "annotations.csv").open() as fh:
        for r in csv.DictReader(fh):
            w = np.array([float(r["coordZ"]), float(r["coordY"]), float(r["coordX"])])
            out.setdefault(r["seriesuid"], []).append((w, float(r["diameter_mm"])))
    return out
