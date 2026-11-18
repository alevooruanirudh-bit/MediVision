"""Enhancement Lab: our DnCNN denoiser + classical CLAHE / unsharp mask / gamma, simulated low dose,
PSNR / SSIM against the original."""

from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np

from app.inference.registry import registry
from ml.metrics import psnr, ssim


def simulate_low_dose(
    img01: np.ndarray, photons: float = 80, sigma: float = 0.02, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = rng.poisson(np.clip(img01, 0, 1) * photons) / photons + rng.normal(
        0, sigma, img01.shape
    )
    return np.clip(noisy, 0, 1).astype(np.float32)


def denoise(img01: np.ndarray, tile: int = 512) -> np.ndarray:
    h = registry.get("denoiser")
    if h is None:
        return (
            cv2.fastNlMeansDenoising((img01 * 255).astype(np.uint8), None, 10).astype(np.float32)
            / 255
        )
    H, W = img01.shape
    # fully convolutional: run in tiles to bound memory
    out = np.zeros_like(img01)
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            patch = img01[y : y + tile, x : x + tile]
            ph, pw = patch.shape
            pad = np.pad(patch, ((0, (-ph) % 8), (0, (-pw) % 8)), mode="reflect")
            res = h.run(pad[None, None].astype(np.float32))[0][0, 0]
            out[y : y + ph, x : x + pw] = res[:ph, :pw]
    return np.clip(out, 0, 1)


def super_resolve(img01: np.ndarray, max_side: int = 768) -> tuple[np.ndarray, str]:
    """x2 super-resolution with our SR-GAN generator (bicubic fallback). Input is first reduced so
    the output stays <= 2*max_side."""
    h, w = img01.shape
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        img01 = cv2.resize(img01, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    gen = registry.get("sr_gan")
    if gen is None:
        up = cv2.resize(img01, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        return np.clip(up, 0, 1), "bicubic x2 (SR-GAN not trained)"
    out = gen.run(img01[None, None].astype(np.float32))[0][0, 0]
    return np.clip(out, 0, 1), "SR-GAN x2 (ours, experimental)"


def fuse(
    base: np.ndarray, over: np.ndarray, alpha: float = 0.5, register: bool = True
) -> dict[str, Any]:
    """Multimodal-style fusion of two co-located images: rigid translation by phase correlation,
    then a false-colour overlay (base = gray, overlay = magenta/green)."""
    from skimage.registration import phase_cross_correlation

    b = cv2.resize(base, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    o = cv2.resize(over, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    shift = [0.0, 0.0]
    if register:
        sh, err, _ = phase_cross_correlation(b, o, upsample_factor=4)
        shift = [float(sh[0]), float(sh[1])]
        m = np.float32([[1, 0, shift[1]], [0, 1, shift[0]]])
        o = cv2.warpAffine(o, m, (512, 512), borderMode=cv2.BORDER_CONSTANT)
    rgb = np.stack([b, b, b], -1) * (1 - alpha)
    rgb[..., 0] += alpha * o
    rgb[..., 2] += alpha * o
    rgb[..., 1] += alpha * b * 0.0
    ncc = float(np.corrcoef(b.ravel(), o.ravel())[0, 1])
    ok, buf = cv2.imencode(
        ".png", cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    )
    return {
        "image": "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode(),
        "shift_px": shift,
        "correlation_after": round(ncc, 3),
        "method": "rigid phase correlation + false-colour fusion (gray base, magenta overlay)",
    }


def apply_ops(img: np.ndarray, ops: dict[str, Any]) -> dict[str, Any]:
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clean = g.astype(np.float32) / 255.0
    x = clean.copy()
    steps = []
    ld = ops.get("simulate_low_dose")
    if ld:
        x = simulate_low_dose(x, float(ld.get("photons", 80)) if isinstance(ld, dict) else 80)
        steps.append("simulated low dose")
    noisy = x.copy()
    if ops.get("denoise"):
        x = denoise(x)
        steps.append("DnCNN denoiser" if registry.get("denoiser") else "NL-means (fallback)")
    if ops.get("clahe"):
        c = ops["clahe"] if isinstance(ops["clahe"], dict) else {}
        u8 = cv2.createCLAHE(
            clipLimit=float(c.get("clip", 2.0)), tileGridSize=(int(c.get("grid", 8)),) * 2
        ).apply((x * 255).astype(np.uint8))
        x = u8.astype(np.float32) / 255
        steps.append("CLAHE")
    if ops.get("unsharp"):
        amt = float(ops["unsharp"].get("amount", 1.0)) if isinstance(ops["unsharp"], dict) else 1.0
        blur = cv2.GaussianBlur(x, (0, 0), 2.0)
        x = np.clip(x + amt * (x - blur), 0, 1)
        steps.append("unsharp mask")
    if ops.get("gamma"):
        x = np.clip(x, 0, 1) ** (1.0 / float(ops["gamma"]))
        steps.append(f"gamma {ops['gamma']}")
    if ops.get("superres"):
        x, how = super_resolve(x)
        steps.append(how)
        clean = cv2.resize(clean, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_CUBIC)
        noisy = cv2.resize(noisy, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_CUBIC)
    metrics = {"psnr": round(psnr(clean, x), 2), "ssim": round(ssim(clean, x), 4)}
    if ld:
        metrics.update(
            {"psnr_noisy": round(psnr(clean, noisy), 2), "ssim_noisy": round(ssim(clean, noisy), 4)}
        )
    return {
        "image": encode_png(x),
        "before": encode_png(noisy if ld else clean),
        "metrics": metrics,
        "steps": steps,
        "denoiser_model": registry.get("denoiser") is not None,
        "superres_model": registry.get("sr_gan") is not None,
        "size": [int(x.shape[1]), int(x.shape[0])],
    }


def encode_png(x01: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", (np.clip(x01, 0, 1) * 255).astype(np.uint8))
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()
