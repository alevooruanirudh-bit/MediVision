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
    }


def encode_png(x01: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", (np.clip(x01, 0, 1) * 255).astype(np.uint8))
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()
