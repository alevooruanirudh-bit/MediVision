"""Pre-processing + augmentation shared by training (ml/) and inference (backend/)."""

from __future__ import annotations

import math
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_gray_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2GRAY)
    if img.dtype != np.uint8:
        lo, hi = float(img.min()), float(img.max())
        img = ((img - lo) / max(hi - lo, 1e-6) * 255).astype(np.uint8)
    return img


def clahe(img: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(to_gray_uint8(img))


def normalize_gray(img01: np.ndarray, channels: int = 3) -> np.ndarray:
    """(H,W) in [0,1] -> (C,H,W) float32, ImageNet-normalised when C == 3."""
    if channels == 1:
        return ((img01 - 0.5) / 0.25)[None].astype(np.float32)
    x = np.repeat(img01[None], 3, axis=0)
    return ((x - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]).astype(np.float32)


def preprocess(img: np.ndarray, size: int, channels: int = 3) -> np.ndarray:
    """Inference pre-processing: any image -> (1,C,size,size) float32. Mirrors eval transforms."""
    g = to_gray_uint8(img)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    return normalize_gray(g.astype(np.float32) / 255.0, channels)[None]


def preprocess_rgb(img: np.ndarray, size: int) -> np.ndarray:
    """Colour-preserving pre-processing (modality router needs colour cues)."""
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    img = cv2.resize(img[..., :3], (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    x = (img / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return x.transpose(2, 0, 1)[None].astype(np.float32)


class Augment:
    """Lightweight numpy/cv2 augmentations for grayscale medical images (+ optional mask).

    No horizontal flips by default for chest X-rays (laterality matters: heart is on the left).
    """

    def __init__(
        self,
        size: int,
        scale: tuple[float, float] = (0.8, 1.0),
        rotate: float = 7.0,
        brightness: float = 0.15,
        contrast: float = 0.15,
        clahe_p: float = 0.3,
        hflip: bool = False,
        vflip: bool = False,
        train: bool = True,
    ) -> None:
        self.size, self.scale, self.rotate = size, scale, rotate
        self.brightness, self.contrast, self.clahe_p = brightness, contrast, clahe_p
        self.hflip, self.vflip, self.train = hflip, vflip, train

    def __call__(
        self, img: np.ndarray, mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        img = to_gray_uint8(img)
        if not self.train:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
            if mask is not None:
                mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
            return img.astype(np.float32) / 255.0, mask
        h, w = img.shape
        s = random.uniform(*self.scale)
        ch, cw = int(h * math.sqrt(s)), int(w * math.sqrt(s))
        y0, x0 = random.randint(0, h - ch), random.randint(0, w - cw)
        img = img[y0 : y0 + ch, x0 : x0 + cw]
        if mask is not None:
            mask = mask[y0 : y0 + ch, x0 : x0 + cw]
        ang = random.uniform(-self.rotate, self.rotate)
        m = cv2.getRotationMatrix2D((cw / 2, ch / 2), ang, 1.0)
        img = cv2.warpAffine(img, m, (cw, ch), borderMode=cv2.BORDER_REFLECT_101)
        if mask is not None:
            mask = cv2.warpAffine(mask, m, (cw, ch), flags=cv2.INTER_NEAREST)
        if random.random() < self.clahe_p:
            img = clahe(img)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if mask is not None:
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        if self.hflip and random.random() < 0.5:
            img = img[:, ::-1]
            mask = mask[:, ::-1] if mask is not None else None
        if self.vflip and random.random() < 0.5:
            img = img[::-1]
            mask = mask[::-1] if mask is not None else None
        x = img.astype(np.float32) / 255.0
        c = 1 + random.uniform(-self.contrast, self.contrast)
        b = random.uniform(-self.brightness, self.brightness)
        x = np.clip((x - 0.5) * c + 0.5 + b, 0, 1)
        return np.ascontiguousarray(x), (np.ascontiguousarray(mask) if mask is not None else None)


def add_low_dose_noise(
    x: torch.Tensor, photons: float | None = None, sigma: float | None = None
) -> torch.Tensor:
    """Simulated low-dose X-ray: Poisson (quantum) + Gaussian (electronic) noise on [0,1] images."""
    photons = photons if photons is not None else random.uniform(30, 300)
    sigma = sigma if sigma is not None else random.uniform(0.0, 0.04)
    noisy = torch.poisson(x.clamp(0, 1) * photons) / photons
    noisy = noisy + torch.randn_like(x) * sigma
    return noisy.clamp(0, 1)


def resize_tensor(x: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
