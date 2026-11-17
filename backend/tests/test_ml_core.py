"""Data splits (no patient leakage), transforms, model output shapes, metrics, calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ml.data.splits import group_split, leakage, stratified_split
from ml.data.transforms import (
    Augment,
    add_low_dose_noise,
    normalize_gray,
    preprocess,
    preprocess_rgb,
)
from ml.metrics import dice_iou, fit_temperature, map50, reliability, sigmoid, youden_threshold

SPLITS = Path(__file__).resolve().parents[2] / "ml" / "data" / "splits"


# ----------------------------------------------------------------------------- splits
@pytest.mark.parametrize(
    "name,group",
    [
        ("nih_full", "patient_id"),
        ("nih_sample", "patient_id"),
        ("lgg", "patient_id"),
        ("iu_xray", "uid"),
        ("rsna", "patient_id"),
    ],
)
def test_no_patient_leakage(name: str, group: str) -> None:
    p = SPLITS / f"{name}.csv"
    if not p.exists():
        pytest.skip(f"{name} split not generated")
    df = pd.read_csv(p)
    assert set(df["split"]) <= {"train", "val", "test"}
    assert all(v == 0 for v in leakage(df, group).values()), leakage(df, group)


def test_group_split_keeps_groups_together() -> None:
    groups = np.repeat(np.arange(200), 5)
    s = group_split(groups, val_frac=0.2, seed=1, test_frac=0.1)
    df = pd.DataFrame({"g": groups, "split": s})
    assert df.groupby("g")["split"].nunique().max() == 1
    frac = (df.groupby("g")["split"].first() == "val").mean()
    assert 0.15 < frac < 0.25


def test_stratified_split_preserves_classes() -> None:
    y = np.array([0] * 100 + [1] * 20)
    s = stratified_split(y, (0.7, 0.15, 0.15), 0)
    for split in ("train", "val", "test"):
        assert (y[s == split] == 1).sum() > 0


# ----------------------------------------------------------------------------- transforms
def test_preprocess_shapes_and_normalisation() -> None:
    img = (np.random.rand(300, 200) * 255).astype(np.uint8)
    x = preprocess(img, 128)
    assert x.shape == (1, 3, 128, 128) and x.dtype == np.float32
    assert preprocess(img, 64, channels=1).shape == (1, 1, 64, 64)
    rgb = (np.random.rand(50, 60, 3) * 255).astype(np.uint8)
    assert preprocess_rgb(rgb, 96).shape == (1, 3, 96, 96)
    assert normalize_gray(np.full((8, 8), 0.5, np.float32)).shape == (3, 8, 8)


def test_augment_keeps_mask_aligned() -> None:
    img = np.zeros((100, 100), np.uint8)
    img[20:60, 30:70] = 255
    mask = (img > 0).astype(np.uint8)
    a = Augment(64, rotate=0, scale=(1.0, 1.0), brightness=0, contrast=0, clahe_p=0)
    x, m = a(img, mask)
    assert x.shape == (64, 64) and m is not None and m.shape == (64, 64)
    assert x.min() >= 0 and x.max() <= 1
    assert np.mean((x > 0.5) == (m > 0)) > 0.97


def test_low_dose_noise_in_range() -> None:
    x = torch.rand(2, 1, 32, 32)
    n = add_low_dose_noise(x, photons=50, sigma=0.02)
    assert n.shape == x.shape and n.min() >= 0 and n.max() <= 1
    assert (n - x).abs().mean() > 0.01


# ----------------------------------------------------------------------------- models
def test_model_output_shapes() -> None:
    from ml.models.autoencoder import DnCNN
    from ml.models.classifier import ClassifierNet, SmallCNN
    from ml.models.report_gen import ReportGenerator
    from ml.models.unet import UNet

    x = torch.randn(2, 3, 64, 64)
    clf = ClassifierNet("resnet18", 14, pretrained=False).eval()
    assert clf(x).shape == (2, 14)
    assert clf.forward_features(x).shape == (2, clf.num_features)
    assert SmallCNN(5)(x).shape == (2, 5)
    g = torch.randn(2, 1, 64, 64)
    assert UNet(1, 1, base=8, depth=3)(g).shape == (2, 1, 64, 64)
    assert UNet(1, 1, base=8, depth=3, attention=True)(g).shape == (2, 1, 64, 64)
    assert DnCNN(depth=4, width=8)(torch.rand(2, 1, 32, 32)).shape == (2, 1, 32, 32)
    rg = ReportGenerator(100, d_model=32, layers=1, heads=2, ffn=64, max_len=20, encoder="resnet18")
    toks = torch.randint(0, 100, (2, 10))
    assert rg(x, toks).shape == (2, 10, 100)
    out = rg.generate(x[:1], bos=1, eos=2, max_new=5, beam=2)
    assert isinstance(out[0], list) and len(out[0]) <= 5


# ----------------------------------------------------------------------------- metrics + calibration
def test_temperature_scaling_recovers_temperature() -> None:
    rng = np.random.default_rng(0)
    true_logits = rng.normal(0, 2, 20000)
    y = (rng.random(20000) < sigmoid(true_logits)).astype(float)
    t = fit_temperature(true_logits * 3.0, y)  # overconfident by 3x
    assert 2.6 < t < 3.4


def test_reliability_perfectly_calibrated_has_low_ece() -> None:
    rng = np.random.default_rng(1)
    p = rng.random(50000)
    y = (rng.random(50000) < p).astype(float)
    assert reliability(y, p)["ece"] < 0.02


def test_youden_threshold_separates() -> None:
    y = np.array([0] * 50 + [1] * 50)
    p = np.r_[np.linspace(0, 0.4, 50), np.linspace(0.6, 1, 50)]
    t = youden_threshold(y, p)
    assert 0.4 <= t <= 0.6


def test_dice_and_map() -> None:
    a = np.zeros((1, 10, 10), bool)
    a[0, 2:6, 2:6] = True
    assert dice_iou(a, a) == pytest.approx((1.0, 1.0))
    gt = [np.array([[0, 0, 10, 10]], float)]
    perfect = [{"boxes": np.array([[0, 0, 10, 10]], float), "scores": np.array([0.9])}]
    assert map50(perfect, gt)["map50"] == pytest.approx(1.0)
    miss = [{"boxes": np.array([[50, 50, 60, 60]], float), "scores": np.array([0.9])}]
    assert map50(miss, gt)["map50"] == 0.0
