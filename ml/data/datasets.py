"""Dataset classes for every model. All return float32 tensors; images are (C,H,W)."""

from __future__ import annotations

import contextlib
import json
import os
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ml.common import DATA_DIR, NIH_LABELS, PROCESSED, ROUTER_CLASSES
from ml.data.transforms import Augment, normalize_gray

RAW = DATA_DIR / "raw"


def _medmnist_npz(flag: str, size: int) -> Path:
    suffix = "" if size == 28 else f"_{size}"
    for key in (flag, f"{flag}{size}", "chestmnist64" if flag == "chestmnist" else flag):
        p = RAW / key / f"{flag}{suffix}.npz"
        if p.exists():
            return p
    hits = list(RAW.rglob(f"{flag}{suffix}.npz"))
    if not hits:
        raise FileNotFoundError(f"{flag}{suffix}.npz not found under {RAW}; run make data-starter")
    return hits[0]


def load_medmnist(flag: str, size: int, split: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(_medmnist_npz(flag, size))
    return z[f"{split}_images"], z[f"{split}_labels"]


def _subset(n: int, limit: int | None, seed: int = 0) -> np.ndarray:
    idx = np.arange(n)
    if limit and limit < n:
        idx = np.random.default_rng(seed).choice(n, limit, replace=False)
    return idx


# --------------------------------------------------------------------------- chest classification
class ChestMNISTDataset(Dataset):
    """ChestMNIST+ (NIH ChestX-ray14 images, official patient-wise split) — 14 labels."""

    labels = NIH_LABELS

    def __init__(
        self,
        split: str,
        size: int = 128,
        augment: Augment | None = None,
        limit: int | None = None,
        src_size: int = 128,
    ) -> None:
        x, y = load_medmnist("chestmnist", src_size, split)
        idx = _subset(len(x), limit)
        self.x, self.y = x[idx], y[idx].astype(np.float32)
        self.aug = augment or Augment(size, train=False)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, _ = self.aug(self.x[i])
        return torch.from_numpy(normalize_gray(img)), torch.from_numpy(self.y[i])


class NIHDataset(Dataset):
    """NIH ChestX-ray14 with metadata (age/sex/view). Used for the full profile (Kaggle, PNG files)
    and for the local NIH sample (cached .npy) used in fairness evaluation."""

    labels = NIH_LABELS

    def __init__(
        self,
        split: str,
        size: int = 224,
        augment: Augment | None = None,
        source: str = "sample",
        image_root: str | None = None,
        limit: int | None = None,
    ) -> None:
        self.aug = augment or Augment(size, train=False)
        self.source = source
        if source == "sample":
            meta = pd.read_csv(PROCESSED / "nih_sample" / "metadata.csv")
            self.cache = np.load(PROCESSED / "nih_sample" / "images_256.npy", mmap_mode="r")
            meta["row"] = np.arange(len(meta))
        else:
            meta = pd.read_csv(PROCESSED / "nih" / "metadata.csv")
            split_df = pd.read_csv(Path(__file__).parent / "splits" / "nih_full.csv")
            meta = meta.drop(columns=["split"], errors="ignore").merge(split_df[["image", "split"]])
            root = Path(image_root or os.environ.get("NIH_IMAGE_ROOT", "/kaggle/input/data"))
            self.paths = {p.name: p for p in root.rglob("*.png")}
        meta = meta[meta["split"] == split] if split != "all" else meta
        idx = _subset(len(meta), limit)
        self.meta = meta.iloc[idx].reset_index(drop=True)
        self.y = self.meta[NIH_LABELS].to_numpy(np.float32)

    def __len__(self) -> int:
        return len(self.meta)

    def image(self, i: int) -> np.ndarray:
        r = self.meta.iloc[i]
        if self.source == "sample":
            return np.asarray(self.cache[int(r["row"])])
        return cv2.imread(str(self.paths[r["image"]]), cv2.IMREAD_GRAYSCALE)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, _ = self.aug(self.image(i))
        return torch.from_numpy(normalize_gray(img)), torch.from_numpy(self.y[i])


# --------------------------------------------------------------------------- segmentation
class SegDataset(Dataset):
    """Generic image/mask dataset from a processed npz (lung or LGG)."""

    def __init__(
        self,
        key: str,
        split: str,
        size: int,
        augment: Augment | None = None,
        channels: int = 1,
        limit: int | None = None,
    ) -> None:
        d = PROCESSED / key
        npz = next(d.glob("*.npz"))
        z = np.load(npz)
        meta = pd.read_csv(d / "metadata.csv")
        keep = np.where(meta["split"].to_numpy() == split)[0]
        keep = keep[_subset(len(keep), limit)]
        self.x, self.m = z["images"][keep], z["masks"][keep]
        self.meta = meta.iloc[keep].reset_index(drop=True)
        self.aug = augment or Augment(size, train=False)
        self.channels = channels

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, m = self.aug(self.x[i], self.m[i])
        assert m is not None
        return torch.from_numpy(normalize_gray(img, self.channels)), torch.from_numpy(
            m.astype(np.float32)[None]
        )


# --------------------------------------------------------------------------- fracture
def _frac_records(split: str, key: str = "fracatlas") -> list[dict[str, Any]]:
    recs = json.loads((PROCESSED / key / "annotations.json").read_text())
    return [r for r in recs if r["split"] == split]


class FractureClsDataset(Dataset):
    def __init__(
        self, split: str, size: int = 224, augment: Augment | None = None, limit: int | None = None
    ) -> None:
        recs = _frac_records(split)
        self.recs = [recs[i] for i in _subset(len(recs), limit)]
        self.aug = augment or Augment(size, train=False)
        self.y = np.array([r["fractured"] for r in self.recs], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.recs)

    def image(self, i: int) -> np.ndarray:
        p = PROCESSED / "fracatlas" / "images" / self.recs[i]["image"]
        return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, _ = self.aug(self.image(i))
        return torch.from_numpy(normalize_gray(img)), torch.tensor([self.y[i]])


class DetDataset(Dataset):
    """Detection targets in torchvision format (FracAtlas or RSNA). Letterboxed to `size`."""

    def __init__(
        self,
        split: str,
        size: int = 512,
        train: bool = False,
        neg_ratio: float = 0.3,
        limit: int | None = None,
        key: str = "fracatlas",
    ) -> None:
        self.key = key
        recs = _frac_records(split, key)
        pos = [r for r in recs if r["boxes"]]
        neg = [r for r in recs if not r["boxes"]]
        random.Random(0).shuffle(neg)
        n_neg = int(len(pos) * neg_ratio) if train else len(neg)
        self.recs = pos + neg[:n_neg]
        if limit:
            self.recs = self.recs[:limit]
        self.size, self.train = size, train

    def __len__(self) -> int:
        return len(self.recs)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        r = self.recs[i]
        img = cv2.imread(str(PROCESSED / self.key / "images" / r["image"]), cv2.IMREAD_GRAYSCALE)
        img, boxes = letterbox(
            img, np.array(r["boxes"], dtype=np.float32).reshape(-1, 4), self.size
        )
        if self.train and random.random() < 0.5:
            img = img[:, ::-1].copy()
            if len(boxes):
                boxes[:, [0, 2]] = self.size - boxes[:, [2, 0]]
        if self.train:
            img = np.clip(img.astype(np.float32) * random.uniform(0.85, 1.15), 0, 255)
        x = torch.from_numpy(np.repeat(img[None].astype(np.float32) / 255.0, 3, 0))
        target = {
            "boxes": torch.from_numpy(boxes).float().reshape(-1, 4),
            "labels": torch.ones(len(boxes), dtype=torch.int64),
        }
        return x, target


FractureDetDataset = DetDataset


def letterbox(img: np.ndarray, boxes: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    canvas = np.zeros((size, size), dtype=np.uint8)
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return canvas, boxes * s


# --------------------------------------------------------------------------- brain MRI
class BrainMRIDataset(Dataset):
    def __init__(
        self, split: str, size: int = 128, augment: Augment | None = None, limit: int | None = None
    ) -> None:
        d = PROCESSED / "brain_mri"
        meta = pd.read_csv(d / "metadata.csv")
        cache = np.load(next(d.glob("images_*.npy")), mmap_mode="r")
        keep = np.where(meta["split"].to_numpy() == split)[0]
        keep = keep[_subset(len(keep), limit)]
        self.x = np.asarray(cache[keep])
        self.y = meta["label"].to_numpy()[keep].astype(np.int64)
        self.aug = augment or Augment(size, train=False)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, _ = self.aug(self.x[i])
        return torch.from_numpy(normalize_gray(img)), torch.tensor(self.y[i])


# --------------------------------------------------------------------------- router
class RouterDataset(Dataset):
    """8-way modality router: chest_xray / bone_xray / brain_mri / head_ct / breast_us / fundus /
    other_medical (abdominal CT, CT patches, paediatric CXR) / not_medical (CIFAR-10 photos).

    Colour is preserved (natural photos are colourful) but half of the natural images are converted
    to grayscale so the router cannot rely on colour alone.
    """

    def __init__(self, split: str, size: int = 96, per_class: int = 1500, seed: int = 0) -> None:
        rng = np.random.default_rng(seed + {"train": 0, "val": 1, "test": 2}[split])
        self.size, self.train = size, split == "train"
        pools: dict[str, list[np.ndarray]] = {c: [] for c in ROUTER_CLASSES}

        def take(arr: np.ndarray, n: int) -> list[np.ndarray]:
            idx = rng.choice(len(arr), min(n, len(arr)), replace=False)
            return [np.asarray(arr[i]) for i in idx]

        n = per_class if split == "train" else per_class // 5
        x, _ = load_medmnist("chestmnist", 128, split)
        pools["chest_xray"] += take(x, n // 2)
        with contextlib.suppress(Exception):
            pools["chest_xray"] += take(
                SegDataset("lung", split if split != "train" else "train", 128).x, n // 2
            )
        bone = FractureClsDataset(split)
        pools["bone_xray"] += [
            bone.image(i) for i in rng.choice(len(bone), min(n, len(bone)), False)
        ]
        pools["brain_mri"] += take(BrainMRIDataset(split, 128).x, n // 2)
        with contextlib.suppress(Exception):
            pools["brain_mri"] += take(SegDataset("lgg", split, 128).x, n // 2)
        with contextlib.suppress(FileNotFoundError):
            hct = HeadCTDataset(split, 128).x
            # 120 training slices only: oversample (augmentations make the copies differ)
            reps = max(1, (n // 2) // max(len(hct), 1))
            pools["head_ct"] += [np.asarray(im) for im in hct] * reps
        with contextlib.suppress(FileNotFoundError):
            xb, _ = load_medmnist("breastmnist", 128, split)
            reps = max(1, (n // 2) // max(len(xb), 1))
            pools["breast_us"] += take(xb, n) * min(reps, 2)
        with contextlib.suppress(FileNotFoundError):
            xr, _ = load_medmnist("retinamnist", 128, split)
            pools["fundus"] += take(xr, n)
        for flag in ("organamnist", "pneumoniamnist", "nodulemnist3d"):
            try:
                if flag == "nodulemnist3d":
                    xv, _ = load_medmnist(flag, 64, split)
                    xo = xv[:, 32]  # axial mid-slices of chest-CT patches
                else:
                    xo, _ = load_medmnist(flag, 128 if flag != "pneumoniamnist" else 64, split)
                pools["other_medical"] += take(xo, n // 3)
            except FileNotFoundError:
                pass
        import torchvision

        cifar = torchvision.datasets.CIFAR10(
            str(RAW / "cifar10"), train=split == "train", download=False
        )
        pools["not_medical"] += take(cifar.data, n)
        self.items: list[tuple[np.ndarray, int]] = []
        for ci, c in enumerate(ROUTER_CLASSES):
            self.items += [(im, ci) for im in pools[c]]
        self.counts = {c: len(pools[c]) for c in ROUTER_CLASSES}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        from ml.data.transforms import preprocess_rgb

        img, y = self.items[i]
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        if y == ROUTER_CLASSES.index("not_medical") and random.random() < 0.5:
            g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.stack([g] * 3, -1)
        if self.train:
            if random.random() < 0.5:
                img = img[:, ::-1]
            h, w = img.shape[:2]
            s = random.uniform(0.75, 1.0)
            ch, cw = int(h * s), int(w * s)
            y0, x0 = random.randint(0, h - ch), random.randint(0, w - cw)
            img = img[y0 : y0 + ch, x0 : x0 + cw]
            if random.random() < 0.3:  # blur so "blurry = natural" is not a shortcut
                img = cv2.GaussianBlur(np.ascontiguousarray(img), (5, 5), 0)
        x = preprocess_rgb(np.ascontiguousarray(img).astype(np.uint8), self.size)[0]
        return torch.from_numpy(x), torch.tensor(y)


# --------------------------------------------------------------------------- denoiser
class DenoiseDataset(Dataset):
    """Clean chest X-ray crops; noise is synthesised on the fly in the trainer."""

    def __init__(self, split: str, size: int = 128, limit: int | None = 4000) -> None:
        src = 224 if size > 128 else 128
        try:
            x, _ = load_medmnist("chestmnist", src, split)
        except FileNotFoundError:
            x, _ = load_medmnist("chestmnist", 128, split)
        self.x = x[_subset(len(x), limit, seed=3)]
        self.size = size
        self.train = split == "train"

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> torch.Tensor:
        img = self.x[i].astype(np.float32) / 255.0
        if img.shape[0] != self.size:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if self.train and random.random() < 0.5:
            img = img[:, ::-1].copy()
        return torch.from_numpy(img)[None]


# --------------------------------------------------------------------------- IU reports
class IUReportDataset(Dataset):
    def __init__(
        self,
        split: str,
        tokenizer: Any,
        size: int = 128,
        max_len: int = 96,
        augment: Augment | None = None,
    ) -> None:
        d = PROCESSED / "iu_xray"
        meta = pd.read_csv(d / "reports.csv").fillna("")
        cache = np.load(next(d.glob("images_*.npy")), mmap_mode="r")
        keep = np.where(meta["split"].to_numpy() == split)[0]
        self.x = np.asarray(cache[keep])
        self.meta = meta.iloc[keep].reset_index(drop=True)
        self.texts = [report_text(r) for _, r in self.meta.iterrows()]
        self.tok, self.max_len = tokenizer, max_len
        self.aug = augment or Augment(size, train=False)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, _ = self.aug(self.x[i])
        ids = self.tok.encode(self.texts[i]).ids[: self.max_len - 2]
        bos, eos, pad = (self.tok.token_to_id(t) for t in ("<bos>", "<eos>", "<pad>"))
        seq = [bos, *ids, eos] + [pad] * (self.max_len - len(ids) - 2)
        return torch.from_numpy(normalize_gray(img)), torch.tensor(seq, dtype=torch.long)


def report_text(row: Any) -> str:
    f = str(row.get("findings", "")).strip()
    imp = str(row.get("impression", "")).strip()
    if f and imp:
        text = f"findings: {f} impression: {imp}"
    else:
        text = f or imp
    return " ".join(text.lower().replace("xxxx", "").split())


# --------------------------------------------------------------------------- new modalities (D-21)
class HeadCTDataset(Dataset):
    """Head CT hemorrhage (Kitamura, Kaggle): 200 axial slices, binary hemorrhage label."""

    def __init__(
        self, split: str, size: int = 192, augment: Augment | None = None, limit: int | None = None
    ) -> None:
        d = PROCESSED / "head_ct"
        if not (d / "metadata.csv").exists():
            raise FileNotFoundError("head_ct not prepared; run make data-starter")
        meta = pd.read_csv(d / "metadata.csv")
        cache = np.load(next(d.glob("images_*.npy")), mmap_mode="r")
        keep = np.where(meta["split"].to_numpy() == split)[0]
        keep = keep[_subset(len(keep), limit)]
        self.x = np.asarray(cache[keep])
        self.y = meta["label"].to_numpy()[keep].astype(np.float32)
        self.aug = augment or Augment(size, train=False)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, _ = self.aug(self.x[i])
        return torch.from_numpy(normalize_gray(img)), torch.tensor([self.y[i]])


class BreastUSDataset(Dataset):
    """BreastMNIST+ (breast ultrasound, Al-Dhabyani 2020). Positive = malignant (MedMNIST label 0)."""

    def __init__(
        self, split: str, size: int = 128, augment: Augment | None = None, limit: int | None = None
    ) -> None:
        x, y = load_medmnist("breastmnist", 128, split)
        idx = _subset(len(x), limit)
        self.x = x[idx]
        self.y = (1 - y[idx, 0]).astype(np.float32)
        self.aug = augment or Augment(size, train=False)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, _ = self.aug(self.x[i])
        return torch.from_numpy(normalize_gray(img)), torch.tensor([self.y[i]])


class RetinaDataset(Dataset):
    """RetinaMNIST+ colour fundus photographs, 5 diabetic-retinopathy grades (DeepDRiD)."""

    def __init__(self, split: str, size: int = 128, train: bool = False) -> None:
        x, y = load_medmnist("retinamnist", 128, split)
        self.x, self.y = x, y[:, 0].astype(np.int64)
        self.size, self.train = size, train

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        from ml.data.transforms import preprocess_rgb

        img = np.asarray(self.x[i])
        if self.train:
            if random.random() < 0.5:
                img = img[:, ::-1]
            if random.random() < 0.5:
                img = img[::-1]
            k = random.randint(0, 3)
            img = np.rot90(img, k)
            h = img.shape[0]
            s = random.uniform(0.85, 1.0)
            c = int(h * s)
            y0, x0 = random.randint(0, h - c), random.randint(0, h - c)
            img = img[y0 : y0 + c, x0 : x0 + c]
            img = np.clip(
                img.astype(np.float32) * random.uniform(0.85, 1.15) + random.uniform(-15, 15),
                0,
                255,
            ).astype(np.uint8)
        x = preprocess_rgb(np.ascontiguousarray(img), self.size)[0]
        return torch.from_numpy(x), torch.tensor(self.y[i])


class Nodule3DDataset(Dataset):
    """NoduleMNIST3D (LIDC-IDRI chest-CT nodule patches, 64^3). Positive = malignant."""

    def __init__(self, split: str, size: int = 64, train: bool = False) -> None:
        x, y = load_medmnist("nodulemnist3d", 64, split)
        self.x, self.y = x, y[:, 0].astype(np.float32)
        self.size, self.train = size, train

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.x[i].astype(np.float32) / 255.0
        if self.train:
            for ax in range(3):
                if random.random() < 0.5:
                    v = np.flip(v, ax)
            v = np.rot90(v, random.randint(0, 3), axes=(1, 2))
            v = np.clip(v * random.uniform(0.9, 1.1) + random.uniform(-0.05, 0.05), 0, 1)
        v = (v - 0.5) / 0.25
        return torch.from_numpy(np.ascontiguousarray(v)[None].astype(np.float32)), torch.tensor(
            [self.y[i]]
        )
