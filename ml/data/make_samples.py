"""Pick a few TEST-split images from each dataset into sample_data/ (+ SOURCES.md with licenses).

python -m ml.data.make_samples
"""

from __future__ import annotations

import json
import shutil

import cv2
import numpy as np
import pandas as pd

from ml.common import DATA_DIR, PROCESSED, ROOT

OUT = ROOT / "sample_data"

CHEST = [  # (NIH image, role)
    ("00000246_016.png", "chest_followup_prior"),
    ("00000246_018.png", "chest_followup_current"),
    ("00000468_005.png", "chest_pneumothorax"),
    ("00001075_046.png", "chest_cardiomegaly_effusion"),
    ("00001248_017.png", "chest_effusion"),
    ("00004344_037.png", "chest_cardiomegaly_edema"),
]


def new_modalities() -> list[dict[str, object]]:
    """First test-split example of each new modality (D-21) — not cherry-picked by model output."""
    from ml.data.datasets import load_medmnist

    out: list[dict[str, object]] = []
    d = PROCESSED / "head_ct"
    if (d / "metadata.csv").exists():
        meta = pd.read_csv(d / "metadata.csv")
        cache = np.load(d / "images_256.npy")
        for lab, role in ((1, "headct_hemorrhage"), (0, "headct_normal")):
            i = int(meta[(meta.split == "test") & (meta.label == lab)].index[0])
            img = cv2.resize(cache[i], (512, 512), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(str(OUT / f"{role}.png"), img)
            out.append(
                {
                    "file": f"{role}.png",
                    "source": "Head CT - hemorrhage (Kitamura, Kaggle), test split",
                    "original": meta.loc[i, "id"],
                    "license": "see Kaggle page (research use)",
                    "modality": "head_ct",
                    "ground_truth": "hemorrhage" if lab else "no hemorrhage",
                    "patient_ref": f"HCT-{meta.loc[i, 'id']}",
                }
            )
    try:
        x, y = load_medmnist("breastmnist", 128, "test")
        i = int(np.where(y[:, 0] == 0)[0][0])
        cv2.imwrite(
            str(OUT / "breast_us_malignant.png"),
            cv2.resize(x[i], (384, 384), interpolation=cv2.INTER_CUBIC),
        )
        out.append(
            {
                "file": "breast_us_malignant.png",
                "source": "BreastMNIST+ (BUSI, Al-Dhabyani 2020), test split",
                "original": f"test[{i}]",
                "license": "CC BY 4.0",
                "modality": "breast_us",
                "ground_truth": "malignant",
                "patient_ref": f"BUS-{i}",
            }
        )
    except FileNotFoundError:
        pass
    try:
        x, y = load_medmnist("retinamnist", 128, "test")
        i = int(np.where(y[:, 0] >= 2)[0][0])
        img = cv2.resize(x[i], (384, 384), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(OUT / "fundus_dr.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        out.append(
            {
                "file": "fundus_dr.png",
                "source": "RetinaMNIST+ (DeepDRiD), test split",
                "original": f"test[{i}]",
                "license": "CC BY 4.0",
                "modality": "fundus",
                "ground_truth": f"ICDR grade {int(y[i, 0])}",
                "patient_ref": f"RET-{i}",
            }
        )
    except FileNotFoundError:
        pass
    try:
        x, y = load_medmnist("nodulemnist3d", 64, "test")
        i = int(np.where(y[:, 0] == 1)[0][0])
        np.save(OUT / "ct_nodule_volume.npy", x[i])
        out.append(
            {
                "file": "ct_nodule_volume.npy",
                "source": "NoduleMNIST3D (LIDC-IDRI), test split — 64³ chest-CT patch",
                "original": f"test[{i}]",
                "license": "CC BY 4.0",
                "modality": "ct_volume",
                "ground_truth": "malignant nodule",
                "patient_ref": f"LIDC-{i}",
            }
        )
    except FileNotFoundError:
        pass
    out += round2_modalities()
    return out


def round2_modalities() -> list[dict[str, object]]:
    """Mammography, PET-CT and a whole-chest-CT slab (D-33/34/36). Held-out cases only, picked by a
    fixed rule (first test-split exam / the pre-declared PET-CT hold-out / first test-split scan with
    a nodule) - never by model output."""
    out: list[dict[str, object]] = []
    d = PROCESSED / "mini_ddsm"
    raw = DATA_DIR / "raw" / "mini_ddsm" / "Mini_DDSM_Upload"
    if (d / "metadata.csv").exists() and raw.exists():
        meta = pd.read_csv(d / "metadata.csv")
        folder = {0: "Normal", 2: "Cancer"}
        for lab3, role in ((2, "mammo_cancer"), (0, "mammo_normal")):
            r = (
                meta[(meta.split == "test") & (meta.label3 == lab3)]
                .sort_values(["case", "view"])
                .iloc[0]
            )
            src = next((raw / folder[lab3]).glob(f"*_{r['case']}_*{r['view']}.LJPEG*"), None)
            if src is None:
                continue
            img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(OUT / f"{role}.png"), img)
            out.append(
                {
                    "file": f"{role}.png",
                    "source": "Mini-DDSM (DDSM screening mammography), test split",
                    "original": src.name,
                    "license": "DDSM - free for research, cite (see Kaggle page)",
                    "modality": "mammography",
                    "ground_truth": (
                        "cancer exam (exam-level label)" if lab3 == 2 else "normal exam"
                    ),
                    "age": None if pd.isna(r["age"]) else int(r["age"]),
                    "sex": "F",
                    "patient_ref": f"DDSM-{r['case']}",
                }
            )
    try:
        from ml.petct_fusion import DEMO_HOLDOUT, load_patients

        p = load_patients()[DEMO_HOLDOUT]
        zs = np.where(p["mask"].any((1, 2)))[0]
        z0, z1 = max(int(zs.min()) - 15, 0), min(int(zs.max()) + 16, len(p["mask"]))
        np.savez_compressed(
            OUT / "petct_sarcoma.npz",
            ct=np.round(p["ct"][z0:z1]).astype(np.int16),
            pet=p["pet"][z0:z1].astype(np.float16),
            spacing=np.array([5.0, 5.0, 5.0], np.float32),
        )
        out.append(
            {
                "file": "petct_sarcoma.npz",
                "source": "TCIA Soft-tissue-Sarcoma FDG-PET/CT (4quant 5 mm release); held out of the "
                "deployed model",
                "original": f"{DEMO_HOLDOUT}, slices {z0}-{z1 - 1}",
                "license": "CC BY 3.0",
                "modality": "petct",
                "ground_truth": "soft-tissue sarcoma (expert contour)",
                "patient_ref": f"STS-{DEMO_HOLDOUT}",
            }
        )
    except (FileNotFoundError, KeyError, ImportError):
        pass
    try:
        from ml.common import ARTIFACTS
        from ml.data import luna as L

        m = json.loads((ARTIFACTS / "lung_nodule_det" / "starter" / "metrics.json").read_text())
        ann = L.annotations()
        uid = sorted(s for s, sp in m["series_split"].items() if sp == "test" and s in ann)[0]
        vol, spacing, origin = L.read_mhd(L.LUNA / "scans" / f"{uid}.mhd")
        w, diam = ann[uid][0]
        c = np.round((w - origin) / spacing).astype(int)
        half = np.ceil(np.array([45.0, 120.0, 150.0]) / spacing).astype(
            int
        )  # 90 x 240 x 300 mm slab
        lo, hi = np.maximum(c - half, 0), np.minimum(c + half, vol.shape)
        slab = vol[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
        np.savez_compressed(
            OUT / "chest_ct_scan.npz",
            volume=slab.astype(np.int16),
            spacing=spacing.astype(np.float32),
        )
        out.append(
            {
                "file": "chest_ct_scan.npz",
                "source": "LUNA16 (LIDC-IDRI) test-split scan, 90 mm slab around the first annotated "
                "nodule",
                "original": f"{uid[-12:]} ({slab.shape[0]} slices)",
                "license": "CC BY 4.0",
                "modality": "ct_chest_scan",
                "ground_truth": f"nodule, diameter {diam:.1f} mm",
                "patient_ref": f"LUNA-{uid[-8:]}",
            }
        )
    except (FileNotFoundError, KeyError, IndexError):
        pass
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    manifest = []
    meta = pd.read_csv(PROCESSED / "nih_sample" / "metadata.csv").set_index("image")
    for name, role in CHEST:
        src = PROCESSED / "nih_sample" / "demo_pool" / name
        if not src.exists():
            continue
        shutil.copy(src, OUT / f"{role}.png")
        r = meta.loc[name]
        manifest.append(
            {
                "file": f"{role}.png",
                "source": "NIH ChestX-ray14 (official test-list patient)",
                "original": name,
                "license": "CC0 1.0",
                "modality": "chest",
                "ground_truth": r["labels"],
                "age": int(r["age"]),
                "sex": r["sex"],
                "view": r["view"],
                "patient_ref": f"NIH-{int(r['patient_id'])}",
            }
        )
    recs = json.loads((PROCESSED / "fracatlas" / "annotations.json").read_text())
    pos = [
        r for r in recs if r["split"] == "test" and r["boxes"] and r["region"] in ("hand", "leg")
    ]
    neg = [r for r in recs if r["split"] == "test" and not r["boxes"] and r["region"] == "hand"]
    for r, role in ((pos[0], "bone_fracture"), (neg[0], "bone_no_fracture")):
        shutil.copy(PROCESSED / "fracatlas" / "images" / r["image"], OUT / f"{role}.jpg")
        manifest.append(
            {
                "file": f"{role}.jpg",
                "source": "FracAtlas (test split)",
                "original": r["image"],
                "license": "CC BY 4.0",
                "modality": "bone",
                "ground_truth": "fracture" if r["boxes"] else "no fracture",
                "region": r["region"],
                "patient_ref": f"FRAC-{r['image']}",
            }
        )
    bm = pd.read_csv(PROCESSED / "brain_mri" / "metadata.csv")
    cache = np.load(PROCESSED / "brain_mri" / "images_224.npy", mmap_mode="r")
    for cls in ("glioma", "meningioma", "notumor"):
        i = int(bm[(bm.split == "test") & (bm.cls == cls)].index[3])
        role = f"brain_{cls}"
        cv2.imwrite(str(OUT / f"{role}.png"), np.asarray(cache[i]))
        manifest.append(
            {
                "file": f"{role}.png",
                "source": "Brain Tumor MRI Dataset (Nickparvar), Testing",
                "original": bm.loc[i, "id"],
                "license": "CC BY 4.0",
                "modality": "brain",
                "ground_truth": cls,
                "patient_ref": f"BRAIN-{bm.loc[i, 'id']}",
            }
        )
    import torchvision

    cifar = torchvision.datasets.CIFAR10(str(DATA_DIR / "raw" / "cifar10"), train=False)
    idx = next(i for i, t in enumerate(cifar.targets) if t == 7)  # horse
    img = cv2.resize(cifar.data[idx], (384, 384), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(OUT / "not_medical_photo.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    manifest.append(
        {
            "file": "not_medical_photo.png",
            "source": "CIFAR-10 test set (upscaled)",
            "original": f"test[{idx}] (horse)",
            "license": "CIFAR-10 terms (Krizhevsky 2009)",
            "modality": "none",
            "ground_truth": "not a medical image",
            "patient_ref": "PHOTO-1",
        }
    )
    manifest += new_modalities()
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    lines = [
        "# Sample data sources",
        "",
        "Test-split images only (never used for training). Used by `app/seed.py` and the E2E tests.",
        "",
        "| File | Source | Original | License | Ground truth |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {m['file']} | {m['source']} | {m['original']} | {m['license']} | {m['ground_truth']} |"
        for m in manifest
    ]
    lines += [
        "",
        "Citations: Wang X. et al. CVPR 2017 (NIH ChestX-ray14); Abedeen I. et al. Sci Data 2023 "
        "(FracAtlas); Nickparvar M. Kaggle 2021 (Brain Tumor MRI); Krizhevsky A. 2009 (CIFAR-10).",
    ]
    (OUT / "SOURCES.md").write_text("\n".join(lines) + "\n")
    print(f"[samples] wrote {len(manifest)} files to {OUT}")


if __name__ == "__main__":
    main()
