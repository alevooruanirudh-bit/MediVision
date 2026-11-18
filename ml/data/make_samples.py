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
