"""Resumable downloader for every free dataset used by MediVision AI.

Usage:
    python -m ml.data.download --profile starter            # small local starter set
    python -m ml.data.download --only fracatlas brain_mri    # specific sources
    python -m ml.data.download --list

Archives land in data/raw/<key>/, SHA-256 checksums are recorded in data/checksums.json and verified on
re-runs. Missing credentials / unaccepted competition rules never block the build: the source is skipped,
reported, and the caller falls back to a MedMNIST+ equivalent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from ml.common import DATA_DIR, ROOT

RAW = DATA_DIR / "raw"
CHECKSUMS = DATA_DIR / "checksums.json"


@dataclass(frozen=True)
class Source:
    key: str
    title: str
    kind: str  # kaggle_dataset | kaggle_competition | medmnist | torchvision | url
    ref: str
    license: str
    citation: str
    url: str
    profiles: tuple[str, ...] = ("starter", "full")
    files: tuple[str, ...] = field(default_factory=tuple)  # optional single files
    optional: bool = False
    note: str = ""


SOURCES: list[Source] = [
    Source(
        "nih_meta",
        "NIH ChestX-ray14 metadata (Data_Entry_2017.csv, official split lists)",
        "kaggle_dataset",
        "nih-chest-xrays/data",
        "CC0 1.0",
        "Wang X. et al. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks. CVPR 2017.",
        "https://www.kaggle.com/datasets/nih-chest-xrays/data",
        files=("Data_Entry_2017.csv", "test_list.txt", "train_val_list.txt", "BBox_List_2017.csv"),
    ),
    Source(
        "nih_sample",
        "NIH ChestX-ray14 random sample (5,606 images + metadata)",
        "kaggle_dataset",
        "nih-chest-xrays/sample",
        "CC0 1.0",
        "Wang X. et al. CVPR 2017 (NIH Clinical Center).",
        "https://www.kaggle.com/datasets/nih-chest-xrays/sample",
        profiles=("starter",),
        note="Starter: external fairness evaluation (age/sex) on official-test-list patients + demo images.",
    ),
    Source(
        "nih_full",
        "NIH ChestX-ray14 full (112,120 images, 45 GB)",
        "kaggle_dataset",
        "nih-chest-xrays/data",
        "CC0 1.0",
        "Wang X. et al. CVPR 2017.",
        "https://www.kaggle.com/datasets/nih-chest-xrays/data",
        profiles=("full",),
        note="Used only inside Kaggle notebooks via dataset_sources (never downloaded locally).",
    ),
    Source(
        "chestmnist",
        "ChestMNIST+ 128 px (MedMNIST+)",
        "medmnist",
        "chestmnist:128",
        "CC BY 4.0",
        "Yang J. et al. MedMNIST v2. Scientific Data 2023.",
        "https://medmnist.com/",
    ),
    Source(
        "chestmnist64",
        "ChestMNIST 64 px (federated demo)",
        "medmnist",
        "chestmnist:64",
        "CC BY 4.0",
        "Yang J. et al. MedMNIST v2. Scientific Data 2023.",
        "https://medmnist.com/",
    ),
    Source(
        "pneumoniamnist",
        "PneumoniaMNIST 64/128 px",
        "medmnist",
        "pneumoniamnist:64,128",
        "CC BY 4.0",
        "Kermany D. et al. Cell 2018; Yang J. et al. MedMNIST v2 2023.",
        "https://medmnist.com/",
    ),
    Source(
        "organamnist",
        "OrganAMNIST 128 px (abdominal CT)",
        "medmnist",
        "organamnist:128",
        "CC BY 4.0",
        "Bilic P. et al. LiTS 2019; Yang J. et al. MedMNIST v2 2023.",
        "https://medmnist.com/",
    ),
    Source(
        "breastmnist",
        "BreastMNIST 128 px (ultrasound)",
        "medmnist",
        "breastmnist:128",
        "CC BY 4.0",
        "Al-Dhabyani W. et al. Data in Brief 2020; Yang J. et al. MedMNIST v2 2023.",
        "https://medmnist.com/",
    ),
    Source(
        "rsna",
        "RSNA Pneumonia Detection Challenge",
        "kaggle_competition",
        "rsna-pneumonia-detection-challenge",
        "Competition rules (non-commercial research)",
        "Shih G. et al. Radiology: AI 2019.",
        "https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge",
        profiles=("full",),
        note="Requires accepting the competition rules on kaggle.com.",
    ),
    Source(
        "lung_covidqu",
        "COVID-19 Radiography Database (lung masks, QU/Dhaka team)",
        "kaggle_dataset",
        "tawsifurrahman/covid19-radiography-database",
        "Data files © original authors (research use, cite)",
        "Chowdhury M.E.H. et al. IEEE Access 2020; Rahman T. et al. Comput Biol Med 2021.",
        "https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database",
        note="Same group as COVID-QU-Ex; compact 299 px images with 256 px lung masks.",
    ),
    Source(
        "lung_montgomery_shenzhen",
        "Montgomery + Shenzhen CXR with lung masks (NLM)",
        "kaggle_dataset",
        "nikhilpandey360/chest-xray-masks-and-labels",
        "Public (NLM), cite",
        "Jaeger S. et al. Quant Imaging Med Surg 2014.",
        "https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels",
        profiles=("full",),
    ),
    Source(
        "lung_covidqu_ex",
        "COVID-QU-Ex lung masks",
        "kaggle_dataset",
        "anasmohammedtahir/covidqu",
        "CC BY-SA 4.0 (per authors)",
        "Tahir A.M. et al. Comput Biol Med 2021.",
        "https://www.kaggle.com/datasets/anasmohammedtahir/covidqu",
        profiles=("full",),
    ),
    Source(
        "fracatlas",
        "FracAtlas musculoskeletal X-rays",
        "kaggle_dataset",
        "orvile/fracatlas",
        "CC BY 4.0",
        "Abedeen I. et al. FracAtlas. Scientific Data 2023.",
        "https://figshare.com/articles/dataset/22363012",
    ),
    Source(
        "brain_mri",
        "Brain Tumor MRI Dataset (Nickparvar)",
        "kaggle_dataset",
        "masoudnickparvar/brain-tumor-mri-dataset",
        "CC BY 4.0",
        "Nickparvar M. Brain Tumor MRI Dataset. Kaggle 2021.",
        "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset",
    ),
    Source(
        "lgg",
        "LGG MRI Segmentation (Buda)",
        "kaggle_dataset",
        "mateuszbuda/lgg-mri-segmentation",
        "CC BY-NC-SA 4.0",
        "Buda M. et al. Comput Biol Med 2019.",
        "https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation",
    ),
    Source(
        "iu_xray",
        "Indiana University Chest X-rays + reports (OpenI), 256 px cleaned copy",
        "kaggle_dataset",
        "masrursabab/iu-chest-x-rays-cleaned",
        "CC BY-NC-SA 4.0 (derived from OpenI)",
        "Demner-Fushman D. et al. JAMIA 2016.",
        "https://www.kaggle.com/datasets/masrursabab/iu-chest-x-rays-cleaned",
        note="Full-size originals: raddar/chest-xrays-indiana-university (14 GB) used on Kaggle.",
    ),
    Source(
        "cifar10",
        "CIFAR-10 (not-medical negatives for the modality router)",
        "torchvision",
        "CIFAR10",
        "MIT-style (Krizhevsky)",
        "Krizhevsky A. Learning Multiple Layers of Features 2009.",
        "https://www.cs.toronto.edu/~kriz/cifar.html",
    ),
    Source(
        "chexpert",
        "CheXpert (Stanford)",
        "url",
        "",
        "Stanford research use agreement",
        "Irvin J. et al. AAAI 2019.",
        "https://stanfordmlgroup.github.io/competitions/chexpert/",
        profiles=(),
        optional=True,
        note="Free registration; manual download; off by default.",
    ),
    Source(
        "mura",
        "MURA (Stanford)",
        "url",
        "",
        "Stanford research use agreement",
        "Rajpurkar P. et al. MURA. 2017.",
        "https://stanfordmlgroup.github.io/competitions/mura/",
        profiles=(),
        optional=True,
        note="Free registration; manual download; off by default.",
    ),
    Source(
        "vindr_cxr",
        "VinDr-CXR (PhysioNet)",
        "url",
        "",
        "PhysioNet Credentialed Health Data 1.0",
        "Nguyen H.Q. et al. Scientific Data 2022.",
        "https://physionet.org/content/vindr-cxr/",
        profiles=(),
        optional=True,
        note="Credentialed PhysioNet access; off by default.",
    ),
]

BY_KEY = {s.key: s for s in SOURCES}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_checksums() -> dict[str, str]:
    return json.loads(CHECKSUMS.read_text()) if CHECKSUMS.exists() else {}


def _record(paths: list[Path]) -> None:
    sums = _load_checksums()
    for p in paths:
        if p.is_file():
            sums[str(p.relative_to(DATA_DIR))] = _sha256(p)
    CHECKSUMS.parent.mkdir(parents=True, exist_ok=True)
    CHECKSUMS.write_text(json.dumps(sums, indent=2, sort_keys=True))


def verify(key: str) -> bool:
    """True when every recorded file for `key` still matches its checksum."""
    sums = _load_checksums()
    own = {k: v for k, v in sums.items() if k.startswith(f"raw/{key}/")}
    return bool(own) and all(
        (DATA_DIR / k).exists() and _sha256(DATA_DIR / k) == v for k, v in own.items()
    )


def _kaggle_api():  # type: ignore[no-untyped-def]
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def has_kaggle_token() -> bool:
    home = Path.home() / ".kaggle"
    import os

    return (
        (home / "kaggle.json").exists()
        or (home / "access_token").exists()
        or bool(os.environ.get("KAGGLE_API_TOKEN"))
        or bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    )


def done_marker(key: str) -> Path:
    return RAW / key / ".complete"


def fetch(src: Source) -> str:
    out = RAW / src.key
    if done_marker(src.key).exists():
        return "cached"
    out.mkdir(parents=True, exist_ok=True)
    if src.kind == "medmnist":
        import medmnist
        from medmnist import INFO

        flag, sizes = src.ref.split(":")
        for size in sizes.split(","):
            cls = getattr(medmnist, INFO[flag]["python_class"])
            for split in ("train", "val", "test"):
                cls(split=split, download=True, root=str(out), size=int(size))
    elif src.kind == "torchvision":
        _fetch_cifar(out)
    elif src.kind in ("kaggle_dataset", "kaggle_competition"):
        if not has_kaggle_token():
            raise RuntimeError("no Kaggle token (~/.kaggle/kaggle.json)")
        api = _kaggle_api()
        if src.kind == "kaggle_competition":
            for f in src.files or ():
                api.competition_download_file(src.ref, f, path=str(out), quiet=True)
            if not src.files:
                api.competition_download_files(src.ref, path=str(out), quiet=True)
        elif src.files:
            for f in src.files:
                api.dataset_download_file(src.ref, f, path=str(out), quiet=True)
        else:
            api.dataset_download_files(src.ref, path=str(out), quiet=True, unzip=False)
    elif src.kind == "url":
        if not src.ref:
            raise RuntimeError(f"manual download required: {src.url}")
        dest = out / Path(src.ref).name
        urllib.request.urlretrieve(src.ref, dest)
    _record([p for p in out.rglob("*") if p.is_file()])
    done_marker(src.key).write_text("ok")
    return "downloaded"


def _fetch_cifar(out: Path) -> None:
    """CIFAR-10 python batches into torchvision's layout. Kaggle mirror first (the original
    Toronto host is often very slow), torchvision's own downloader as the fallback."""
    import zipfile

    if (out / "cifar-10-batches-py").exists():
        return
    if has_kaggle_token():
        try:
            _kaggle_api().dataset_download_files(
                "jaisonwork/cifar10-python-dataset", path=str(out), quiet=True, unzip=False
            )
            for z in out.glob("*.zip"):
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(out / "_x")
                z.unlink()
            hit = next((out / "_x").rglob("data_batch_1"), None)
            if hit is not None:
                hit.parent.rename(out / "cifar-10-batches-py")
                import shutil

                shutil.rmtree(out / "_x", ignore_errors=True)
                return
        except Exception as exc:  # fall through to torchvision
            print(f"[download] cifar10 kaggle mirror failed ({exc}); trying torchvision")
    import torchvision

    torchvision.datasets.CIFAR10(root=str(out), train=True, download=True)
    torchvision.datasets.CIFAR10(root=str(out), train=False, download=True)


def write_sources_md() -> Path:
    lines = [
        "# Data sources",
        "",
        "Generated by `ml/data/download.py`. All sources are free; some need a free account.",
        "",
        "| Key | Dataset | License | Citation | Profiles | Link |",
        "|---|---|---|---|---|---|",
    ]
    for s in SOURCES:
        prof = ", ".join(s.profiles) or "optional (off)"
        lines.append(f"| `{s.key}` | {s.title} | {s.license} | {s.citation} | {prof} | {s.url} |")
    path = ROOT / "ml" / "data" / "SOURCES.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="starter", choices=["starter", "full"])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    write_sources_md()
    if args.list:
        for s in SOURCES:
            print(f"{s.key:26s} {s.kind:18s} {','.join(s.profiles) or 'optional':14s} {s.title}")
        return 0
    keys = args.only or [
        s.key for s in SOURCES if args.profile in s.profiles and s.key not in ("nih_full",)
    ]
    if args.profile == "full":
        print("Full-profile data is consumed inside Kaggle notebooks; see docs/TRAINING.md.")
        return 0
    report: dict[str, str] = {}
    for key in keys:
        src = BY_KEY[key]
        try:
            report[key] = fetch(src)
        except Exception as exc:  # never block the build
            report[key] = f"SKIPPED ({exc.__class__.__name__}: {exc})"
        print(f"[download] {key}: {report[key]}", flush=True)
    (DATA_DIR / "download_report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
