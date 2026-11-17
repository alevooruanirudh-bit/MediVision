"""Convert raw downloads into compact, model-ready caches + patient-level split CSVs.

    python -m ml.data.prepare --all
    python -m ml.data.prepare --only fracatlas brain_mri

Each preparer reads archives directly from the zip (no extraction), writes `data/processed/<key>/`,
writes split CSVs into `ml/data/splits/`, and (optionally) deletes the raw archive to save disk.
Splits are grouped by patient wherever the dataset provides patient identifiers.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from ml.common import DATA_DIR, NIH_LABELS, PROCESSED, ROOT
from ml.data.splits import group_split, save_split, stratified_split

RAW = DATA_DIR / "raw"
SPLITS = ROOT / "ml" / "data" / "splits"


def _zip(key: str) -> zipfile.ZipFile:
    zips = sorted((RAW / key).glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"no archive for {key} in {RAW / key}")
    return zipfile.ZipFile(max(zips, key=lambda p: p.stat().st_size))


def _read_img(zf: zipfile.ZipFile, name: str, mode: str = "L") -> Image.Image:
    with zf.open(name) as fh:
        return Image.open(io.BytesIO(fh.read())).convert(mode)


def _sq(img: Image.Image, size: int) -> np.ndarray:
    return np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.uint8)


def _out(key: str) -> Path:
    d = PROCESSED / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def _maybe_unzip_csv(path: Path) -> Path:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            inner = zf.namelist()[0]
            target = path.parent / Path(inner).name
            target.write_bytes(zf.read(inner))
            return target
    return path


# ----------------------------------------------------------------------------- NIH metadata
def prepare_nih_meta() -> None:
    d = RAW / "nih_meta"
    entry = _maybe_unzip_csv(next(d.glob("Data_Entry_2017.csv*")))
    tv = _maybe_unzip_csv(next(d.glob("train_val_list.txt*")))
    test = (d / "test_list.txt").read_text().split()
    df = pd.read_csv(entry)
    df = df.rename(
        columns={
            "Image Index": "image",
            "Finding Labels": "labels",
            "Patient ID": "patient_id",
            "Patient Age": "age",
            "Patient Gender": "sex",
            "View Position": "view",
            "OriginalImagePixelSpacing[x": "spacing_x",
        }
    )
    keep = ["image", "labels", "patient_id", "age", "sex", "view", "spacing_x"]
    df = df[keep]
    for lab in NIH_LABELS:
        df[lab] = df["labels"].str.contains(lab).astype(np.uint8)
    out = _out("nih")
    df.to_csv(out / "metadata.csv", index=False)
    # Full-profile split: official test list (patient-disjoint), our grouped 90/10 train/val.
    test_set = set(test)
    df["split"] = np.where(df["image"].isin(test_set), "test", "trainval")
    trainval = df[df["split"] == "trainval"].copy()
    trainval["split"] = group_split(trainval["patient_id"].to_numpy(), val_frac=0.1, seed=42)
    full = pd.concat([trainval, df[df["split"] == "test"]])
    save_split(full[["image", "patient_id", "split"]], SPLITS / "nih_full.csv")
    _ = tv  # train_val list is implied by "not test"; kept for reference
    print(f"[prepare] nih_meta: {len(df)} rows, split counts {full['split'].value_counts().to_dict()}")


# ----------------------------------------------------------------------------- NIH sample
def prepare_nih_sample(size: int = 256, demo_size: int = 768) -> None:
    meta = pd.read_csv(PROCESSED / "nih" / "metadata.csv")
    official = pd.read_csv(SPLITS / "nih_full.csv").set_index("image")["split"]
    zf = _zip("nih_sample")
    names = [n for n in zf.namelist() if n.lower().endswith(".png") and "/images/" in n]
    by_name = {Path(n).name: n for n in names}
    sub = meta[meta["image"].isin(by_name)].copy().reset_index(drop=True)
    sub["official_split"] = sub["image"].map(official).fillna("trainval")
    out = _out("nih_sample")
    demo_dir = out / "demo_pool"
    demo_dir.mkdir(exist_ok=True)
    imgs = np.zeros((len(sub), size, size), dtype=np.uint8)
    demo_rows = []
    rng = np.random.default_rng(0)
    for i, row in sub.iterrows():
        img = _read_img(zf, by_name[row["image"]])
        imgs[i] = _sq(img, size)
        is_demo = row["official_split"] == "test" and (row["labels"] != "No Finding" or rng.random() < 0.05)
        if is_demo and len(demo_rows) < 120:
            img.resize((demo_size, demo_size), Image.BILINEAR).save(demo_dir / row["image"])
            demo_rows.append(row["image"])
    np.save(out / f"images_{size}.npy", imgs)
    # Patient-level split of this sample. "test" = official test-list patients (never seen by the
    # ChestMNIST-trained starter models, whose training set is the official train_val list).
    sub["split"] = np.where(sub["official_split"] == "test", "test", "trainval")
    tv_mask = sub["split"] == "trainval"
    sub.loc[tv_mask, "split"] = group_split(sub.loc[tv_mask, "patient_id"].to_numpy(), 0.15, 42)
    sub.to_csv(out / "metadata.csv", index=False)
    save_split(sub[["image", "patient_id", "split"]], SPLITS / "nih_sample.csv")
    print(f"[prepare] nih_sample: {len(sub)} images, demo pool {len(demo_rows)}")


# ----------------------------------------------------------------------------- Lungs
def prepare_lung_covidqu(size: int = 256, per_class: int | None = None) -> None:
    zf = _zip("lung_covidqu")
    names = zf.namelist()
    masks = {n for n in names if "/masks/" in n and n.endswith(".png")}
    pairs = []
    for m in sorted(masks):
        img = m.replace("/masks/", "/images/")
        if img in names:
            cls = m.split("/")[-3]
            pairs.append((img, m, cls))
    df = pd.DataFrame(pairs, columns=["image", "mask", "cls"])
    caps = per_class or {"Normal": 1600, "Lung_Opacity": 600, "COVID": 500, "Viral Pneumonia": 400}
    if isinstance(caps, dict):
        parts = [g.sample(min(len(g), caps.get(str(c), 400)), random_state=0)
                 for c, g in df.groupby("cls")]
        df = pd.concat(parts).reset_index(drop=True)
    imgs = np.zeros((len(df), size, size), dtype=np.uint8)
    msk = np.zeros((len(df), size, size), dtype=np.uint8)
    for i, r in df.iterrows():
        imgs[i] = _sq(_read_img(zf, r["image"]), size)
        m = _read_img(zf, r["mask"]).resize((size, size), Image.NEAREST)
        msk[i] = (np.asarray(m) > 127).astype(np.uint8)
    out = _out("lung")
    np.savez_compressed(out / f"lung_{size}.npz", images=imgs, masks=msk)
    df["id"] = [Path(p).stem for p in df["image"]]
    # No patient IDs are published for this database -> stratified random split by class (D-07).
    df["split"] = stratified_split(df["cls"].to_numpy(), (0.75, 0.1, 0.15), seed=42)
    df[["id", "cls", "split"]].to_csv(out / "metadata.csv", index=False)
    save_split(df[["id", "cls", "split"]], SPLITS / "lung.csv")
    print(f"[prepare] lung: {len(df)} image/mask pairs")


# ----------------------------------------------------------------------------- FracAtlas
def prepare_fracatlas(max_side: int = 512) -> None:
    zf = _zip("fracatlas")
    names = zf.namelist()
    root = next(n.split("FracAtlas/")[0] for n in names if n.endswith("dataset.csv")) + "FracAtlas/"
    meta = pd.read_csv(io.BytesIO(zf.read(root + "dataset.csv")))
    img_path = {Path(n).name: n for n in names if n.endswith(".jpg") and "/images/" in n}
    out = _out("fracatlas")
    (out / "images").mkdir(exist_ok=True)
    records = []
    for _, r in meta.iterrows():
        name = r["image_id"]
        img = _read_img(zf, img_path[name])
        w, h = img.size
        scale = max_side / max(w, h)
        nw, nh = round(w * scale), round(h * scale)
        img.resize((nw, nh), Image.BILINEAR).save(out / "images" / name, quality=92)
        boxes = []
        yolo = root + f"Annotations/YOLO/{Path(name).stem}.txt"
        if yolo in names:
            for line in zf.read(yolo).decode().strip().splitlines():
                parts = line.split()
                if len(parts) == 5:
                    _, cx, cy, bw, bh = map(float, parts)
                    boxes.append(
                        [(cx - bw / 2) * nw, (cy - bh / 2) * nh, (cx + bw / 2) * nw, (cy + bh / 2) * nh]
                    )
        region = next((k for k in ("hand", "leg", "hip", "shoulder") if r[k] == 1), "mixed")
        records.append(
            {"image": name, "fractured": int(r["fractured"]), "region": region, "width": nw,
             "height": nh, "boxes": boxes}
        )
    df = pd.DataFrame(records)
    # Official "Fracture Split" for fractured images; negatives split 70/15/15 at random.
    split = {}
    for s, f in (("train", "train.csv"), ("val", "valid.csv"), ("test", "test.csv")):
        p = root + f"Utilities/Fracture Split/{f}"
        if p in names:
            for n in pd.read_csv(io.BytesIO(zf.read(p))).iloc[:, 0]:
                split[str(n)] = s
    df["split"] = df["image"].map(split)
    neg = df["split"].isna()
    df.loc[neg, "split"] = stratified_split(df.loc[neg, "fractured"].to_numpy(), (0.7, 0.15, 0.15), 42)
    (out / "annotations.json").write_text(json.dumps(df.to_dict(orient="records")))
    save_split(df[["image", "fractured", "split"]], SPLITS / "fracatlas.csv")
    print(f"[prepare] fracatlas: {len(df)} images, {int(df.fractured.sum())} fractured")


# ----------------------------------------------------------------------------- Brain MRI
def prepare_brain_mri(size: int = 224) -> None:
    from ml.common import BRAIN_CLASSES

    zf = _zip("brain_mri")
    rows = []
    for n in zf.namelist():
        parts = n.split("/")
        if n.lower().endswith((".jpg", ".png")) and len(parts) >= 3 and parts[-2] in BRAIN_CLASSES:
            rows.append((n, parts[-2], "test" if "Testing" in n else "trainval"))
    df = pd.DataFrame(rows, columns=["path", "cls", "split"]).reset_index(drop=True)
    imgs = np.zeros((len(df), size, size), dtype=np.uint8)
    for i, r in df.iterrows():
        imgs[i] = _sq(_read_img(zf, r["path"]), size)
    tv = df["split"] == "trainval"
    df.loc[tv, "split"] = stratified_split(df.loc[tv, "cls"].to_numpy(), (0.85, 0.15, 0.0), 42)
    df["label"] = df["cls"].map({c: i for i, c in enumerate(BRAIN_CLASSES)})
    df["id"] = [f"{Path(p).parent.name}_{Path(p).stem}" for p in df["path"]]
    out = _out("brain_mri")
    np.save(out / f"images_{size}.npy", imgs)
    df[["id", "cls", "label", "split"]].to_csv(out / "metadata.csv", index=False)
    save_split(df[["id", "cls", "split"]], SPLITS / "brain_mri.csv")
    print(f"[prepare] brain_mri: {len(df)} images {df['split'].value_counts().to_dict()}")


# ----------------------------------------------------------------------------- LGG
def prepare_lgg(size: int = 256) -> None:
    zf = _zip("lgg")
    masks = sorted(
        n for n in zf.namelist()
        if n.endswith("_mask.tif") and n.startswith("kaggle_3m/")
    )
    imgs = np.zeros((len(masks), size, size), dtype=np.uint8)
    msk = np.zeros((len(masks), size, size), dtype=np.uint8)
    rows = []
    for i, m in enumerate(masks):
        img = _read_img(zf, m.replace("_mask.tif", ".tif"), mode="RGB")
        flair = np.asarray(img)[..., 1]  # channel order: pre-contrast, FLAIR, post-contrast
        imgs[i] = _sq(Image.fromarray(flair), size)
        msk[i] = (np.asarray(_read_img(zf, m).resize((size, size), Image.NEAREST)) > 127)
        patient = m.split("/")[1]
        rows.append({"id": Path(m).stem.replace("_mask", ""), "patient_id": patient,
                     "has_tumor": int(msk[i].any())})
    df = pd.DataFrame(rows)
    df["split"] = group_split(df["patient_id"].to_numpy(), val_frac=0.15, seed=42, test_frac=0.15)
    out = _out("lgg")
    np.savez_compressed(out / f"lgg_{size}.npz", images=imgs, masks=msk)
    df.to_csv(out / "metadata.csv", index=False)
    save_split(df[["id", "patient_id", "split"]], SPLITS / "lgg.csv")
    print(f"[prepare] lgg: {len(df)} slices, {df.patient_id.nunique()} patients")


# ----------------------------------------------------------------------------- IU X-ray
def prepare_iu_xray(size: int = 224) -> None:
    zf = _zip("iu_xray")
    names = zf.namelist()
    csv_name = next(n for n in names if n.endswith(".csv"))
    df = pd.read_csv(io.BytesIO(zf.read(csv_name)))
    df.columns = [c.strip().lower() for c in df.columns]
    img_col = next(c for c in df.columns if "image" in c or "filename" in c or "path" in c)
    find_col = next((c for c in df.columns if "finding" in c), None)
    imp_col = next((c for c in df.columns if "impression" in c), None)
    uid_col = next((c for c in df.columns if c in ("uid", "id", "patient_id")), None)
    proj_col = next((c for c in df.columns if "projection" in c or "view" in c), None)
    pool = [n for n in names if n.lower().endswith(".png")]
    dirs = sorted({str(Path(n).parent) for n in pool})
    best = min(dirs, key=lambda d: abs(int("".join(ch for ch in d.split("/")[-1] if ch.isdigit()) or 0) - 256))
    by_name = {Path(n).name: n for n in pool if str(Path(n).parent) == best}
    rows, imgs = [], []
    for _, r in df.iterrows():
        fname = Path(str(r[img_col])).name
        if fname not in by_name:
            continue
        if proj_col and str(r[proj_col]).lower().startswith("lat"):
            continue  # frontal views only
        findings = str(r[find_col]) if find_col and pd.notna(r[find_col]) else ""
        impression = str(r[imp_col]) if imp_col and pd.notna(r[imp_col]) else ""
        if not (findings or impression):
            continue
        imgs.append(_sq(_read_img(zf, by_name[fname]), size))
        rows.append({"image": fname, "uid": str(r[uid_col]) if uid_col else fname.split("_")[0],
                     "findings": findings, "impression": impression})
    out = _out("iu_xray")
    meta = pd.DataFrame(rows)
    meta["split"] = group_split(meta["uid"].to_numpy(), val_frac=0.1, seed=42, test_frac=0.1)
    np.save(out / f"images_{size}.npy", np.stack(imgs))
    meta.to_csv(out / "reports.csv", index=False)
    save_split(meta[["image", "uid", "split"]], SPLITS / "iu_xray.csv")
    print(f"[prepare] iu_xray: {len(meta)} frontal images with reports (from {best})")


PREPARERS: dict[str, Callable[[], None]] = {
    "nih_meta": prepare_nih_meta,
    "nih_sample": prepare_nih_sample,
    "lung_covidqu": prepare_lung_covidqu,
    "fracatlas": prepare_fracatlas,
    "brain_mri": prepare_brain_mri,
    "lgg": prepare_lgg,
    "iu_xray": prepare_iu_xray,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--delete-raw", action="store_true", help="delete raw archives after success")
    args = ap.parse_args(argv)
    keys = list(PREPARERS) if args.all else (args.only or [])
    status = {}
    for k in keys:
        try:
            PREPARERS[k]()
            status[k] = "ok"
            if args.delete_raw and k not in ("nih_meta",):
                for z in (RAW / k).glob("*.zip"):
                    z.unlink()
        except FileNotFoundError as exc:
            status[k] = f"skipped: {exc}"
            print(f"[prepare] {k}: skipped ({exc})")
    (PROCESSED / "prepare_report.json").write_text(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
