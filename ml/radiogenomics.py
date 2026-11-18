"""Radiogenomics (C6) + brain-segmentation fairness (M29) on the LGG / TCGA cohort (110 patients).

    python -m ml.radiogenomics [--profile starter]

1. Per patient, radiomic features of the tumour are computed from the FLAIR slices and the expert
   masks: size (tumour volume proxy in voxels, slice extent), shape on the largest slice
   (circularity, solidity, eccentricity), intensity (mean, SD, skewness, entropy) and GLCM texture.
2. They are linked to TCGA molecular subtypes (RNASeq / methylation / copy-number / CoC clusters),
   histological grade and death, using logistic regression with repeated stratified 5-fold CV.
   A label-permutation baseline is reported next to every AUROC — with n ~ 100 most associations
   are weak, and we say so.
3. Fairness: per-patient Dice of OUR Attention U-Net (test patients) by sex, age band and race.

Genomic labels come from the dataset's data.csv (TCGA; Buda et al. 2019). Nothing here is a
clinical predictor.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from ml.common import ARTIFACTS, DATA_DIR, PROCESSED, write_json

TARGETS = {
    "RNASeqCluster": "RNA-seq cluster 1 vs rest",
    "MethylationCluster": "methylation cluster 4 vs rest",
    "CNCluster": "copy-number cluster 2 vs rest",
    "COCCluster": "cluster-of-clusters 2 vs rest",
    "neoplasm_histologic_grade": "WHO grade III vs II",
    "death01": "death during follow-up",
}
POSITIVE = {
    "RNASeqCluster": 1,
    "MethylationCluster": 4,
    "CNCluster": 2,
    "COCCluster": 2,
    "neoplasm_histologic_grade": 2,
    "death01": 1,
}


def patient_features(imgs: np.ndarray, masks: np.ndarray) -> dict[str, float]:
    from skimage.feature import graycomatrix, graycoprops
    from skimage.measure import label, regionprops

    areas = masks.reshape(len(masks), -1).sum(1)
    if areas.sum() == 0:
        return {}
    k = int(areas.argmax())
    m, img = masks[k] > 0, imgs[k]
    vals = np.concatenate([imgs[i][masks[i] > 0] for i in np.where(areas > 0)[0]]).astype(float)
    z = (vals - vals.mean()) / (vals.std() + 1e-9)
    hist, _ = np.histogram(vals, 32, (0, 256))
    p = hist / hist.sum()
    props = max(regionprops(label(m)), key=lambda r: r.area)
    ys, xs = np.where(m)
    crop = img[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] // 8
    q = np.where(m[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1], crop, 32).astype(np.uint8)
    g = graycomatrix(q, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], 33, symmetric=True)[:32, :32]
    g = g / np.maximum(g.sum((0, 1), keepdims=True), 1)
    return {
        "volume_vox": float(areas.sum()),
        "slices": float((areas > 0).sum()),
        "max_area": float(areas.max()),
        "circularity": float(4 * np.pi * props.area / max(props.perimeter, 1) ** 2),
        "solidity": float(props.solidity),
        "eccentricity": float(props.eccentricity),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "skew": float((z**3).mean()),
        "entropy": float(-(p[p > 0] * np.log2(p[p > 0])).sum()),
        **{
            f"glcm_{c}": float(graycoprops(g, c).mean())
            for c in ("contrast", "homogeneity", "energy")
        },
    }


def cv_auroc(X: np.ndarray, y: np.ndarray, repeats: int = 10, seed: int = 0) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    aucs = []
    for r in range(repeats):
        skf = StratifiedKFold(5, shuffle=True, random_state=seed + r)
        p = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
            clf.fit(X[tr], y[tr])
            p[te] = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y, p))
    return float(np.mean(aucs)), float(np.std(aucs))


def segmentation_fairness(profile: str, clin: pd.DataFrame) -> dict[str, Any]:
    import torch

    from ml.data.datasets import SegDataset
    from ml.evaluate import load_trained

    try:
        model, cfg = load_trained("brain_seg", profile, "cpu")
    except FileNotFoundError:
        return {}
    ds = SegDataset("lgg", "test", cfg["img_size"])
    inter: dict[str, float] = {}
    tot: dict[str, float] = {}
    with torch.no_grad():
        for i in range(0, len(ds), 64):
            batch = [ds[j] for j in range(i, min(i + 64, len(ds)))]
            x = torch.stack([b[0] for b in batch])
            y = torch.stack([b[1] for b in batch])
            pr = (model(x) > 0).float()
            for j, (pp, yy) in enumerate(zip(pr, y)):
                pid = ds.meta.loc[i + j, "patient_id"][:12]
                inter[pid] = inter.get(pid, 0) + float((pp * yy).sum())
                tot[pid] = tot.get(pid, 0) + float(pp.sum() + yy.sum())
    dice = {p: 2 * inter[p] / tot[p] for p in inter if tot[p] > 0}
    c = clin.set_index("Patient")
    rows = []
    for p, d in dice.items():
        if p not in c.index:
            continue
        r = c.loc[p]
        age = r["age_at_initial_pathologic"]
        if pd.isna(age):
            continue
        rows.append(
            {
                "dice": d,
                "sex": f"code {int(r['gender'])}" if pd.notna(r["gender"]) else "unknown",
                "age": "<40" if age < 40 else "40-59" if age < 60 else "60+",
                "race": f"code {int(r['race'])}" if pd.notna(r["race"]) else "unknown",
            }
        )
    df = pd.DataFrame(rows)
    out: dict[str, Any] = {
        "n_patients": len(df),
        "overall_dice": float(df["dice"].mean()),
        "groups": {},
    }
    for col in ("sex", "age", "race"):
        for g, sub in df.groupby(col):
            out["groups"][f"{col}:{g}"] = {
                "n": int(len(sub)),
                "dice": round(float(sub["dice"].mean()), 4),
            }
    for col in ("sex", "race"):
        vals = [v["dice"] for k, v in out["groups"].items() if k.startswith(col) and v["n"] >= 3]
        out[f"{col}_gap"] = round(max(vals) - min(vals), 4) if len(vals) > 1 else None
    out["note"] = (
        "Sex / race are shown as the numeric codes of the dataset's data.csv (the Kaggle copy does "
        "not document the code-to-category mapping, so we do not guess it); "
        "groups with n < 3 are shown but excluded from the gap. Small n — no fairness claim."
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="starter")
    args = ap.parse_args(argv)
    clin = pd.read_csv(DATA_DIR / "raw" / "lgg_genomics" / "kaggle_3m" / "data.csv")
    z = np.load(PROCESSED / "lgg" / "lgg_256.npz")
    meta = pd.read_csv(PROCESSED / "lgg" / "metadata.csv")
    meta["pid"] = meta["patient_id"].str[:12]
    feats = {}
    for pid, idx in meta.groupby("pid").groups.items():
        f = patient_features(z["images"][idx], z["masks"][idx])
        if f:
            feats[pid] = f
    F = pd.DataFrame.from_dict(feats, orient="index")
    df = F.join(clin.set_index("Patient"), how="inner")
    cols = list(F.columns)
    results = {}
    rng = np.random.default_rng(0)
    for t, desc in TARGETS.items():
        sub = df.dropna(subset=[t])
        y = (sub[t].to_numpy() == POSITIVE[t]).astype(int)
        if y.sum() < 8 or (1 - y).sum() < 8:
            continue
        X = sub[cols].to_numpy()
        auc, sd = cv_auroc(X, y)
        perm = [cv_auroc(X, rng.permutation(y), repeats=2, seed=k)[0] for k in range(10)]
        results[t] = {
            "description": desc,
            "n": int(len(y)),
            "n_positive": int(y.sum()),
            "cv_auroc": round(auc, 3),
            "cv_auroc_sd": round(sd, 3),
            "permutation_auroc_mean": round(float(np.mean(perm)), 3),
            "permutation_auroc_p95": round(float(np.percentile(perm, 95)), 3),
            "above_chance": bool(auc > np.percentile(perm, 95)),
        }
    corr = {
        c: round(
            float(np.corrcoef(df[c], (df["neoplasm_histologic_grade"] == 2).astype(float))[0, 1]), 3
        )
        for c in cols
        if df["neoplasm_histologic_grade"].notna().all()
    }
    out = {
        "model": "radiogenomics",
        "profile": args.profile,
        "task": "radiogenomics",
        "n_patients": int(len(df)),
        "features": cols,
        "test": {"targets": results},
        "feature_grade_correlation": corr,
        "fairness_brain_seg": segmentation_fairness(args.profile, clin),
        "method": "Expert-mask radiomics (FLAIR) -> logistic regression, repeated stratified 5-fold "
        "CV; permutation baseline (10 label shuffles).",
        "note": "Exploratory, n ~ 100; associations are hypotheses, not predictors.",
    }
    write_json(ARTIFACTS / "radiogenomics" / args.profile / "metrics.json", out)
    for t, r in results.items():
        print(
            f"[radiogenomics] {t}: AUROC {r['cv_auroc']} ± {r['cv_auroc_sd']} "
            f"(perm p95 {r['permutation_auroc_p95']}) above_chance={r['above_chance']}"
        )
    print("[radiogenomics] fairness", out["fairness_brain_seg"].get("groups"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
