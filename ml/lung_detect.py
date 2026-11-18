"""Whole-scan lung-nodule detection on chest CT (M5, D-36).

    python -m ml.lung_detect --profile starter            # preprocess (cached) + train + FROC eval

Pipeline (identical at inference in the backend):
  1 mm isotropic resampling -> lung mask -> multi-scale LoG blob candidates (<= 600 / scan)
  -> our 3-D CNN scores a 32 mm cube around each candidate -> ranked detections.

Data: LUNA16 (LIDC-IDRI) - only a laptop-sized subset of subset0 is used (see download.py). Scans
are split by series (= patient) into train / val / test. Positives are the radiologist-annotated
nodules (>= 3 mm, agreed by >= 3 of 4 readers) plus candidates that fall inside one; negatives are
all other candidates (hard negatives). Evaluation follows the LUNA16 FROC protocol: a nodule is hit
when a detection lies within its radius; sensitivity is read at 1/8 ... 8 false positives per scan,
CPM = their mean. The candidate generator's own recall bounds the achievable sensitivity.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ml.common import PROCESSED, artifact_dir, get_device, seed_everything, write_json
from ml.data import luna as L
from ml.models.cnn3d import SmallCNN3D

CACHE = PROCESSED / "luna16"
FP_RATES = [0.125, 0.25, 0.5, 1, 2, 4, 8]
PROFILES = {"starter": {"epochs": 25, "width": 16}, "full": {"epochs": 60, "width": 32}}
BIG = 48  # positive cubes are cached larger so training can jitter the 32 mm crop


def split_series(series: list[str], ann: dict[str, Any], seed: int = 42) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    pos = [s for s in series if s in ann]
    neg = [s for s in series if s not in ann]
    out = {}
    for group in (pos, neg):
        g = list(group)
        rng.shuffle(g)
        n_te, n_va = round(len(g) * 0.3), round(len(g) * 0.15)
        for i, s in enumerate(g):
            out[s] = "test" if i < n_te else "val" if i < n_te + n_va else "train"
    return out


def nodule_hits(cands: np.ndarray, nodules: list[tuple[np.ndarray, float]]) -> np.ndarray:
    """(n_cand,) index of the nodule each candidate falls inside (radius, min 3 mm), else -1."""
    hit = np.full(len(cands), -1)
    for j, (c, d) in enumerate(nodules):
        if len(cands):
            inside = np.linalg.norm(cands - c, axis=1) <= max(d / 2, 3.0)
            hit[inside & (hit < 0)] = j
    return hit


def preprocess(uid: str, ann: dict[str, Any]) -> Path:
    out = CACHE / f"{uid}.npz"
    if out.exists():
        return out
    vol, origin = L.load_scan_iso(uid)
    mask = L.lung_mask(vol)
    cands = L.candidates(vol, mask)
    nods = [(w - origin, d) for w, d in ann.get(uid, [])]
    hit = nodule_hits(cands, nods)
    patches = (
        np.stack([L.extract_patch(vol, c) for c in cands])
        if len(cands)
        else np.zeros((0, 32, 32, 32), np.int16)
    )
    big = (
        np.stack([L.extract_patch(vol, c, BIG) for c, _ in nods])
        if nods
        else np.zeros((0, BIG, BIG, BIG), np.int16)
    )
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        cands=cands,
        hit=hit,
        patches=patches,
        nod_centres=np.array([c for c, _ in nods]).reshape(-1, 3),
        nod_diam=np.array([d for _, d in nods]),
        nod_cubes=big,
        shape=np.array(vol.shape),
        lung_frac=np.float32(mask.mean()),
    )
    return out


def _aug(x: torch.Tensor) -> torch.Tensor:
    for ax in (2, 3, 4):
        if torch.rand(1) < 0.5:
            x = x.flip(ax)
    k = int(torch.randint(0, 4, (1,)))
    return torch.rot90(x, k, (3, 4))


def _crop(cube: np.ndarray, rng: np.random.Generator, jitter: int = 4) -> np.ndarray:
    o = (BIG - 32) // 2 + rng.integers(-jitter, jitter + 1, 3)
    return cube[o[0] : o[0] + 32, o[1] : o[1] + 32, o[2] : o[2] + 32]


def load_split(paths: list[Path]) -> dict[str, Any]:
    P, Y, cubes = [], [], []
    for p in paths:
        z = np.load(p)
        P.append(z["patches"])
        Y.append((z["hit"] >= 0).astype(np.float32))
        cubes.append(z["nod_cubes"])
    return {"patches": np.concatenate(P), "y": np.concatenate(Y), "cubes": np.concatenate(cubes)}


def train_model(
    tr: dict[str, Any], va_paths: list[Path], epochs: int, width: int, device: str
) -> tuple[SmallCNN3D, list[dict[str, float]]]:
    seed_everything(0)
    rng = np.random.default_rng(0)
    model = SmallCNN3D(1, width, 0.3).to(device)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    pos_idx = np.where(tr["y"] > 0)[0]
    neg_idx = np.where(tr["y"] == 0)[0]
    n_cubes = len(tr["cubes"])
    best, best_state, hist = -1.0, None, []
    for ep in range(epochs):
        model.train()
        # per epoch: every negative once is too many - sample 6 negatives per positive draw
        n_pos = max(n_cubes * 8, len(pos_idx) * 2)
        negs = rng.choice(neg_idx, min(len(neg_idx), n_pos * 3), replace=False)
        items: list[tuple[np.ndarray, float]] = []
        for _ in range(n_pos):
            if n_cubes and rng.random() < 0.7:
                items.append((_crop(tr["cubes"][rng.integers(n_cubes)], rng), 1.0))
            elif len(pos_idx):
                items.append((tr["patches"][rng.choice(pos_idx)], 1.0))
        items += [(tr["patches"][i], 0.0) for i in negs]
        order = rng.permutation(len(items))
        losses = []
        for b in range(0, len(order), 32):
            batch = [items[i] for i in order[b : b + 32]]
            x = torch.from_numpy(np.stack([L.norm_patch(p) for p, _ in batch]))[:, None]
            y = torch.tensor([[t] for _, t in batch])
            x = _aug(x).to(device)
            loss = F.binary_cross_entropy_with_logits(model(x), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        sched.step()
        cpm = froc(score_scans(model, va_paths, device))["cpm"]
        hist.append(
            {"epoch": ep + 1, "train_loss": round(float(np.mean(losses)), 4), "val_cpm": cpm}
        )
        print(
            f"[lung_det] epoch {ep + 1}/{epochs} loss {np.mean(losses):.4f} val CPM {cpm:.3f}",
            flush=True,
        )
        if cpm >= best:
            best, best_state = cpm, {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval(), hist


@torch.no_grad()
def score_patches(model: torch.nn.Module, patches: np.ndarray, device: str) -> np.ndarray:
    out = []
    for b in range(0, len(patches), 64):
        x = torch.from_numpy(np.stack([L.norm_patch(p) for p in patches[b : b + 64]]))[:, None].to(
            device
        )
        out.append(torch.sigmoid(model(x)).cpu().numpy()[:, 0])
    return np.concatenate(out) if out else np.zeros(0)


def score_scans(model: torch.nn.Module, paths: list[Path], device: str) -> list[dict[str, Any]]:
    model.eval()
    res = []
    for p in paths:
        z = np.load(p)
        res.append(
            {
                "probs": score_patches(model, z["patches"], device),
                "hit": z["hit"],
                "n_nodules": len(z["nod_diam"]),
                "diam": z["nod_diam"],
            }
        )
    return res


def froc(scans: list[dict[str, Any]]) -> dict[str, Any]:
    """LUNA16-style FROC. Each candidate: (prob, nodule id or -1). A nodule counts once."""
    n_scans = len(scans)
    n_nod = sum(s["n_nodules"] for s in scans)
    if n_scans == 0 or n_nod == 0:
        return {"cpm": 0.0, "sens": {}, "candidate_recall": 0.0}
    recs = []
    offset = 0
    for s in scans:
        for p, h in zip(s["probs"], s["hit"], strict=True):
            recs.append((float(p), offset + int(h) if h >= 0 else -1))
        offset += s["n_nodules"]
    recs.sort(key=lambda r: -r[0])
    found: set[int] = set()
    fps = 0
    curve = []  # (fp per scan, sensitivity, threshold)
    for p, h in recs:
        if h >= 0:
            found.add(h)
        else:
            fps += 1
        curve.append((fps / n_scans, len(found) / n_nod, p))
    sens = {}
    for r in FP_RATES:
        ok = [c[1] for c in curve if c[0] <= r]
        sens[str(r)] = round(max(ok) if ok else 0.0, 3)
    cand_recall = len({h for _, h in recs if h >= 0}) / n_nod
    thr_1fp = max((c for c in curve if c[0] <= 1.0), key=lambda c: c[1], default=(0, 0, 0.5))[2]
    return {
        "cpm": round(float(np.mean(list(sens.values()))), 3),
        "sens": sens,
        "candidate_recall": round(cand_recall, 3),
        "n_scans": n_scans,
        "n_nodules": n_nod,
        "threshold_at_1fp": round(float(thr_1fp), 4),
        "curve": curve,
    }


def _plot(f: dict[str, Any], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (out / "plots").mkdir(exist_ok=True)
    c = np.array([(a, b) for a, b, _ in f["curve"]])
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(np.maximum(c[:, 0], 1e-3), c[:, 1], lw=2)
    ax.axhline(
        f["candidate_recall"], ls="--", lw=1, color="gray", label="candidate recall (upper bound)"
    )
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.125, 16)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positives per scan")
    ax.set_ylabel("Sensitivity")
    ax.set_title(f"Whole-scan nodule detection FROC (test, CPM {f['cpm']:.3f})", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "plots" / "froc.png", dpi=130)
    plt.close(fig)


def run(profile: str = "starter") -> dict[str, Any]:
    cfg = PROFILES[profile]
    t0 = time.time()
    ann = L.annotations()
    series_file = L.LUNA / "series.txt"
    if not series_file.exists():
        raise FileNotFoundError("luna16 not downloaded (python -m ml.data.download --only luna16)")
    series = [
        s for s in series_file.read_text().split() if (L.LUNA / "scans" / f"{s}.mhd").exists()
    ]
    split = split_series(series, ann)
    paths: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    for i, uid in enumerate(series):
        paths[split[uid]].append(preprocess(uid, ann))
        print(f"[lung_det] preprocessed {i + 1}/{len(series)}", flush=True)
    device = get_device()
    tr = load_split(paths["train"])
    print(
        f"[lung_det] train candidates {len(tr['y'])} (pos {int(tr['y'].sum())}), nodule cubes {len(tr['cubes'])}",
        flush=True,
    )
    model, hist = train_model(tr, paths["val"], cfg["epochs"], cfg["width"], device)
    val = froc(score_scans(model, paths["val"], device))
    test = froc(score_scans(model, paths["test"], device))
    out = artifact_dir("lung_nodule_det", profile)
    model_cfg = {
        "task": "binary",
        "label_name": "Lung nodule",
        "img_size": 32,
        "dataset": "luna16",
        "model": {"arch": "cnn3d", "width": cfg["width"], "dropout": 0.3},
    }
    torch.save({"state_dict": model.state_dict(), "config": model_cfg}, out / "model.pt")
    write_json(out / "config.json", model_cfg)
    cpu = model.to("cpu").eval()
    torch.onnx.export(
        cpu,
        (torch.zeros(1, 1, 32, 32, 32),),
        str(out / "model.onnx"),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "n"}, "logits": {0: "n"}},
        opset_version=18,
        dynamo=False,
    )
    _plot(test, out)
    metrics = {
        "name": "lung_nodule_det",
        "profile": profile,
        "task": "detection_3d",
        "dataset": "LUNA16 (LIDC-IDRI) subset0 - laptop subset",
        "scans": {k: len(v) for k, v in paths.items()},
        "nodules": {
            k: int(sum(len(ann.get(s, [])) for s in series if split[s] == k)) for k in paths
        },
        "test_froc": {k: v for k, v in test.items() if k != "curve"},
        "val_froc": {k: v for k, v in val.items() if k != "curve"},
        "operating_threshold": val["threshold_at_1fp"],
        "operating_point": "1 false positive / scan on the validation scans",
        "history": hist,
        "series_split": split,
        "caveat": "Small subset of LUNA16 (one of ten folds, laptop budget); the LUNA16 'excluded' "
        "findings are counted as false positives here, which is stricter than the official script.",
        "seconds": round(time.time() - t0, 1),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_json(out / "metrics.json", metrics)
    (out / "card.md").write_text(
        "# lung_nodule_det - whole-scan lung-nodule detector (ours)\n\n"
        "Lung mask + multi-scale LoG candidates + 3-D CNN (32 mm cubes, from scratch) on a LUNA16 subset.\n"
        f"Test FROC CPM {test['cpm']}; sensitivity at 1 FP/scan {test['sens'].get('1')}; candidate recall "
        f"{test['candidate_recall']}.\n\nResearch prototype; not for clinical use.\n"
    )
    print(f"[lung_det] test {metrics['test_froc']}", flush=True)
    return metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="starter")
    ap.add_argument("--preprocess-only", action="store_true")
    a = ap.parse_args(argv)
    if a.preprocess_only:
        ann = L.annotations()
        for uid in (L.LUNA / "series.txt").read_text().split():
            if (L.LUNA / "scans" / f"{uid}.mhd").exists():
                preprocess(uid, ann)
                print(f"[lung_det] cached {uid[-8:]}", flush=True)
        return 0
    run(a.profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
