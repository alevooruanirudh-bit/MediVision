"""PET-CT multimodal fusion study + tumour segmenter (C7, D-34).

    python -m ml.petct_fusion --profile starter

Data: TCIA Soft-tissue-Sarcoma FDG-PET/CT (Vallières 2015), the 4quant Kaggle release in which PET
and CT are already co-registered and resampled to 5 mm voxels, with expert tumour contours. Only 7
patients carry labels in that file, so this is a small, honest study:

* the same 2.5-D U-Net (3 adjacent slices per modality) is trained three ways - CT only, PET only,
  PET+CT early fusion (6 channels);
* evaluation is leave-one-patient-out (7 folds), 3-D Dice per held-out patient;
* the deployed PET+CT model is trained on 6 patients (DEMO_HOLDOUT kept out for the demo sample).

Quantitative PET read-outs (SUVmax, SUVmean, metabolic tumour volume, TLG) are computed from the
predicted mask at inference time. PET values in this file are SUV (body-weight).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ml.common import DATA_DIR, artifact_dir, get_device, seed_everything, write_json
from ml.models.unet import UNet

H5 = DATA_DIR / "raw" / "sarcoma_petct" / "lab_petct_vox_5.00mm.h5"
MODES = {"ct": ("ct",), "pet": ("pet",), "petct": ("ct", "pet")}
PAD = 112  # 100 x 100 slices padded to a multiple of 16
VOXEL_ML = 0.125  # 5 mm isotropic voxel = 0.125 ml
# The deployed model is trained on the other 6 patients so the demo sample is never seen in training
# (fixed in advance: the last patient id, not chosen by model output).
DEMO_HOLDOUT = "STS_031"
PROFILES = {"starter": {"epochs": 30, "base": 16}, "full": {"epochs": 60, "base": 32}}


def norm_ct(ct: np.ndarray) -> np.ndarray:
    """Soft-tissue window [-250, 350] HU -> [0, 1]."""
    return np.clip((ct + 250.0) / 600.0, 0, 1).astype(np.float32)


def norm_pet(suv: np.ndarray) -> np.ndarray:
    """log(1 + SUV) scaled so SUV 20 -> 1 (compresses the bladder / brain uptake)."""
    return np.clip(np.log1p(np.clip(suv, 0, None)) / np.log1p(20.0), 0, 1.5).astype(np.float32)


def load_patients() -> dict[str, dict[str, np.ndarray]]:
    import h5py

    if not H5.exists():
        raise FileNotFoundError(
            "sarcoma_petct not downloaded (python -m ml.data.download --only sarcoma_petct)"
        )
    out = {}
    with h5py.File(H5, "r") as f:
        for pid in f["ct_data"]:
            out[pid] = {
                "ct": np.asarray(f["ct_data"][pid], np.float32),
                "pet": np.asarray(f["pet_data"][pid], np.float32),
                "mask": (np.asarray(f["label_data"][pid]) > 0).astype(np.uint8),
            }
    return out


def stack_input(ct: np.ndarray, pet: np.ndarray, mode: str) -> np.ndarray:
    """(Z, C, PAD, PAD) float32: for each modality the slices z-1, z, z+1."""
    chans = []
    for m in MODES[mode]:
        v = norm_ct(ct) if m == "ct" else norm_pet(pet)
        vp = np.concatenate([v[:1], v, v[-1:]], 0)
        chans += [vp[:-2], vp[1:-1], vp[2:]]
    x = np.stack(chans, 1)
    h, w = x.shape[2:]
    return np.pad(x, ((0, 0), (0, 0), (0, PAD - h), (0, PAD - w)))


def dice3d(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = float((pred & gt).sum())
    s = float(pred.sum() + gt.sum())
    return 1.0 if s == 0 else 2 * inter / s


def _augment(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.rand(1) < 0.5:
        x, y = x.flip(-1), y.flip(-1)
    if torch.rand(1) < 0.5:
        x, y = x.flip(-2), y.flip(-2)
    k = int(torch.randint(0, 4, (1,)))
    x, y = torch.rot90(x, k, (-2, -1)), torch.rot90(y, k, (-2, -1))
    return x * (0.9 + 0.2 * torch.rand(1)), y


def train_one(
    pats: dict[str, dict[str, np.ndarray]],
    ids: list[str],
    mode: str,
    epochs: int,
    base: int,
    device: str,
    seed: int = 0,
) -> UNet:
    seed_everything(seed)
    xs, ys = [], []
    for pid in ids:
        p = pats[pid]
        x = stack_input(p["ct"], p["pet"], mode)
        y = np.pad(p["mask"], ((0, 0), (0, PAD - 100), (0, PAD - 100)))[:, None]
        xs.append(x)
        ys.append(y)
    X, Y = np.concatenate(xs), np.concatenate(ys).astype(np.float32)
    pos = np.where(Y.reshape(len(Y), -1).max(1) > 0)[0]
    neg = np.setdiff1d(np.arange(len(Y)), pos)
    model = UNet(X.shape[1], 1, base, 4).to(device)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        idx = np.concatenate([pos, rng.choice(neg, min(len(neg), len(pos)), replace=False)])
        rng.shuffle(idx)
        for b in range(0, len(idx), 16):
            j = idx[b : b + 16]
            x, y = _augment(torch.from_numpy(X[j]), torch.from_numpy(Y[j]))
            x, y = x.to(device), y.to(device)
            logit = model(x)
            p = torch.sigmoid(logit)
            dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
            loss = F.binary_cross_entropy_with_logits(logit, y) + dice
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return model.eval()


@torch.no_grad()
def predict_volume(
    model: torch.nn.Module, ct: np.ndarray, pet: np.ndarray, mode: str, device: str
) -> np.ndarray:
    x = torch.from_numpy(stack_input(ct, pet, mode))
    out = []
    for b in range(0, len(x), 32):
        out.append(torch.sigmoid(model(x[b : b + 32].to(device))).cpu().numpy()[:, 0])
    h, w = ct.shape[1:]
    return np.concatenate(out)[:, :h, :w]


def pet_readout(pet: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if mask.sum() == 0:
        return {"suv_max": 0.0, "suv_mean": 0.0, "mtv_ml": 0.0, "tlg": 0.0}
    v = pet[mask > 0]
    mtv = float(mask.sum() * VOXEL_ML)
    return {
        "suv_max": round(float(v.max()), 2),
        "suv_mean": round(float(v.mean()), 2),
        "mtv_ml": round(mtv, 1),
        "tlg": round(float(v.mean()) * mtv, 1),
    }


def _plots(
    cv: dict[str, Any], pats: dict[str, dict[str, np.ndarray]], model: UNet, device: str, out: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (out / "plots").mkdir(exist_ok=True)
    ids = list(cv["per_patient"])
    fig, ax = plt.subplots(figsize=(7, 3.4))
    w = 0.26
    for k, m in enumerate(MODES):
        ax.bar(
            np.arange(len(ids)) + (k - 1) * w,
            [cv["per_patient"][i][m] for i in ids],
            w,
            label=m.upper().replace("PETCT", "PET+CT"),
        )
    ax.set_xticks(range(len(ids)), ids, rotation=30, fontsize=8)
    ax.set_ylabel("3-D Dice (held-out patient)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Leave-one-patient-out tumour segmentation")
    fig.tight_layout()
    fig.savefig(out / "plots" / "dice_by_mode.png", dpi=130)
    plt.close(fig)

    pid = DEMO_HOLDOUT if DEMO_HOLDOUT in pats else max(pats, key=lambda i: pats[i]["mask"].sum())
    p = pats[pid]
    z = int(np.argmax(p["mask"].sum((1, 2))))
    pr = predict_volume(model, p["ct"], p["pet"], "petct", device) > 0.5
    fig, axs = plt.subplots(1, 3, figsize=(9, 3.2))
    axs[0].imshow(norm_ct(p["ct"][z]), cmap="gray")
    axs[0].set_title("CT")
    axs[1].imshow(norm_pet(p["pet"][z]), cmap="hot")
    axs[1].set_title("PET (SUV)")
    axs[2].imshow(norm_ct(p["ct"][z]), cmap="gray")
    axs[2].imshow(np.ma.masked_less(norm_pet(p["pet"][z]), 0.25), cmap="hot", alpha=0.55)
    axs[2].contour(p["mask"][z], [0.5], colors="cyan", linewidths=1)
    axs[2].contour(pr[z], [0.5], colors="lime", linewidths=1, linestyles="--")
    axs[2].set_title(f"{pid} fused (cyan expert, green ours)")
    for a in axs:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out / "plots" / "fusion_example.png", dpi=130)
    plt.close(fig)


def run(profile: str = "starter") -> dict[str, Any]:
    cfg = PROFILES[profile]
    device = get_device()
    t0 = time.time()
    pats = load_patients()
    ids = sorted(pats)
    per_patient: dict[str, dict[str, float]] = {i: {} for i in ids}
    for m in MODES:
        for held in ids:
            model = train_one(
                pats, [i for i in ids if i != held], m, cfg["epochs"], cfg["base"], device
            )
            p = pats[held]
            pr = predict_volume(model, p["ct"], p["pet"], m, device) > 0.5
            per_patient[held][m] = round(dice3d(pr, p["mask"] > 0), 3)
            print(f"[petct] {m:6s} held-out {held}: Dice {per_patient[held][m]:.3f}", flush=True)
    summary = {}
    for m in MODES:
        d = np.array([per_patient[i][m] for i in ids])
        summary[m] = {
            "mean_dice": round(float(d.mean()), 3),
            "sd": round(float(d.std(ddof=1)), 3),
            "median_dice": round(float(np.median(d)), 3),
        }
    wins = {
        f"petct_vs_{m}": int(sum(per_patient[i]["petct"] > per_patient[i][m] for i in ids))
        for m in ("ct", "pet")
    }
    final_ids = [i for i in ids if i != DEMO_HOLDOUT]
    final = train_one(pats, final_ids, "petct", cfg["epochs"], cfg["base"], device)
    demo = pats.get(DEMO_HOLDOUT)
    demo_dice = (
        round(
            dice3d(
                predict_volume(final, demo["ct"], demo["pet"], "petct", device) > 0.5,
                demo["mask"] > 0,
            ),
            3,
        )
        if demo is not None
        else None
    )
    out = artifact_dir("petct_seg", profile)
    model_cfg = {
        "task": "segmentation",
        "img_size": PAD,
        "dataset": "sarcoma_petct",
        "model": {"arch": "unet", "in_ch": 6, "base": cfg["base"], "depth": 4},
    }
    torch.save({"state_dict": final.state_dict(), "config": model_cfg}, out / "model.pt")
    write_json(out / "config.json", model_cfg)
    final_cpu = final.to("cpu").eval()
    dummy = torch.zeros(1, 6, PAD, PAD)
    torch.onnx.export(
        final_cpu,
        (dummy,),
        str(out / "model.onnx"),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "n"}, "logits": {0: "n"}},
        opset_version=18,
        dynamo=False,
    )
    cv = {"per_patient": per_patient}
    _plots(cv, pats, final_cpu, "cpu", out)
    metrics = {
        "name": "petct_seg",
        "profile": profile,
        "task": "segmentation_fusion",
        "dataset": "TCIA Soft-tissue-Sarcoma FDG-PET/CT (4quant 5 mm co-registered release)",
        "n_patients": len(ids),
        "protocol": "leave-one-patient-out, 3-D Dice on the held-out patient",
        "modes": summary,
        "per_patient": per_patient,
        "petct_wins": wins,
        "deployed_model_patients": final_ids,
        "demo_holdout": DEMO_HOLDOUT,
        "demo_holdout_dice": demo_dice,
        "fusion_gain_vs_ct": round(summary["petct"]["mean_dice"] - summary["ct"]["mean_dice"], 3),
        "fusion_gain_vs_pet": round(summary["petct"]["mean_dice"] - summary["pet"]["mean_dice"], 3),
        "caveat": "7 patients only - indicative, not a powered comparison.",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": round(time.time() - t0, 1),
    }
    write_json(out / "metrics.json", metrics)
    (out / "card.md").write_text(
        "# petct_seg - PET-CT fusion tumour segmenter (ours)\n\n"
        "2.5-D U-Net, 6 input channels (3 CT + 3 PET slices), trained from scratch on 7 TCIA\n"
        "soft-tissue-sarcoma patients (co-registered FDG-PET/CT, 5 mm); the deployed model uses 6\n"
        f"(patient {DEMO_HOLDOUT} is held out as the demo case). Leave-one-patient-out Dice:\n"
        + "\n".join(f"- {m}: {s['mean_dice']} (sd {s['sd']})" for m, s in summary.items())
        + "\n\nResearch prototype; tiny cohort; not for clinical use.\n"
    )
    print(f"[petct] done: {summary}", flush=True)
    return metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="starter")
    run(ap.parse_args(argv).profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
