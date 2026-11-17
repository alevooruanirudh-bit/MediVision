"""Evaluation: metrics, calibration (temperature scaling), operating thresholds, fairness, plots.

    python -m ml.evaluate --all --profile starter
    python -m ml.evaluate --model chest_ensemble --profile starter

Every model gets artifacts/<model>/<profile>/{metrics.json, plots/*.png, card.md}. The chest ensemble
also gets calibration.json (per-member per-label temperatures, Youden thresholds, OOD energy cut-off).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from ml.common import (  # noqa: E402
    ARTIFACTS,
    BRAIN_CLASSES,
    NIH_LABELS,
    ROUTER_CLASSES,
    artifact_dir,
    get_device,
    load_config,
    write_json,
)
from ml.metrics import (  # noqa: E402
    classification_report,
    dice_iou,
    fit_temperature,
    fit_temperature_multiclass,
    map50,
    multilabel_auroc,
    psnr,
    reliability,
    roc_points,
    safe_auroc,
    sens_spec,
    sigmoid,
    ssim,
    youden_threshold,
)
from ml.models import build_model  # noqa: E402

CHEST_MEMBERS = ["chest_densenet121", "chest_efficientnet_b2", "chest_convnext_tiny"]
PLOT_STYLE = {"figure.facecolor": "#0b1020", "axes.facecolor": "#0b1020", "axes.edgecolor": "#5b6b8c",
              "text.color": "#e2e8f0", "axes.labelcolor": "#e2e8f0", "xtick.color": "#94a3b8",
              "ytick.color": "#94a3b8", "grid.color": "#1e293b"}


def load_trained(name: str, profile: str, device: str = "cpu") -> tuple[torch.nn.Module, dict[str, Any]]:
    ck = torch.load(ARTIFACTS / name / profile / "model.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), cfg


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _history(name: str, profile: str) -> list[dict[str, Any]]:
    p = ARTIFACTS / name / profile / "history.json"
    return json.loads(p.read_text()) if p.exists() else []


# ----------------------------------------------------------------------------- plots
def plot_roc(curves: dict[str, list[list[float]]], aucs: dict[str, float | None], path: Path,
             title: str) -> None:
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6, 6))
        for k, pts in curves.items():
            if pts:
                a = np.array(pts)
                ax.plot(a[:, 0], a[:, 1], lw=1.4, label=f"{k} ({(aucs.get(k) or 0):.2f})")
        ax.plot([0, 1], [0, 1], "--", color="#475569", lw=1)
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title=title)
        ax.legend(fontsize=7, loc="lower right", frameon=False)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)


def plot_reliability(rel: dict[str, Any], path: Path, title: str) -> None:
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(5, 5))
        c = [b["confidence"] for b in rel["bins"]]
        a = [b["accuracy"] for b in rel["bins"]]
        ax.plot([0, 1], [0, 1], "--", color="#475569")
        ax.plot(c, a, "o-", color="#22d3ee")
        ax.set(xlabel="Predicted probability", ylabel="Observed frequency",
               title=f"{title} (ECE {rel['ece']:.3f})")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)


def plot_confusion(cm: list[list[int]], classes: list[str], path: Path, title: str) -> None:
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 5))
        m = np.array(cm)
        ax.imshow(m, cmap="magma")
        ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(classes)), classes, fontsize=8)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if m[i, j] < m.max() / 2 else "black")
        ax.set(xlabel="Predicted", ylabel="True", title=title)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)


def plot_history(hist: list[dict[str, Any]], path: Path, title: str) -> None:
    if not hist:
        return
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ep = [h["epoch"] for h in hist]
        ax.plot(ep, [h["train_loss"] for h in hist], color="#8b5cf6", label="train loss")
        ax2 = ax.twinx()
        for k in hist[0]:
            if k.startswith("val_"):
                ax2.plot(ep, [h[k] for h in hist], color="#22d3ee", label=k)
                break
        ax.set(xlabel="epoch", title=title)
        fig.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)


# ----------------------------------------------------------------------------- chest ensemble
@torch.no_grad()
def _logits_and_feats(model: torch.nn.Module, loader: DataLoader, device: str
                      ) -> tuple[np.ndarray, np.ndarray]:
    outs, ys = [], []
    for x, y in loader:
        outs.append(model(x.to(device)).float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(outs), np.concatenate(ys)


@torch.no_grad()
def mc_dropout_var(model: torch.nn.Module, loader: DataLoader, device: str, samples: int = 8,
                   limit: int = 512) -> float:
    """Mean per-label variance of sigmoid outputs under MC-dropout on the head."""
    model.eval()
    for m in model.head.modules():  # type: ignore[union-attr]
        if isinstance(m, torch.nn.Dropout):
            m.train()
    vs, seen = [], 0
    for x, _ in loader:
        f = model.forward_features(x.to(device))  # type: ignore[operator]
        ps = torch.stack([torch.sigmoid(model.head(f)) for _ in range(samples)])  # type: ignore[operator]
        vs.append(ps.var(0).mean().item())
        seen += len(x)
        if seen >= limit:
            break
    model.eval()
    return float(np.mean(vs))


def joint_energy(logits: np.ndarray) -> np.ndarray:
    """Multi-label JointEnergy score (Wang et al. 2021): higher = more in-distribution."""
    return np.sum(np.log1p(np.exp(np.clip(logits, -30, 30))), axis=1)


def evaluate_chest_ensemble(profile: str) -> dict[str, Any]:
    from ml.data.datasets import NIHDataset
    from ml.train import make_datasets

    device = get_device()
    out = artifact_dir("chest_ensemble", profile)
    plots = out / "plots"
    plots.mkdir(exist_ok=True)
    members = [m for m in CHEST_MEMBERS if (ARTIFACTS / m / profile / "model.pt").exists()]
    if not members:
        raise FileNotFoundError("no chest members trained")
    val_l, test_l, nih_l, ood_l = [], [], [], []
    temps, per_member, mc_vars = {}, {}, {}
    y_val = y_test = y_nih = None
    nih_meta = None
    for m in members:
        model, cfg = load_trained(m, profile, device)
        cfg_eval = dict(cfg)
        cfg_eval["train"] = {**cfg["train"], "limit_val": 6000}
        _, vds, tds = make_datasets(cfg_eval)
        vl = DataLoader(vds, 128)
        tl = DataLoader(tds, 128)
        lv, y_val = _logits_and_feats(model, vl, device)
        lt, y_test = _logits_and_feats(model, tl, device)
        t = [fit_temperature(lv[:, k], y_val[:, k]) for k in range(len(NIH_LABELS))]
        temps[m] = t
        val_l.append(sigmoid(lv / np.array(t)))
        test_l.append(lt)
        per_auc, macro = multilabel_auroc(y_test, lt)
        per_member[m] = {"macro_auroc": macro, "per_label_auroc": dict(zip(NIH_LABELS, per_auc))}
        mc_vars[m] = mc_dropout_var(model, tl, device)
        try:  # external fairness set: NIH official-test-list patients with age/sex metadata
            nds = NIHDataset("test", cfg["img_size"], source="sample")
            ln, y_nih = _logits_and_feats(model, DataLoader(nds, 128), device)
            nih_l.append(ln)
            nih_meta = nds.meta
        except FileNotFoundError:
            pass
        ood_l.append(_ood_logits(model, cfg["img_size"], device))
        del model
    temps_arr = {m: np.array(temps[m]) for m in members}
    p_val = np.mean(val_l, 0)
    p_test = np.mean([sigmoid(lt / temps_arr[m]) for lt, m in zip(test_l, members)], 0)
    std_test = np.std([sigmoid(lt / temps_arr[m]) for lt, m in zip(test_l, members)], 0)
    thresholds = [youden_threshold(y_val[:, k], p_val[:, k]) for k in range(len(NIH_LABELS))]
    per_auc, macro = multilabel_auroc(y_test, p_test)
    per_label = {}
    for k, lab in enumerate(NIH_LABELS):
        se, sp = sens_spec(y_test[:, k], p_test[:, k], thresholds[k])
        per_label[lab] = {"auroc": per_auc[k], "threshold": thresholds[k], "sensitivity": se,
                          "specificity": sp, "prevalence": float(y_test[:, k].mean()),
                          "roc": roc_points(y_test[:, k], p_test[:, k], 40)}
    uncal = np.mean([sigmoid(lt) for lt in test_l], 0)
    rel_cal = reliability(y_test.ravel(), p_test.ravel(), 10)
    rel_uncal = reliability(y_test.ravel(), uncal.ravel(), 10)
    # energy OOD threshold on in-distribution validation logits (5th percentile)
    e_val = np.mean([joint_energy(_logit(v)) for v in val_l], 0)
    e_test = np.mean([joint_energy(lt) for lt in test_l], 0)
    e_ood = np.mean([joint_energy(o) for o in ood_l], 0)
    e_thr = float(np.percentile(e_val, 5))
    ood_auc = safe_auroc(np.r_[np.ones(len(e_test)), np.zeros(len(e_ood))], np.r_[e_test, e_ood])
    fairness = {}
    if nih_l and nih_meta is not None and y_nih is not None:
        p_nih = np.mean([sigmoid(ln / temps_arr[m]) for ln, m in zip(nih_l, members)], 0)
        fairness = fairness_report(y_nih, p_nih, nih_meta)
    metrics = {
        "model": "chest_ensemble", "profile": profile, "task": "multilabel",
        "generated_at": _now(), "members": members, "labels": NIH_LABELS,
        "dataset": "ChestMNIST+ (NIH ChestX-ray14, official patient-wise split)"
        if profile == "starter" else "NIH ChestX-ray14 (patient-level split)",
        "n_test": int(len(y_test)),
        "test": {"macro_auroc": macro, "per_label": per_label,
                 "reliability_calibrated": rel_cal, "reliability_uncalibrated": rel_uncal,
                 "mean_ensemble_std": float(std_test.mean()), "mc_dropout_variance": mc_vars,
                 "ood_energy_auroc": ood_auc},
        "per_member": per_member, "fairness": fairness,
        "history": {m: _history(m, profile) for m in members},
    }
    calib = {"labels": NIH_LABELS, "members": members, "temperatures": temps,
             "thresholds": dict(zip(NIH_LABELS, thresholds)), "energy_threshold": e_thr,
             "energy_val_percentiles": {p: float(np.percentile(e_val, p)) for p in (1, 5, 25, 50)}}
    write_json(out / "calibration.json", calib)
    write_json(out / "metrics.json", metrics)
    plot_roc({k: v["roc"] for k, v in per_label.items()}, {k: v["auroc"] for k, v in per_label.items()},
             plots / "roc.png", f"Chest ensemble ROC — macro AUROC {macro:.3f}")
    plot_reliability(rel_cal, plots / "reliability.png", "Calibrated ensemble")
    for m in members:
        plot_history(_history(m, profile), plots / f"history_{m}.png", m)
    for m in members:  # copy member summaries next to each member too
        write_json(ARTIFACTS / m / profile / "metrics.json",
                   {"model": m, "profile": profile, "task": "multilabel", "generated_at": _now(),
                    "test": per_member[m], "history": _history(m, profile),
                    "mc_dropout_variance": mc_vars[m]})
    print(f"[eval] chest_ensemble/{profile}: macro AUROC {macro:.4f} (members: "
          + ", ".join(f"{m}={per_member[m]['macro_auroc']:.3f}" for m in members) + ")")
    return metrics


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@torch.no_grad()
def _ood_logits(model: torch.nn.Module, size: int, device: str, n: int = 400) -> np.ndarray:
    """Chest-model logits on clearly out-of-distribution inputs (CIFAR photos, brain MRI, bone X-ray)."""
    from ml.data.datasets import BrainMRIDataset, FractureClsDataset
    from ml.data.transforms import preprocess

    imgs: list[np.ndarray] = []
    try:
        import torchvision

        from ml.common import DATA_DIR

        c = torchvision.datasets.CIFAR10(str(DATA_DIR / "raw" / "cifar10"), train=False)
        imgs += [c.data[i] for i in range(n // 2)]
    except Exception:
        pass
    try:
        b = BrainMRIDataset("test", 128)
        imgs += [b.x[i] for i in range(min(n // 4, len(b)))]
        f = FractureClsDataset("test")
        imgs += [f.image(i) for i in range(min(n // 4, len(f)))]
    except Exception:
        pass
    if not imgs:
        return np.zeros((1, len(NIH_LABELS)))
    x = torch.from_numpy(np.concatenate([preprocess(i, size) for i in imgs]))
    return np.concatenate([model(x[i : i + 128].to(device)).cpu().numpy() for i in range(0, len(x), 128)])


def fairness_report(y: np.ndarray, p: np.ndarray, meta: Any) -> dict[str, Any]:
    groups: dict[str, np.ndarray] = {
        "sex:F": (meta["sex"] == "F").to_numpy(),
        "sex:M": (meta["sex"] == "M").to_numpy(),
        "age:<40": (meta["age"] < 40).to_numpy(),
        "age:40-59": ((meta["age"] >= 40) & (meta["age"] < 60)).to_numpy(),
        "age:60+": (meta["age"] >= 60).to_numpy(),
        "view:PA": (meta["view"] == "PA").to_numpy(),
        "view:AP": (meta["view"] == "AP").to_numpy(),
    }
    out: dict[str, Any] = {"dataset": "NIH ChestX-ray14 sample — official test-list patients",
                           "n": int(len(y)), "groups": {}}
    for g, m in groups.items():
        if m.sum() < 20:
            continue
        per, macro = multilabel_auroc(y[m], p[m])
        out["groups"][g] = {"n": int(m.sum()), "macro_auroc": macro,
                            "per_label_auroc": dict(zip(NIH_LABELS, per))}
    _, overall = multilabel_auroc(y, p)
    out["overall_macro_auroc"] = overall
    vals = [v["macro_auroc"] for v in out["groups"].values() if v["macro_auroc"] == v["macro_auroc"]]
    sex = [out["groups"].get(k, {}).get("macro_auroc") for k in ("sex:F", "sex:M")]
    out["max_gap"] = float(max(vals) - min(vals)) if vals else None
    out["sex_gap"] = float(abs(sex[0] - sex[1])) if all(s is not None for s in sex) else None
    return out


# ----------------------------------------------------------------------------- generic models
def evaluate_model(name: str, profile: str) -> dict[str, Any]:
    from ml.train import det_collate, make_datasets, predict_detection

    if name == "chest_ensemble":
        return evaluate_chest_ensemble(profile)
    if name == "report_gen":
        return _evaluate_report(profile)
    if name == "copilot_intent":
        p = ARTIFACTS / "copilot_intent" / profile / "metrics.json"
        return json.loads(p.read_text()) if p.exists() else {}
    device = "cpu" if load_config(name, profile)["task"] == "detection" else get_device()
    model, cfg = load_trained(name, profile, device)
    out = artifact_dir(name, profile)
    plots = out / "plots"
    plots.mkdir(exist_ok=True)
    task = cfg["task"]
    train_ds, val_ds, test_ds = make_datasets(cfg)
    res: dict[str, Any] = {"model": name, "profile": profile, "task": task, "generated_at": _now(),
                           "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
                           "history": _history(name, profile)}
    if task == "detection":
        preds, gts = predict_detection(model, DataLoader(test_ds, 8, collate_fn=det_collate), device)
        res["test"] = map50(preds, gts)
        res["test"]["map50_iou_0.3"] = map50(preds, gts, 0.3)["map50"]
        # image-level detection of "any fracture" using the max box score
        y = np.array([len(g) > 0 for g in gts]).astype(int)
        s = np.array([float(p["scores"].max()) if len(p["scores"]) else 0.0 for p in preds])
        res["test"]["image_level_auroc"] = safe_auroc(y, s)
        res["test"]["score_threshold"] = 0.3
    elif task in ("binary", "multiclass"):
        lv, yv = _logits_and_feats(model, DataLoader(val_ds, 64), device)
        lt, yt = _logits_and_feats(model, DataLoader(test_ds, 64), device)
        if task == "binary":
            t = fit_temperature(lv[:, 0], yv[:, 0])
            pv, pt = sigmoid(lv[:, 0] / t), sigmoid(lt[:, 0] / t)
            thr = youden_threshold(yv[:, 0], pv)
            se, sp = sens_spec(yt[:, 0], pt, thr)
            pred = pt >= thr
            tp = float(np.sum(pred & (yt[:, 0] == 1)))
            f1 = 2 * tp / max(pred.sum() + yt[:, 0].sum(), 1)
            roc = roc_points(yt[:, 0], pt)
            auc = safe_auroc(yt[:, 0], pt)
            rel = reliability(yt[:, 0], pt)
            res["test"] = {"auroc": auc, "threshold": thr, "sensitivity": se, "specificity": sp,
                           "f1": float(f1), "temperature": t, "roc": roc, "reliability": rel}
            plot_roc({"fracture": roc}, {"fracture": auc}, plots / "roc.png", f"{name} ROC")
            plot_reliability(rel, plots / "reliability.png", name)
        else:
            classes = BRAIN_CLASSES if cfg["dataset"] == "brain_mri" else ROUTER_CLASSES
            t = fit_temperature_multiclass(lv, yv)
            rep = classification_report(yt, lt / t, classes)
            rep["temperature"] = t
            z = lt / t
            prob = np.exp(z - z.max(1, keepdims=True))
            prob /= prob.sum(1, keepdims=True)
            rep["reliability"] = reliability((prob.argmax(1) == yt).astype(float), prob.max(1))
            res["test"] = rep
            if hasattr(test_ds, "counts"):
                res["class_counts_test"] = test_ds.counts
            plot_confusion(rep["confusion_matrix"], classes, plots / "confusion.png", name)
    elif task == "segmentation":
        lt, yt = _logits_and_feats(model, DataLoader(test_ds, 32), device)
        d, i = dice_iou(lt[:, 0] > 0, yt[:, 0] > 0.5)
        pos = yt[:, 0].reshape(len(yt), -1).max(1) > 0.5
        dp, ip = dice_iou(lt[pos, 0] > 0, yt[pos, 0] > 0.5) if pos.any() else (None, None)
        res["test"] = {"dice": d, "iou": i, "dice_positive_slices": dp, "iou_positive_slices": ip,
                       "n_positive": int(pos.sum())}
    elif task == "denoise":
        res["test"] = _evaluate_denoiser(model, test_ds, device)
    plot_history(res["history"], plots / "history.png", name)
    write_json(out / "metrics.json", res)
    print(f"[eval] {name}/{profile}: " + json.dumps(
        {k: v for k, v in res["test"].items() if isinstance(v, int | float)}, default=str)[:300])
    return res


@torch.no_grad()
def _evaluate_denoiser(model: torch.nn.Module, ds: Any, device: str, n: int = 200) -> dict[str, Any]:
    import cv2

    from ml.data.transforms import add_low_dose_noise

    torch.manual_seed(0)
    levels = {"very_low_dose (40 photons)": (40, 0.03), "low_dose (100 photons)": (100, 0.02),
              "moderate (300 photons)": (300, 0.01)}
    out: dict[str, Any] = {"levels": {}}
    for lname, (ph, sg) in levels.items():
        r: dict[str, list[float]] = {k: [] for k in ("psnr_noisy", "psnr_denoised", "ssim_noisy",
                                                     "ssim_denoised", "psnr_gaussian_blur")}
        for i in range(min(n, len(ds))):
            c = ds[i][None]
            nz = add_low_dose_noise(c, ph, sg)
            d = model(nz.to(device)).cpu().numpy()[0, 0]
            c0, n0 = c.numpy()[0, 0], nz.numpy()[0, 0]
            r["psnr_noisy"].append(psnr(c0, n0))
            r["psnr_denoised"].append(psnr(c0, d))
            r["ssim_noisy"].append(ssim(c0, n0))
            r["ssim_denoised"].append(ssim(c0, d))
            r["psnr_gaussian_blur"].append(psnr(c0, cv2.GaussianBlur(n0, (5, 5), 1.0)))
        out["levels"][lname] = {k: float(np.mean(v)) for k, v in r.items()}
    mid = out["levels"]["low_dose (100 photons)"]
    out.update({"psnr_noisy": mid["psnr_noisy"], "psnr_denoised": mid["psnr_denoised"],
                "ssim_noisy": mid["ssim_noisy"], "ssim_denoised": mid["ssim_denoised"]})
    return out


def _evaluate_report(profile: str) -> dict[str, Any]:
    p = ARTIFACTS / "report_gen" / profile / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


# ----------------------------------------------------------------------------- model cards
def write_card(name: str, profile: str) -> None:
    d = ARTIFACTS / name / profile
    mp = d / "metrics.json"
    if not mp.exists():
        return
    m = json.loads(mp.read_text())
    cfgp = d / "config.json"
    cfg = json.loads(cfgp.read_text()) if cfgp.exists() else {}
    t = m.get("test", {})
    headline = {k: v for k, v in t.items() if isinstance(v, int | float)}
    lines = [f"## {name} ({profile})", "",
             f"- Task: {m.get('task')}  ·  dataset: {cfg.get('dataset', m.get('dataset', 'n/a'))}",
             f"- Architecture: {cfg.get('model', {})}",
             f"- Input size: {cfg.get('img_size', 'n/a')}  ·  epochs run: {len(m.get('history', []) or [])}",
             f"- Test metrics: {json.dumps(headline, default=lambda x: round(x, 4))}",
             f"- Generated: {m.get('generated_at', '')}", ""]
    (d / "card.md").write_text("\n".join(lines))


ALL = ["chest_ensemble", "lung_unet", "fracture_cls", "fracture_det", "brain_cls", "brain_seg",
       "router", "denoiser", "report_gen"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--profile", default="starter")
    args = ap.parse_args(argv)
    names = ALL if args.all else [args.model]
    for n in names:
        try:
            evaluate_model(n, args.profile)
            write_card(n, args.profile)
        except FileNotFoundError as exc:
            print(f"[eval] {n}/{args.profile}: skipped ({exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
