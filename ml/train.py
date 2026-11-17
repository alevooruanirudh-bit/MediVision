"""One config-driven trainer for every MediVision model.

    python -m ml.train --model chest_densenet121 --profile starter [--resume]

Features: AMP (CUDA), cosine LR with warm-up, early stopping on a monitored validation metric,
class-imbalance-aware losses, best/last checkpointing, resume, deterministic seeding, per-epoch
history (for the UI's training curves). Writes artifacts/<model>/<profile>/model.pt.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ml.common import (
    ARTIFACTS,
    artifact_dir,
    get_device,
    load_config,
    pretrained_backbone,
    seed_everything,
    write_json,
)
from ml.data.transforms import Augment, add_low_dose_noise
from ml.losses import build_loss
from ml.metrics import classification_report, dice_iou, map50, multilabel_auroc, psnr, safe_auroc
from ml.models import build_model, labels_for


# ----------------------------------------------------------------------------- datasets
def make_augment(cfg: dict[str, Any], train: bool) -> Augment:
    a = cfg.get("augment", {})
    return Augment(
        cfg["img_size"],
        scale=tuple(a.get("scale", (0.85, 1.0))),  # type: ignore[arg-type]
        rotate=a.get("rotate", 7),
        brightness=a.get("brightness", 0.1),
        contrast=a.get("contrast", 0.1),
        clahe_p=a.get("clahe_p", 0.0),
        hflip=a.get("hflip", False),
        train=train,
    )


def make_datasets(cfg: dict[str, Any], tokenizer: Any = None) -> tuple[Dataset, Dataset, Dataset]:
    from ml.data import datasets as D

    t = cfg.get("train", {})
    size, name = cfg["img_size"], cfg["dataset"]
    aug_tr, aug_ev = make_augment(cfg, True), make_augment(cfg, False)
    lt, lv = t.get("limit_train"), t.get("limit_val")
    if name == "chestmnist":
        return (
            D.ChestMNISTDataset("train", size, aug_tr, lt),
            D.ChestMNISTDataset("val", size, aug_ev, lv),
            D.ChestMNISTDataset("test", size, aug_ev),
        )
    if name == "nih":
        return (
            D.NIHDataset("train", size, aug_tr, "full", limit=lt),
            D.NIHDataset("val", size, aug_ev, "full", limit=lv),
            D.NIHDataset("test", size, aug_ev, "full"),
        )
    if name in ("lung", "lgg"):
        return (
            D.SegDataset(name, "train", size, aug_tr),
            D.SegDataset(name, "val", size, aug_ev),
            D.SegDataset(name, "test", size, aug_ev),
        )
    if name == "fracatlas_cls":
        return (
            D.FractureClsDataset("train", size, aug_tr),
            D.FractureClsDataset("val", size, aug_ev),
            D.FractureClsDataset("test", size, aug_ev),
        )
    if name == "fracatlas_det":
        nr = t.get("neg_ratio", 0.3)
        return (
            D.FractureDetDataset("train", size, True, nr),
            D.FractureDetDataset("val", size),
            D.FractureDetDataset("test", size),
        )
    if name == "rsna":
        return (
            D.DetDataset("train", size, True, 1.0, key="rsna"),
            D.DetDataset("val", size, key="rsna", limit=lv),
            D.DetDataset("test", size, key="rsna"),
        )
    if name == "brain_mri":
        return (
            D.BrainMRIDataset("train", size, aug_tr),
            D.BrainMRIDataset("val", size, aug_ev),
            D.BrainMRIDataset("test", size, aug_ev),
        )
    if name == "router":
        pc = cfg.get("per_class", 1500)
        return (
            D.RouterDataset("train", size, pc),
            D.RouterDataset("val", size, pc),
            D.RouterDataset("test", size, pc),
        )
    if name == "denoise":
        return (
            D.DenoiseDataset("train", size, lt),
            D.DenoiseDataset("val", size, lv),
            D.DenoiseDataset("test", size, lv),
        )
    if name == "iu_xray":
        ml = cfg["model"]["max_len"] + 2
        return (
            D.IUReportDataset("train", tokenizer, size, ml, aug_tr),
            D.IUReportDataset("val", tokenizer, size, ml, aug_ev),
            D.IUReportDataset("test", tokenizer, size, ml, aug_ev),
        )
    raise ValueError(f"unknown dataset {name}")


# ----------------------------------------------------------------------------- helpers
def cosine_with_warmup(
    opt: torch.optim.Optimizer, warmup: int, total: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        prog = (step - warmup) / max(total - warmup, 1)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    outs, ys = [], []
    for x, y in loader:
        outs.append(model(x.to(device)).float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(outs), np.concatenate(ys)


def validate(task: str, model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    if task == "denoise":
        model.eval()
        gen = torch.Generator().manual_seed(0)
        vals, base = [], []
        with torch.no_grad():
            for clean in loader:
                torch.manual_seed(int(torch.randint(0, 10_000, (1,), generator=gen)))
                noisy = add_low_dose_noise(clean, photons=80, sigma=0.02)
                den = model(noisy.to(device)).cpu().numpy()
                for c, n, d in zip(clean.numpy(), noisy.numpy(), den):
                    vals.append(psnr(c[0], d[0]))
                    base.append(psnr(c[0], n[0]))
        return {"val_psnr": float(np.mean(vals)), "val_psnr_noisy": float(np.mean(base))}
    logits, y = predict(model, loader, device)
    if task == "multilabel":
        _, macro = multilabel_auroc(y, logits)
        return {"val_macro_auroc": macro}
    if task == "binary":
        return {"val_auroc": safe_auroc(y[:, 0], logits[:, 0]) or 0.0}
    if task == "multiclass":
        rep = classification_report(y, logits, [str(i) for i in range(logits.shape[1])])
        return {"val_macro_f1": rep["macro_f1"], "val_accuracy": rep["accuracy"]}
    if task == "segmentation":
        d, i = dice_iou(logits[:, 0] > 0, y[:, 0] > 0.5)
        return {"val_dice": d, "val_iou": i}
    raise ValueError(task)


def class_balance(
    cfg: dict[str, Any], ds: Dataset
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    y = getattr(ds, "y", None)
    if y is None:
        return None, None
    y = np.asarray(y)
    if cfg["task"] in ("multilabel", "binary"):
        y2 = y.reshape(len(y), -1)
        pos = y2.mean(0).clip(1e-3, 1)
        return torch.tensor(((1 - pos) / pos).clip(1, 30), dtype=torch.float32), None
    if cfg["task"] == "multiclass":
        cnt = np.bincount(y, minlength=int(y.max()) + 1).astype(np.float32)
        w = cnt.sum() / (len(cnt) * np.maximum(cnt, 1))
        return None, torch.tensor(w, dtype=torch.float32)
    return None, None


class EarlyStopper:
    def __init__(self, patience: int, mode: str = "max") -> None:
        self.patience, self.mode = patience, mode
        self.best = -math.inf if mode == "max" else math.inf
        self.bad = 0

    def step(self, value: float) -> bool:
        """Returns True when `value` is a new best."""
        better = value > self.best if self.mode == "max" else value < self.best
        if better:
            self.best, self.bad = value, 0
            return True
        self.bad += 1
        return False

    @property
    def stop(self) -> bool:
        return self.bad > self.patience


def _log(msg: str) -> None:
    print(f"[train] {msg}", flush=True)


# ----------------------------------------------------------------------------- main loop
def train(
    name: str,
    profile: str = "starter",
    resume: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = load_config(name, profile)
    if overrides:
        cfg.update(overrides)
    seed_everything(cfg.get("seed", 42))
    task = cfg["task"]
    if task == "detection":
        return train_detection(cfg, resume)
    if task == "report":
        from ml.train_report import train_report

        return train_report(cfg, resume)

    t = cfg["train"]
    device = get_device(t.get("device", "auto"))
    out = artifact_dir(name, profile)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    train_ds, val_ds, _ = make_datasets(cfg)
    bs, nw = t["batch_size"], t.get("num_workers", 0)
    tl = DataLoader(
        train_ds, bs, shuffle=True, num_workers=nw, drop_last=True, persistent_workers=nw > 0
    )
    vl = DataLoader(val_ds, bs * 2, shuffle=False, num_workers=nw)
    model = build_model(cfg).to(device)
    pos_w, cls_w = class_balance(cfg, train_ds)
    if task == "denoise":
        loss_fn: nn.Module = nn.L1Loss()
    else:
        loss_fn = build_loss(cfg, pos_w, cls_w)
    loss_fn = loss_fn.to(device)
    o = cfg["optim"]
    opt = torch.optim.AdamW(model.parameters(), lr=o["lr"], weight_decay=o.get("weight_decay", 0))
    epochs = t["epochs"]
    steps = epochs * len(tl)
    sched = cosine_with_warmup(opt, t.get("warmup_epochs", 0) * len(tl), steps)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    monitor = t.get("monitor", "val_loss")
    stopper = EarlyStopper(t.get("patience", 3), "min" if monitor.endswith("loss") else "max")
    history: list[dict[str, Any]] = []
    start_epoch = 0
    if resume and (ckpt_dir / "last.pt").exists():
        st = torch.load(ckpt_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start_epoch, history = st["epoch"] + 1, st["history"]
        stopper.best = st["best"]
        _log(f"resumed {name} at epoch {start_epoch}")
    _log(
        f"{name}/{profile} on {device}: {len(train_ds)} train / {len(val_ds)} val, "
        f"pretrained_backbone={pretrained_backbone()}"
    )
    for epoch in range(start_epoch, epochs):
        model.train()
        t0, running, n = time.time(), 0.0, 0
        for batch in tl:
            if task == "denoise":
                clean = batch.to(device)
                x, y = add_low_dose_noise(batch).to(device), clean
            else:
                x, y = batch[0].to(device), batch[1].to(device)
            with torch.autocast("cuda", enabled=use_amp):
                pred = model(x)
                loss = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item() * len(x)
            n += len(x)
        metrics = validate(task, model, vl, device)
        rec = {
            "epoch": epoch + 1,
            "train_loss": running / max(n, 1),
            **metrics,
            "lr": opt.param_groups[0]["lr"],
            "seconds": round(time.time() - t0, 1),
        }
        history.append(rec)
        is_best = stopper.step(metrics[monitor])
        _log(
            f"{name} epoch {epoch + 1}/{epochs} "
            + " ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()
            )
        )
        if is_best:
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "epoch": epoch,
                "history": history,
                "best": stopper.best,
            },
            ckpt_dir / "last.pt",
        )
        if stopper.stop:
            _log(f"early stopping at epoch {epoch + 1}")
            break
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device))
    return finalize(cfg, model, history, {monitor: stopper.best})


def finalize(
    cfg: dict[str, Any], model: nn.Module, history: list[dict[str, Any]], best: dict[str, Any]
) -> dict[str, Any]:
    out = artifact_dir(cfg["name"], cfg["profile"])
    torch.save(
        {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "config": cfg,
            "labels": labels_for(cfg),
        },
        out / "model.pt",
    )
    write_json(out / "history.json", history)
    write_json(out / "config.json", cfg)
    summary = {
        "model": cfg["name"],
        "profile": cfg["profile"],
        "best": best,
        "epochs_run": len(history),
        "pretrained_backbone": pretrained_backbone(),
    }
    write_json(out / "train_summary.json", summary)
    _log(f"saved {out / 'model.pt'}")
    return summary


# ----------------------------------------------------------------------------- detection
def det_collate(batch: list[Any]) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    return [b[0] for b in batch], [b[1] for b in batch]


@torch.no_grad()
def predict_detection(
    model: nn.Module, loader: DataLoader, device: str
) -> tuple[list[dict[str, np.ndarray]], list[np.ndarray]]:
    model.eval()
    preds, gts = [], []
    for xs, ts in loader:
        outs = model([x.to(device) for x in xs])
        for o, t in zip(outs, ts):
            preds.append({"boxes": o["boxes"].cpu().numpy(), "scores": o["scores"].cpu().numpy()})
            gts.append(t["boxes"].numpy())
    return preds, gts


def train_detection(cfg: dict[str, Any], resume: bool = False) -> dict[str, Any]:
    t = cfg["train"]
    device = get_device(t.get("device", "cpu"))
    torch.set_num_threads(max(torch.get_num_threads(), 8))
    train_ds, val_ds, _ = make_datasets(cfg)
    tl = DataLoader(
        train_ds,
        t["batch_size"],
        shuffle=True,
        collate_fn=det_collate,
        num_workers=t.get("num_workers", 0),
    )
    vl = DataLoader(
        val_ds, t["batch_size"], collate_fn=det_collate, num_workers=t.get("num_workers", 0)
    )
    model = build_model(cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    o = cfg["optim"]
    opt = torch.optim.SGD(
        params,
        lr=o["lr"],
        momentum=o.get("momentum", 0.9),
        weight_decay=o.get("weight_decay", 1e-4),
    )
    epochs = t["epochs"]
    sched = cosine_with_warmup(opt, t.get("warmup_epochs", 1) * len(tl), epochs * len(tl))
    stopper = EarlyStopper(t.get("patience", 3))
    out = artifact_dir(cfg["name"], cfg["profile"])
    (out / "checkpoints").mkdir(exist_ok=True)
    history: list[dict[str, Any]] = []
    _log(f"{cfg['name']} detection on {device}: {len(train_ds)} train / {len(val_ds)} val")
    for epoch in range(epochs):
        model.train()
        t0, running = time.time(), 0.0
        for xs, ts in tl:
            xs = [x.to(device) for x in xs]
            ts = [{k: v.to(device) for k, v in tt.items()} for tt in ts]
            losses = model(xs, ts)
            loss = sum(losses.values())
            opt.zero_grad()
            loss.backward()  # type: ignore[union-attr]
            nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            sched.step()
            running += loss.item()  # type: ignore[union-attr]
        preds, gts = predict_detection(model, vl, device)
        m = map50(preds, gts)
        rec = {
            "epoch": epoch + 1,
            "train_loss": running / len(tl),
            "val_map50": m["map50"],
            "val_recall": m["recall"],
            "seconds": round(time.time() - t0, 1),
        }
        history.append(rec)
        _log(f"{cfg['name']} epoch {epoch + 1}/{epochs} {rec}")
        if stopper.step(m["map50"]):
            torch.save(model.state_dict(), out / "checkpoints" / "best.pt")
        if stopper.stop:
            break
    model.load_state_dict(torch.load(out / "checkpoints" / "best.pt", map_location=device))
    return finalize(cfg, model, history, {"val_map50": stopper.best})


TRAINERS: dict[str, Callable[..., dict[str, Any]]] = {"default": train}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--profile", default="starter")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    train(args.model, args.profile, args.resume)
    _ = ARTIFACTS
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
