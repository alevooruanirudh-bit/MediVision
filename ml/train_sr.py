"""Train the x2 super-resolution GAN (python -m ml.train sr_gan --profile starter).

LR images are made by area-downsampling HR chest X-rays (ChestMNIST+ 128 px, official split).
Stage 1: L1 pre-training of the generator. Stage 2: L1 + light LSGAN adversarial loss.
Metrics: PSNR / SSIM of bicubic vs ours on the test split (the GAN can trade a little PSNR for
sharper texture; both numbers are reported honestly).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ml.common import artifact_dir, get_device, seed_everything, write_json
from ml.data.datasets import load_medmnist
from ml.models.srgan import PatchDiscriminator, SRGenerator


def _hr(split: str, size: int, limit: int | None) -> torch.Tensor:
    x, _ = load_medmnist("chestmnist", 128, split)
    if limit and limit < len(x):
        x = x[np.random.default_rng(0).choice(len(x), limit, replace=False)]
    t = torch.from_numpy(np.asarray(x)).float()[:, None] / 255.0
    if size != 128:
        t = F.interpolate(t, size=(size, size), mode="area")
    return t


def lowres(hr: torch.Tensor, scale: int = 2) -> torch.Tensor:
    return F.interpolate(hr, scale_factor=1 / scale, mode="area")


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b, reduction="none").mean((1, 2, 3)).clamp_min(1e-10)
    return float((10 * torch.log10(1 / mse)).mean())


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    from skimage.metrics import structural_similarity

    return float(
        np.mean(
            [
                structural_similarity(x[0].numpy(), y[0].numpy(), data_range=1.0)
                for x, y in zip(a, b)
            ]
        )
    )


@torch.no_grad()
def evaluate(g: SRGenerator, hr: torch.Tensor, device: str, scale: int = 2) -> dict[str, float]:
    g.eval()
    outs, bics = [], []
    for i in range(0, len(hr), 64):
        h = hr[i : i + 64]
        lr = lowres(h, scale)
        bics.append(F.interpolate(lr, scale_factor=scale, mode="bicubic").clamp(0, 1))
        outs.append(g(lr.to(device)).cpu())
    o, b = torch.cat(outs), torch.cat(bics)
    return {
        "psnr_bicubic": psnr(b, hr),
        "psnr_ours": psnr(o, hr),
        "ssim_bicubic": ssim(b[:200], hr[:200]),
        "ssim_ours": ssim(o[:200], hr[:200]),
    }


def train_sr(cfg: dict[str, Any]) -> dict[str, Any]:
    seed_everything(42)
    device = get_device()
    t = cfg["train"]
    m = cfg["model"]
    size, scale = cfg["img_size"] * m.get("scale", 2), m.get("scale", 2)
    tr = _hr("train", size, t.get("limit_train"))
    va = _hr("val", size, t.get("limit_val"))
    te = _hr("test", size, 1000)
    g = SRGenerator(m.get("width", 48), m.get("blocks", 6), scale).to(device)
    d = PatchDiscriminator().to(device)
    og = torch.optim.Adam(g.parameters(), cfg["optim"]["lr"], betas=(0.9, 0.99))
    od = torch.optim.Adam(d.parameters(), cfg["optim"]["lr"], betas=(0.9, 0.99))
    lw = cfg.get("loss", {})
    bs = t.get("batch_size", 32)
    history, t0 = [], time.time()
    pre = t.get("pretrain_epochs", 2)
    for ep in range(1, t["epochs"] + pre + 1):
        adv = ep > pre
        g.train()
        d.train()
        perm = torch.randperm(len(tr))
        gl = dl = 0.0
        for i in range(0, len(tr), bs):
            hr = tr[perm[i : i + bs]]
            if torch.rand(1) < 0.5:
                hr = hr.flip(3)
            hr = hr.to(device)
            lr = lowres(hr, scale)
            sr = g(lr)
            if adv:
                od.zero_grad()
                l_d = 0.5 * ((d(hr) - 1).pow(2).mean() + d(sr.detach()).pow(2).mean())  # LSGAN
                l_d.backward()
                od.step()
                dl += l_d.item() * len(hr)
            og.zero_grad()
            loss = lw.get("l1", 1.0) * F.l1_loss(sr, hr)
            if adv:
                loss = loss + lw.get("adv", 0.003) * (d(sr) - 1).pow(2).mean()
            loss.backward()
            og.step()
            gl += loss.item() * len(hr)
        ev = evaluate(g, va, device, scale)
        row = {
            "epoch": ep,
            "stage": "adversarial" if adv else "l1_pretrain",
            "train_loss": gl / len(tr),
            "d_loss": dl / len(tr) if adv else None,
            "val_psnr": ev["psnr_ours"],
            "val_psnr_bicubic": ev["psnr_bicubic"],
            "seconds": round(time.time() - t0, 1),
        }
        history.append(row)
        print(f"[sr_gan] {row}", flush=True)
    out = artifact_dir(cfg["name"], cfg["profile"])
    g = g.cpu().eval()
    torch.save(
        {"state_dict": g.state_dict(), "config": cfg, "labels": []},
        out / "model.pt",
    )
    write_json(out / "history.json", history)
    write_json(out / "config.json", cfg)
    test = evaluate(g, te, "cpu", scale)
    res = {
        "model": cfg["name"],
        "profile": cfg["profile"],
        "task": "superres",
        "n_train": len(tr),
        "n_val": len(va),
        "n_test": len(te),
        "history": history,
        "test": test,
        "note": "LR = area-downsampled HR (x2). GAN outputs can contain invented texture; "
        "never use super-resolved pixels for diagnosis.",
    }
    write_json(out / "metrics.json", res)
    write_json(out / "train_summary.json", {"model": cfg["name"], "best": test})
    print(f"[sr_gan] test {test}")
    return res
