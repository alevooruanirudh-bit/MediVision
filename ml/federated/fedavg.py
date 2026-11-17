"""Real Federated Averaging (McMahan et al. 2017) across 3 simulated hospitals on PneumoniaMNIST 64 px.

Non-IID by design: each hospital has a different label mix, size, and scanner "look" (contrast /
brightness / noise shift). Compares FedAvg vs centralised (pooled data — not allowed in practice) vs
local-only (each hospital alone). Runs on CPU in well under two minutes; events are yielded so the
backend can stream them over SSE.

    python -m ml.federated.fedavg
"""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from ml.common import ARTIFACTS, write_json


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = F.relu(self.c3(x))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


@dataclass
class Hospital:
    name: str
    x: torch.Tensor
    y: torch.Tensor
    style: str

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n": int(len(self.y)),
            "pneumonia_rate": round(float(self.y.mean()), 3),
            "scanner": self.style,
        }


def _load(size: int = 64) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    from ml.data.datasets import load_medmnist

    return {s: load_medmnist("pneumoniamnist", size, s) for s in ("train", "val", "test")}


def _style(
    x: np.ndarray, contrast: float, bright: float, noise: float, rng: np.random.Generator
) -> np.ndarray:
    x = x.astype(np.float32) / 255.0
    x = np.clip((x - 0.5) * contrast + 0.5 + bright + rng.normal(0, noise, x.shape), 0, 1)
    return x.astype(np.float32)


def make_hospitals(seed: int = 0) -> tuple[list[Hospital], torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    data = _load()
    x, y = data["train"]
    y = y[:, 0]
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    # (name, n_pos, n_neg, contrast, brightness, noise, description)
    plan = [
        ("St. Aurora (urban referral)", 1600, 250, 1.25, 0.03, 0.01, "high contrast"),
        ("Riverside General", 700, 700, 1.0, 0.0, 0.02, "standard"),
        ("Hillcrest Rural Clinic", 120, 450, 0.75, -0.06, 0.05, "low contrast, noisy"),
    ]
    hospitals, ip, ineg = [], 0, 0
    for name, npos, nneg, c, b, nz, style in plan:
        idx = np.r_[pos[ip : ip + npos], neg[ineg : ineg + nneg]]
        ip, ineg = ip + npos, ineg + nneg
        xs = _style(x[idx], c, b, nz, rng)
        hospitals.append(
            Hospital(
                name,
                torch.from_numpy(xs)[:, None],
                torch.from_numpy(y[idx].astype(np.float32)),
                style,
            )
        )
    xt, yt = data["test"]
    # The global test set mixes all three scanner styles equally.
    parts = np.array_split(np.arange(len(yt)), 3)
    xt_s = np.concatenate(
        [_style(xt[p], c, b, nz, rng) for p, (_, _, _, c, b, nz, _) in zip(parts, plan)]
    )
    return hospitals, torch.from_numpy(xt_s)[:, None], torch.from_numpy(yt[:, 0].astype(np.float32))


def _train_local(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    lr: float,
    bs: int = 64,
    seed: int = 0,
) -> float:
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    last = 0.0
    for _ in range(epochs):
        perm = torch.randperm(len(y), generator=g)
        for i in range(0, len(y), bs):
            b = perm[i : i + bs]
            loss = F.binary_cross_entropy_with_logits(model(x[b])[:, 0], y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = loss.item()
    return last


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    model.eval()
    p = torch.sigmoid(model(x)[:, 0]).numpy()
    yy = y.numpy()
    return {"accuracy": float(((p >= 0.5) == yy).mean()), "auroc": float(roc_auc_score(yy, p))}


def fedavg(global_model: nn.Module, locals_: list[nn.Module], weights: list[float]) -> None:
    gs = global_model.state_dict()
    tot = sum(weights)
    for k in gs:
        gs[k] = sum(m.state_dict()[k] * (w / tot) for m, w in zip(locals_, weights))
    global_model.load_state_dict(gs)


def run(
    rounds: int = 12, local_epochs: int = 1, lr: float = 2e-3, seed: int = 0
) -> Iterator[dict[str, Any]]:
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    t0 = time.time()
    hospitals, xt, yt = make_hospitals(seed)
    yield {
        "type": "setup",
        "hospitals": [h.summary for h in hospitals],
        "rounds": rounds,
        "local_epochs": local_epochs,
        "n_test": int(len(yt)),
        "algorithm": "FedAvg",
    }
    global_model = TinyCNN()
    fed_curve = []
    for r in range(1, rounds + 1):
        locals_, per = [], []
        for i, h in enumerate(hospitals):
            m = copy.deepcopy(global_model)
            loss = _train_local(m, h.x, h.y, local_epochs, lr, seed=seed * 100 + r * 10 + i)
            locals_.append(m)
            per.append(
                {
                    "name": h.name,
                    "loss": round(loss, 4),
                    **{k: round(v, 4) for k, v in evaluate(m, xt, yt).items()},
                }
            )
        fedavg(global_model, locals_, [len(h.y) for h in hospitals])
        g = evaluate(global_model, xt, yt)
        fed_curve.append(g)
        yield {
            "type": "round",
            "round": r,
            "hospitals": per,
            "global": {k: round(v, 4) for k, v in g.items()},
            "elapsed_s": round(time.time() - t0, 1),
        }
    # Baselines with the same total number of local epochs.
    xs = torch.cat([h.x for h in hospitals])
    ys = torch.cat([h.y for h in hospitals])
    cen = TinyCNN()
    cen_curve = []
    for r in range(1, rounds + 1):
        _train_local(cen, xs, ys, local_epochs, lr, seed=seed + r)
        cen_curve.append(evaluate(cen, xt, yt))
    yield {
        "type": "baseline",
        "name": "centralized",
        "curve": cen_curve,
        "elapsed_s": round(time.time() - t0, 1),
    }
    local_only = {}
    for i, h in enumerate(hospitals):
        m = TinyCNN()
        _train_local(m, h.x, h.y, rounds * local_epochs, lr, seed=seed + 50 + i)
        local_only[h.name] = evaluate(m, xt, yt)
    yield {
        "type": "baseline",
        "name": "local_only",
        "results": local_only,
        "elapsed_s": round(time.time() - t0, 1),
    }
    summary = {
        "federated": fed_curve[-1],
        "centralized": cen_curve[-1],
        "local_only": local_only,
        "local_only_mean_auroc": float(np.mean([v["auroc"] for v in local_only.values()])),
        "federated_curve": fed_curve,
        "centralized_curve": cen_curve,
        "hospitals": [h.summary for h in hospitals],
        "rounds": rounds,
        "seconds": round(time.time() - t0, 1),
    }
    write_json(ARTIFACTS / "federated" / "last_run.json", summary)
    yield {"type": "done", "summary": summary}


if __name__ == "__main__":
    for ev in run():
        if ev["type"] == "round":
            print(f"round {ev['round']:2d} global {ev['global']}  ({ev['elapsed_s']}s)")
        elif ev["type"] == "done":
            s = ev["summary"]
            print(
                json.dumps(
                    {k: s[k] for k in ("federated", "centralized", "local_only", "seconds")},
                    indent=1,
                )
            )
