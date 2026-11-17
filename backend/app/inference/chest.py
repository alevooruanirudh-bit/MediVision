"""Chest X-ray ensemble inference: calibrated probabilities, ensemble + MC-dropout uncertainty,
operating thresholds, certainty wording and the energy-score OOD guard."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.inference.registry import ModelHandle, registry
from ml.common import NIH_LABELS
from ml.data.transforms import preprocess

DISPLAY = {"Pleural_Thickening": "Pleural Thickening"}


def display_name(label: str) -> str:
    return DISPLAY.get(label, label)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def gelu(x: np.ndarray) -> np.ndarray:
    from scipy.special import erf

    return 0.5 * x * (1 + erf(x / np.sqrt(2)))


def mc_dropout_head(
    feats: np.ndarray, head: dict[str, np.ndarray], samples: int, rng: np.random.Generator
) -> np.ndarray:
    """MC-dropout over the classifier head in numpy: (samples, classes) logits."""
    p = float(head.get("dropout", 0.3))
    out = []
    for _ in range(samples):
        m1 = (rng.random(feats.shape) > p) / (1 - p)
        h = gelu((feats * m1) @ head["w1"].T + head["b1"])
        m2 = (rng.random(h.shape) > p) / (1 - p)
        out.append((h * m2) @ head["w2"].T + head["b2"])
    return np.stack(out)


def certainty(p: float, thr: float) -> str:
    if p >= thr and (p >= 0.5 or p >= 2 * thr):
        return "Likely"
    if p >= thr:
        return "Possible"
    return "Unlikely"


def joint_energy(logits: np.ndarray) -> float:
    return float(np.sum(np.log1p(np.exp(np.clip(logits, -30, 30)))))


def member_forward(
    h: ModelHandle, img: np.ndarray, samples: int, rng: np.random.Generator
) -> dict[str, Any]:
    x = preprocess(img, h.img_size)
    logits, feats = h.run(x)[:2]
    mc = mc_dropout_head(feats[0], h.head, samples, rng) if h.head is not None and samples else None
    return {"logits": logits[0], "mc_logits": mc}


def analyze_chest(img: np.ndarray, samples: int = 8, seed: int = 0) -> dict[str, Any]:
    members = registry.chest_members()
    if not members:
        return {"available": False, "findings": [], "error": "chest models not trained"}
    cal = registry.extra.get("chest_calibration", {})
    temps = cal.get("temperatures", {})
    thr = cal.get("thresholds", {})
    rng = np.random.default_rng(seed)
    probs, mc_vars, energies, member_probs = [], [], [], {}
    for h in members:
        out = member_forward(h, img, samples, rng)
        t = np.array(temps.get(h.name, [1.0] * len(NIH_LABELS)))
        p = sigmoid(out["logits"] / t)
        probs.append(p)
        member_probs[h.name] = p
        energies.append(joint_energy(out["logits"]))
        if out["mc_logits"] is not None:
            mc_vars.append(sigmoid(out["mc_logits"] / t).var(0))
    P = np.stack(probs)
    mean, std = P.mean(0), P.std(0)
    mcv = np.mean(mc_vars, 0) if mc_vars else np.zeros_like(mean)
    unc = np.sqrt(std**2 + mcv)
    findings = []
    for k, lab in enumerate(NIH_LABELS):
        t_k = float(thr.get(lab, 0.5))
        pk = float(mean[k])
        findings.append(
            {
                "label": display_name(lab),
                "key": lab,
                "probability": round(pk, 4),
                "uncertainty": round(float(unc[k]), 4),
                "ensemble_std": round(float(std[k]), 4),
                "mc_dropout_var": round(float(mcv[k]), 5),
                "threshold": round(t_k, 4),
                "positive": pk >= t_k,
                "certainty": certainty(pk, t_k),
                "ratio_to_threshold": round(pk / max(t_k, 1e-6), 3),
                "member_probs": {m: round(float(v[k]), 4) for m, v in member_probs.items()},
                "kind": "classification",
            }
        )
    findings.sort(key=lambda f: (-f["positive"], -f["ratio_to_threshold"]))
    e = float(np.mean(energies))
    e_thr = cal.get("energy_threshold")
    ood = bool(e_thr is not None and e < e_thr)
    return {
        "available": True,
        "findings": findings,
        "members": [h.name for h in members],
        "profile": registry.extra.get("chest_profile", members[0].profile),
        "energy": round(e, 3),
        "energy_threshold": e_thr,
        "energy_ood": ood,
        "img_sizes": {h.name: h.img_size for h in members},
    }
