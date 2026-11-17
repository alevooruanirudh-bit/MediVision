"""Urgency scoring 0-100 -> CRITICAL / URGENT / ROUTINE / NORMAL (weights in triage_weights.yaml)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

WEIGHTS_FILE = Path(__file__).with_name("triage_weights.yaml")


@lru_cache
def config() -> dict[str, Any]:
    return yaml.safe_load(WEIGHTS_FILE.read_text())


def score(findings: list[dict[str, Any]], ood: bool = False) -> dict[str, Any]:
    cfg = config()
    w = {k.lower(): v for k, v in cfg["weights"].items()}
    lv = cfg["levels"]
    survive = 1.0
    drivers = []
    for f in findings:
        if not f.get("positive"):
            continue
        p, t = float(f["probability"]), float(f.get("threshold", 0.5))
        ev = 0.5 + 0.5 * min(max((p - t) / max(1 - t, 1e-6), 0.0), 1.0)
        conf = 1 - min(2 * float(f.get("uncertainty", 0.0)), 0.5)
        wi = float(w.get(f["label"].lower(), 0.5))
        contrib = wi * ev * conf
        survive *= 1 - contrib
        drivers.append(
            {
                "label": f["label"],
                "weight": wi,
                "evidence": round(ev, 3),
                "confidence": round(conf, 3),
                "contribution": round(100 * contrib, 1),
            }
        )
    s = round(100 * (1 - survive), 1)
    if s >= lv["CRITICAL"]:
        level = "CRITICAL"
    elif s >= lv["URGENT"]:
        level = "URGENT"
    elif s >= lv["ROUTINE"] or ood:
        level = "ROUTINE"
    else:
        level = "NORMAL"
    drivers.sort(key=lambda d: -d["contribution"])
    note = "Doubtful / out-of-distribution input — needs human review." if ood else None
    return {"score": s, "level": level, "drivers": drivers, "note": note}
