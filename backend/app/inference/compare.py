"""Longitudinal comparison of two analysed studies of one patient."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.inference import xai
from app.models import Study
from app.services import storage


def compare_results(a: Study, b: Study) -> dict[str, Any]:
    """a = prior, b = current."""
    ra, rb = a.results, b.results
    fa = {f["label"]: f for f in ra.get("findings", [])}
    fb = {f["label"]: f for f in rb.get("findings", [])}
    deltas = []
    for label in sorted(set(fa) | set(fb)):
        pa = fa.get(label, {}).get("probability", 0.0)
        pb = fb.get(label, {}).get("probability", 0.0)
        posa, posb = bool(fa.get(label, {}).get("positive")), bool(
            fb.get(label, {}).get("positive")
        )
        status = (
            "new"
            if posb and not posa
            else "resolved" if posa and not posb else "persistent" if posa and posb else "absent"
        )
        d = pb - pa
        trend = "increased" if d > 0.05 else "decreased" if d < -0.05 else "stable"
        deltas.append(
            {
                "label": label,
                "prior": round(pa, 4),
                "current": round(pb, 4),
                "delta": round(d, 4),
                "status": status,
                "trend": trend,
                "key": (fb.get(label) or fa.get(label) or {}).get("key", label),
            }
        )
    deltas.sort(key=lambda x: (x["status"] not in ("new", "resolved"), -abs(x["delta"])))
    ma, mb = ra.get("measurements", {}) or {}, rb.get("measurements", {}) or {}
    ctr = None
    if ma.get("ctr") and mb.get("ctr"):
        ctr = {"prior": ma["ctr"], "current": mb["ctr"], "delta": round(mb["ctr"] - ma["ctr"], 3)}
    ta, tb = ra.get("triage", {}), rb.get("triage", {})
    summary = interval_paragraph(deltas, ctr, ta, tb, a, b)
    return {
        "prior": a.id,
        "current": b.id,
        "deltas": deltas,
        "ctr": ctr,
        "triage": {"prior": ta, "current": tb},
        "summary": summary,
    }


def interval_paragraph(
    deltas: list[dict[str, Any]],
    ctr: dict[str, Any] | None,
    ta: dict[str, Any],
    tb: dict[str, Any],
    a: Study,
    b: Study,
) -> str:
    parts = [f"Compared with the prior study ({a.study_date or a.created_at.date()}):"]
    new = [d["label"] for d in deltas if d["status"] == "new"]
    res = [d["label"] for d in deltas if d["status"] == "resolved"]
    pers = [d for d in deltas if d["status"] == "persistent"]
    if new:
        parts.append("new " + ", ".join(new).lower() + " is suggested by the models.")
    if res:
        parts.append(", ".join(res) + (" is" if len(res) == 1 else " are") + " no longer flagged.")
    for d in pers:
        word = {
            "increased": "slightly more conspicuous",
            "decreased": "less conspicuous",
            "stable": "similar",
        }[d["trend"]]
        parts.append(
            f"{d['label']} persists and appears {word} (p {d['prior']:.2f} → {d['current']:.2f})."
        )
    if ctr:
        if abs(ctr["delta"]) >= 0.03:
            parts.append(
                f"The estimated cardiothoracic ratio has {'increased' if ctr['delta'] > 0 else 'decreased'} "
                f"from {ctr['prior']:.2f} to {ctr['current']:.2f}."
            )
        else:
            parts.append(
                f"Cardiothoracic ratio is stable ({ctr['prior']:.2f} → {ctr['current']:.2f})."
            )
    if not (new or res or pers):
        parts.append(
            "no finding passes threshold on either study; no significant interval change by AI."
        )
    parts.append(
        f"Triage {ta.get('level', 'n/a')} → {tb.get('level', 'n/a')}. "
        "Interval change by AI is approximate; positioning and exposure differences affect it."
    )
    return " ".join(parts)


def heatmap_difference(a: Study, b: Study, key: str) -> str | None:
    """Overlay of Grad-CAM change for one finding, computed on study b's geometry."""
    from app.inference.pipeline import explain

    heats = []
    for s in (a, b):
        p = storage.study_dir(s.id) / "overlays" / f"heat_{key}_gradcam.npy"
        if not p.exists():
            explain(s, key)
        heats.append(np.load(p).astype(np.float32))
    h, w = heats[1].shape
    ha = cv2.resize(heats[0], (w, h))
    return storage.url(
        storage.save_rgba(b.id, f"diff_{key}_vs_{a.id}", xai.diff_overlay(ha, heats[1]))
    )
