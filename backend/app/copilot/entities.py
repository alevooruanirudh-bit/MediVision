"""Entity extraction for the Copilot: finding names + synonyms (gazetteer + rapidfuzz fuzzy matching),
lung zones / sides, and report-section / style names."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from rapidfuzz import fuzz, process

from app.copilot.kb import entries

# canonical model labels (as shown in the UI) -> KB id
LABELS = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Effusion": "Effusion",
    "Infiltration": "Infiltration",
    "Mass": "Mass",
    "Nodule": "Nodule",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
    "Emphysema": "Emphysema",
    "Fibrosis": "Fibrosis",
    "Pleural Thickening": "Pleural_Thickening",
    "Hernia": "Hernia",
    "Fracture": "Fracture",
    "Glioma": "glioma",
    "Meningioma": "meningioma",
    "Pituitary tumour": "pituitary",
}
KB_TO_LABEL = {v: k for k, v in LABELS.items()}
SECTIONS = {
    "impression": "impression",
    "findings": "findings",
    "recommendation": "recommendations",
    "recommendations": "recommendations",
    "measurements": "measurements",
    "technique": "technique",
    "indication": "indication",
    "report": "all",
}
STYLES = {
    "concise": "concise",
    "short": "concise",
    "shorter": "concise",
    "brief": "concise",
    "detailed": "detailed",
    "detail": "detailed",
    "expand": "detailed",
    "longer": "detailed",
    "patient": "patient",
    "lay": "patient",
    "plain": "patient",
    "simple": "patient",
    "friendly": "patient",
}
METHODS = {
    "grad-cam++": "gradcampp",
    "gradcam++": "gradcampp",
    "grad cam plus plus": "gradcampp",
    "occlusion": "occlusion",
    "grad-cam": "gradcam",
    "gradcam": "gradcam",
}


@lru_cache
def gazetteer() -> dict[str, str]:
    """surface form -> KB id"""
    g: dict[str, str] = {}
    for e in entries():
        for s in [e["id"].replace("_", " "), e["name"], *e.get("synonyms", [])]:
            g[s.lower()] = e["id"]
    for label, kb in LABELS.items():
        g[label.lower()] = kb
    return g


def find_terms(text: str, threshold: int = 88) -> list[str]:
    """KB ids mentioned in text (exact phrase match first, then fuzzy for typos)."""
    t = " " + re.sub(r"[^a-z0-9+ -]", " ", text.lower()) + " "
    hits: list[tuple[int, str]] = []
    g = gazetteer()
    for surface in sorted(g, key=len, reverse=True):
        pos = t.find(" " + surface + " ")
        if pos < 0 and surface.endswith("s") is False:
            pos = t.find(" " + surface + "s ")
        if pos >= 0 and g[surface] not in [h[1] for h in hits]:
            hits.append((pos, g[surface]))
    if not hits:
        words = t.split()
        grams = words + [" ".join(words[i : i + 2]) for i in range(len(words) - 1)]
        for gram in grams:
            if len(gram) < 5:
                continue
            m = process.extractOne(gram, list(g), scorer=fuzz.ratio, score_cutoff=threshold)
            if m and g[m[0]] not in [h[1] for h in hits]:
                hits.append((t.find(gram), g[m[0]]))
    return [h[1] for h in sorted(hits)]


def extract(text: str) -> dict[str, Any]:
    t = text.lower()
    terms = find_terms(text)
    findings = [KB_TO_LABEL[k] for k in terms if k in KB_TO_LABEL]
    side = "left" if re.search(r"\bleft\b", t) else "right" if re.search(r"\bright\b", t) else None
    zone = next((z for z in ("upper", "middle", "lower") if re.search(rf"\b{z}\b", t)), None)
    section = next((v for k, v in SECTIONS.items() if re.search(rf"\b{k}\b", t)), None)
    style = next((v for k, v in STYLES.items() if re.search(rf"\b{k}\b", t)), None)
    method = next((v for k, v in METHODS.items() if k in t), None)
    return {
        "terms": terms,
        "findings": findings,
        "side": side,
        "zone": zone,
        "section": section,
        "style": style,
        "method": method,
    }
