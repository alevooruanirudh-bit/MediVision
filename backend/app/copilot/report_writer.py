"""MediVision Copilot report writer (default engine, no LLM).

Three-stage clinical NLG:
  1. content selection   — findings passing calibrated thresholds, borderline high-severity findings
                           ("cannot exclude"), pertinent negatives; urgency ordering.
  2. sentence planning   — finding + location (heatmap peak mapped to lung zone / region) + certainty
                           wording from the calibrated probability.
  3. realisation         — phrase bank mined from IU X-ray reports (normal / negative statements) and
                           hand-written positive templates, several variants per pattern chosen
                           deterministically from the study id.
Deterministic, fast, unit-tested.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import DISCLAIMER
from app.copilot.kb import entry
from app.inference.triage import config as triage_config

HERE = Path(__file__).parent

LUNG = {
    "Atelectasis",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
}
PLEURA = {"Effusion", "Pneumothorax", "Pleural Thickening"}
CARDIAC = {"Cardiomegaly", "Hernia"}

NP: dict[str, tuple[str, str]] = {  # (with article, bare)
    "Atelectasis": ("atelectasis", "atelectasis"),
    "Effusion": ("a pleural effusion", "pleural effusion"),
    "Infiltration": ("an infiltrate", "infiltrate"),
    "Mass": ("a mass", "mass"),
    "Nodule": ("a pulmonary nodule", "pulmonary nodule"),
    "Pneumonia": ("pneumonia", "pneumonia"),
    "Pneumothorax": ("a pneumothorax", "pneumothorax"),
    "Consolidation": ("consolidation", "consolidation"),
    "Edema": ("pulmonary oedema", "pulmonary oedema"),
    "Emphysema": ("emphysematous change", "emphysematous change"),
    "Fibrosis": ("fibrotic change", "fibrotic change"),
    "Pleural Thickening": ("pleural thickening", "pleural thickening"),
    "Hernia": ("a hiatus hernia", "hiatus hernia"),
    "Cardiomegaly": ("cardiomegaly", "cardiomegaly"),
    "Fracture": ("a fracture", "fracture"),
    "Glioma": ("a glioma", "glioma"),
    "Meningioma": ("a meningioma", "meningioma"),
    "Pituitary tumour": ("a pituitary tumour", "pituitary tumour"),
    "Intracranial hemorrhage": ("intracranial haemorrhage", "intracranial haemorrhage"),
    "Suspicious breast lesion": ("a suspicious breast lesion", "suspicious breast lesion"),
    "Referable diabetic retinopathy": (
        "referable diabetic retinopathy",
        "referable diabetic retinopathy",
    ),
    "Malignancy-suspicious lung nodule": (
        "a lung nodule with malignant features",
        "lung nodule with malignant features",
    ),
}
NEW_PIPES = {
    "head_ct": (
        "Single axial non-contrast head CT slice, brain window (de-identified upload).",
        "Brain",
        "No intracranial haemorrhage identified by the AI model on this slice. A single slice "
        "cannot exclude haemorrhage elsewhere.",
        "No intracranial haemorrhage identified on this slice.",
    ),
    "breast_us": (
        "Single breast ultrasound image (de-identified upload).",
        "Breast",
        "No features suspicious for malignancy identified by the AI model.",
        "No sonographic features suspicious for malignancy identified by the AI model.",
    ),
    "fundus": (
        "Single colour fundus photograph (de-identified upload).",
        "Retina",
        "No referable diabetic retinopathy identified by the AI model.",
        "No referable diabetic retinopathy identified.",
    ),
    "ct_volume": (
        "Chest CT sub-volume centred on a nodule (64³ patch, de-identified upload).",
        "Lung nodule",
        "The nodule does not show features the AI model associates with malignancy.",
        "Nodule without AI-suspected malignant features; follow guideline-based surveillance.",
    ),
}
LIKELY = [
    "Findings are consistent with {np}{zone}.",
    "Appearances are likely to represent {np}{zone}.",
    "There is likely {np}{zone}.",
]
POSSIBLE = [
    "Possible {npb}{zone}.",
    "Findings may represent {np}{zone}.",
    "Subtle features raise the possibility of {np}{zone}.",
]
EXCLUDE = ["{NPB} cannot be excluded{zone}.", "A subtle {npb} cannot be excluded{zone}."]
IMP_LIKELY = ["{NPB}{zone} (likely).", "Likely {npb}{zone}."]
IMP_POSSIBLE = ["Possible {npb}{zone}.", "{NPB}{zone} is possible."]
IMP_EXCLUDE = ["{NPB} cannot be excluded{zone}."]
HIGH_SEVERITY = {
    "Pneumothorax",
    "Mass",
    "Pneumonia",
    "Consolidation",
    "Edema",
    "Effusion",
    "Fracture",
    "Glioma",
    "Meningioma",
    "Pituitary tumour",
}


@lru_cache
def phrase_bank() -> dict[str, Any]:
    p = HERE / "phrase_bank.json"
    return json.loads(p.read_text()) if p.exists() else {"normal": {}, "negative": {}}


def _pick(options: list[str], seed: str) -> str:
    if not options:
        return ""
    i = int(hashlib.sha256(seed.encode()).hexdigest(), 16) % len(options)
    return options[i]


def _sentence(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    s = s[0].upper() + s[1:]
    return s if s.endswith((".", "!", "?")) else s + "."


def _bank(kind: str, key: str, seed: str, fallback: str) -> str:
    opts = [x["text"] for x in phrase_bank().get(kind, {}).get(key, [])][:4]
    return _sentence(_pick(opts, seed) or fallback)


def zone_clause(f: dict[str, Any]) -> str:
    z = f.get("zone")
    if not z or "cardiomediastinal" in z or f["label"] in CARDIAC:
        return ""
    if "image coordinates" in z:
        return f" at the {z.split(' (')[0]}"
    return f" in the {z}"


def realise(f: dict[str, Any], mode: str, seed: str, table: dict[str, list[str]]) -> str:
    np_, npb = NP.get(f["label"], (f["label"].lower(), f["label"].lower()))
    tpl = _pick(table[mode], seed + f["label"])
    return tpl.format(np=np_, npb=npb, NPB=npb[0].upper() + npb[1:], zone=zone_clause(f))


def certainty_mode(f: dict[str, Any]) -> str | None:
    if f.get("positive"):
        return "likely" if f.get("certainty") == "Likely" else "possible"
    if (
        f["label"] in HIGH_SEVERITY
        and f.get("ratio_to_threshold", 0) >= 0.7
        and f["probability"] >= 0.12
    ):
        return "exclude"
    return None


def select(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Content selection + urgency ordering."""
    w = triage_config()["weights"]
    chosen = []
    for f in results.get("findings", []):
        mode = certainty_mode(f)
        if mode:
            chosen.append({**f, "mode": mode})
    order = {"likely": 0, "possible": 1, "exclude": 2}
    chosen.sort(
        key=lambda f: (order[f["mode"]], -w.get(f["label"], 0.5) * f.get("ratio_to_threshold", 1))
    )
    return chosen


def _prob_note(f: dict[str, Any]) -> str:
    u = f.get("uncertainty")
    unc = f" ± {u:.2f}" if u else ""
    return f" (AI p = {f['probability']:.2f}{unc}; threshold {f.get('threshold', 0.5):.2f})"


def write_report(results: dict[str, Any], study: Any, style: str = "standard") -> dict[str, Any]:
    seed = getattr(study, "id", "seed")
    pipe = results.get("pipeline")
    chosen = select(results)
    meas = results.get("measurements", {}) or {}
    mod = results.get("modality", {}) or {}
    tables = {"likely": LIKELY, "possible": POSSIBLE, "exclude": EXCLUDE}
    imp_tables = {"likely": IMP_LIKELY, "possible": IMP_POSSIBLE, "exclude": IMP_EXCLUDE}
    findings: dict[str, str] = {}
    if pipe == "chest":
        view = "frontal"
        technique = f"Single {view} chest radiograph (de-identified upload)."
        lung = [f for f in chosen if f["label"] in LUNG]
        findings["Lungs"] = " ".join(
            realise(f, f["mode"], seed, tables) + _prob_note(f) for f in lung
        ) or _bank("normal", "lungs", seed, "The lungs are clear.")
        pl = [f for f in chosen if f["label"] in PLEURA]
        neg = [] if any(f["label"] == "Pneumothorax" for f in pl) else ["No pneumothorax."]
        if not any(f["label"] == "Effusion" for f in pl):
            neg = (
                [_bank("negative", "Effusion", seed, "No pleural effusion or pneumothorax.")]
                if not any(f["label"] == "Pneumothorax" for f in pl)
                else ["No pleural effusion."]
            )
        findings["Pleura"] = " ".join(
            [realise(f, f["mode"], seed, tables) + _prob_note(f) for f in pl] + neg
        )
        ctr = meas.get("ctr")
        cardio = next((f for f in chosen if f["label"] == "Cardiomegaly"), None)
        ctr_txt = f" (estimated CTR {ctr:.2f})" if ctr else ""
        if cardio or (ctr and ctr > 0.5):
            heart = f"The cardiac silhouette appears enlarged{ctr_txt}."
            if cardio and cardio["mode"] == "possible":
                heart = f"The cardiac silhouette is at the upper limit of normal or mildly enlarged{ctr_txt}."
        else:
            heart = (
                _bank("normal", "heart", seed, "Heart size is normal.").rstrip(".") + ctr_txt + "."
            )
        hernia = next((f for f in chosen if f["label"] == "Hernia"), None)
        findings["Heart and mediastinum"] = heart + (
            " " + realise(hernia, hernia["mode"], seed, tables) if hernia else ""
        )
        findings["Bones and soft tissues"] = (
            "Not assessed by the AI models (no skeletal model for chest radiographs)."
        )
    elif pipe == "fracture":
        technique = "Single musculoskeletal radiograph (de-identified upload)."
        fr = next((f for f in chosen if f["label"] == "Fracture"), None)
        det = results.get("detections", [])
        if fr:
            box = f" The detector marks {len(det)} candidate region(s)." if det else ""
            findings["Osseous structures"] = (
                realise(fr, fr["mode"], seed, tables) + _prob_note(fr) + box
            )
        else:
            findings["Osseous structures"] = (
                "No fracture identified by the AI model. Subtle or "
                "non-displaced fractures may not be visible on a single projection."
            )
    elif pipe == "brain":
        technique = "Single axial brain MRI slice (de-identified upload)."
        br = results.get("extra", {}).get("brain", {}) or {}
        tum = [f for f in chosen if f["label"] in ("Glioma", "Meningioma", "Pituitary tumour")]
        if tum:
            f = tum[0]
            seg = ""
            if meas.get("tumour_area_px"):
                seg = (
                    f" The segmentation model outlines a lesion of ~{meas['tumour_area_px']} px"
                    + (f" (~{meas['tumour_area_mm2']} mm²)" if meas.get("tumour_area_mm2") else "")
                    + "."
                )
            findings["Brain"] = realise(f, f["mode"], seed, tables) + _prob_note(f) + seg
        else:
            findings["Brain"] = (
                f"No features of glioma, meningioma or pituitary tumour identified by the "
                f"AI model on this slice (no-tumour probability "
                f"{br.get('no_tumour_prob', 0):.2f})."
            )
    elif pipe in NEW_PIPES:
        technique, sec, normal, _ = NEW_PIPES[pipe]
        pos = [f for f in chosen if f["label"] in NP]
        if pos:
            f = pos[0]
            extra_txt = ""
            if pipe == "fundus" and f.get("grade"):
                extra_txt = f" Most probable ICDR grade: {f['grade']}."
            if pipe == "breast_us" and f.get("risk_category"):
                extra_txt = f" AI suspicion category: {f['risk_category']}."
            if pipe == "ct_volume" and meas.get("volume_mm3"):
                extra_txt = (
                    f" Segmented volume ~{meas['volume_mm3']:.0f} mm³ "
                    f"(equivalent diameter {meas.get('equivalent_diameter_mm', 0):.1f} mm)."
                )
            findings[sec] = realise(f, f["mode"], seed, tables) + _prob_note(f) + extra_txt
        else:
            findings[sec] = normal
            fs = results.get("findings") or []
            if pipe == "fundus" and fs and fs[0].get("grade"):
                findings[sec] += f" Most probable ICDR grade: {fs[0]['grade']}."
    else:
        technique = "Image could not be analysed."
        findings["Summary"] = (
            mod.get("warning") or "The input was not recognised as a supported modality."
        )

    # Measurements
    m_lines = []
    if pipe == "chest" and meas.get("available"):
        if meas.get("ctr"):
            m_lines.append(
                f"Cardiothoracic ratio (estimate): {meas['ctr']:.2f}"
                + (" — above 0.5" if meas.get("ctr_flag") else " — within normal range")
            )
        if meas.get("right_lung_area_px"):
            unit = "mm²" if meas.get("right_lung_area_mm2") else "px"
            r = meas.get("right_lung_area_mm2") or meas["right_lung_area_px"]
            l_ = meas.get("left_lung_area_mm2") or meas["left_lung_area_px"]
            m_lines.append(f"Lung areas: right {r:,} {unit}, left {l_:,} {unit}")
        if meas.get("symmetry_index"):
            m_lines.append(f"Lung symmetry index: {meas['symmetry_index']:.2f}")
    if pipe == "brain" and meas.get("available"):
        m_lines.append(
            f"Segmented lesion area: {meas.get('tumour_area_px', 0):,} px"
            + (
                f" ({meas['tumour_area_mm2']} mm²)"
                if meas.get("tumour_area_mm2")
                else " (pixel spacing unknown)"
            )
        )
    if pipe == "ct_volume" and meas.get("available"):
        m_lines.append(
            f"Nodule volume: {meas['volume_mm3']:.0f} mm³ · equivalent diameter "
            f"{meas['equivalent_diameter_mm']:.1f} mm · {meas['slices_with_nodule']} slices"
            + (" (voxel spacing assumed 1 mm — not in file)" if meas.get("spacing_assumed") else "")
        )
    rad = results.get("radiomics") or {}
    if rad.get("first_order"):
        fo, tx = rad["first_order"], rad.get("texture_glcm") or {}
        m_lines.append(
            f"Radiomics ({rad.get('roi')}): mean {fo.get('mean')}, SD {fo.get('std')}, "
            f"entropy {fo.get('entropy_bits')} bits"
            + (f", GLCM contrast {tx['contrast']}, homogeneity {tx['homogeneity']}" if tx else "")
            + " (descriptive only)"
        )
    if pipe == "fracture" and results.get("detections"):
        m_lines.append(
            f"Detector candidate boxes: {len(results['detections'])} "
            f"(top score {results['detections'][0]['score']:.2f})"
        )

    # Impression (numbered, urgency-ordered)
    imp = [realise(f, f["mode"], seed + "imp", imp_tables) for f in chosen]
    if not imp:
        if pipe == "chest":
            imp = [
                _sentence(
                    _pick([x["text"] for x in phrase_bank().get("impression_normal", [])][:4], seed)
                    or "No acute cardiopulmonary abnormality"
                )
            ]
        elif pipe == "fracture":
            imp = ["No fracture identified by the AI model."]
        elif pipe == "brain":
            imp = ["No tumour of the trained types identified on this slice."]
        elif pipe in NEW_PIPES:
            imp = [NEW_PIPES[pipe][3]]
        else:
            imp = ["Not analysed — unsupported or non-medical input."]
    if (
        pipe == "chest"
        and meas.get("ctr_flag")
        and not any("cardiomegaly" in i.lower() for i in imp)
    ):
        imp.append(
            f"Estimated cardiothoracic ratio {meas['ctr']:.2f} (> 0.5); correlate for cardiomegaly."
        )

    # Recommendations from the knowledge base
    recs: list[str] = []
    for f in chosen[:3]:
        e = entry(f.get("key") or f["label"]) or entry(f["label"])
        if e and e.get("next_steps"):
            recs.append(f"{f['label']}: {e['next_steps'][0]}")
    recs.append(
        "Clinical correlation is recommended; this AI draft must be reviewed by a qualified clinician."
    )

    # AI confidence & limitations
    prof = results.get("model_profile", "starter")
    lims = [
        f"Models: MediVision self-trained '{prof}' profile ({results.get('backend', 'onnx')} runtime)."
    ]
    if pipe == "chest":
        cm = results.get("extra", {}).get("chest", {}) or {}
        lims.append(
            f"Chest ensemble of {len(cm.get('members') or [])} networks; probabilities are "
            "temperature-calibrated; uncertainty = ensemble spread + MC-dropout."
        )
    unc = sorted(results.get("findings", []), key=lambda f: -f.get("uncertainty", 0))[:1]
    if unc and unc[0].get("uncertainty", 0) > 0.08:
        lims.append(
            f"Highest model uncertainty: {unc[0]['label']} (± {unc[0]['uncertainty']:.2f})."
        )
    if mod.get("warning"):
        lims.append(f"Input check: {mod['warning']}")
    if prof == "starter":
        lims.append(
            "Starter models were trained on low-resolution images for a few epochs on a laptop; "
            "accuracy is limited (see the Lab page)."
        )
    lims.append(
        "The AI cannot assess clinical history, prior imaging (unless compared), or findings "
        "outside its trained label set."
    )
    return {
        "indication": getattr(study, "indication", None) or "Not provided.",
        "technique": technique,
        "findings": findings,
        "measurements": m_lines or ["No automatic measurements available."],
        "impression": [f"{i + 1}. {t}" for i, t in enumerate(imp)],
        "recommendations": recs,
        "ai_confidence": lims,
        "disclaimer": DISCLAIMER + " AI-generated draft — requires review and sign-off.",
        "meta": {
            "engine": "rules",
            "style": style,
            "pipeline": pipe,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "n_selected": len(chosen),
        },
    }


SECTION_ORDER = [
    "indication",
    "technique",
    "findings",
    "measurements",
    "impression",
    "recommendations",
    "ai_confidence",
    "disclaimer",
]
SECTION_TITLES = {
    "indication": "Indication",
    "technique": "Technique",
    "findings": "Findings",
    "measurements": "Measurements",
    "impression": "Impression",
    "recommendations": "Recommendations",
    "ai_confidence": "AI confidence & limitations",
    "disclaimer": "Disclaimer",
}


def to_text(sections: dict[str, Any]) -> str:
    out = []
    for k in SECTION_ORDER:
        v = sections.get(k)
        if v is None:
            continue
        out.append(SECTION_TITLES[k].upper())
        if isinstance(v, dict):
            out += [f"{sk}: {sv}" for sk, sv in v.items()]
        elif isinstance(v, list):
            out += [str(x) for x in v]
        else:
            out.append(str(v))
        out.append("")
    return "\n".join(out).strip()
