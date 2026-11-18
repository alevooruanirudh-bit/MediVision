"""MediVision Copilot Q&A engine (self-built, no LLM).

intent model (TF-IDF + LR, trained by us) + entity extraction (gazetteer + rapidfuzz) -> answer
composers grounded in THIS study's outputs (probabilities, uncertainty, heatmap zone, measurements,
comparison deltas) + BM25 retrieval over our knowledge base. Answers carry UI action chips and follow-up
suggestions. The engine never states a definitive diagnosis and always recommends clinical correlation.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sqlmodel import Session, select

from app.copilot import entities as E
from app.copilot.kb import entry, search
from app.copilot.report_writer import write_report
from app.copilot.rewrite import rewrite
from app.inference.registry import registry
from app.models import ChatMessage, Report, Study
from ml.copilot.text import featurize

NEEDS_FOCUS = {
    "explain_finding",
    "why_finding",
    "where_finding",
    "show_heatmap",
    "differential",
    "patient_explanation_focus",
}
CORRELATE = "This is AI decision support, not a diagnosis — please correlate clinically."
FALLBACK_SUGGESTIONS = [
    "What did you find?",
    "Is this scan normal?",
    "How urgent is this?",
    "What are the limitations?",
]


@dataclass
class Answer:
    text: str
    intent: str
    confidence: float
    focus: str | None = None
    actions: list[dict[str, Any]] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    entities: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class Ctx:
    question: str
    study: Study
    results: dict[str, Any]
    session: Session | None
    ent: dict[str, Any]
    focus: dict[str, Any] | None  # finding dict from results
    focus_label: str | None


def classify(question: str) -> tuple[str, float, dict[str, float]]:
    model = registry.extra.get("intent_model")
    if model is None:
        registry.load()
        model = registry.extra.get("intent_model")
    if model is None:
        return "unknown", 0.0, {}
    probs = model.predict_proba([featurize(question)])[0]
    classes = list(model.classes_)
    order = probs.argsort()[::-1]
    return (
        classes[order[0]],
        float(probs[order[0]]),
        {classes[i]: float(probs[i]) for i in order[:3]},
    )


# ----------------------------------------------------------------------------- helpers
def _pct(p: float) -> str:
    return f"{p * 100:.0f}%"


def _positives(r: dict[str, Any]) -> list[dict[str, Any]]:
    return [f for f in r.get("findings", []) if f.get("positive")]


def _find(r: dict[str, Any], label: str | None) -> dict[str, Any] | None:
    if not label:
        return None
    return next((f for f in r.get("findings", []) if f["label"].lower() == label.lower()), None)


def _heat_action(f: dict[str, Any], method: str = "gradcam") -> dict[str, Any]:
    return {
        "label": f"Show heatmap: {f['label']}",
        "type": "show_heatmap",
        "finding": f["key"],
        "method": method,
    }


def _desc(f: dict[str, Any]) -> str:
    z = f" in the {f['zone']}" if f.get("zone") and "image" not in (f.get("zone") or "") else ""
    return (
        f"{f['label']} — {f['certainty'].lower()} (calibrated probability {_pct(f['probability'])}, "
        f"operating threshold {_pct(f['threshold'])}){z}"
    )


def _label_metrics(label: str) -> dict[str, Any]:
    m = registry.extra.get("chest_metrics", {}).get("test", {}).get("per_label", {})
    key = label.replace(" ", "_")
    return m.get(key) or m.get(label) or {}


def _prior(ctx: Ctx) -> Study | None:
    if ctx.session is None:
        return None
    q = select(Study).where(
        Study.patient_id == ctx.study.patient_id,
        Study.id != ctx.study.id,
        Study.status == "analyzed",
    )
    priors = [s for s in ctx.session.exec(q) if s.created_at <= ctx.study.created_at] or list(
        ctx.session.exec(q)
    )
    return sorted(priors, key=lambda s: s.created_at)[-1] if priors else None


def _report_sections(ctx: Ctx) -> dict[str, Any]:
    if ctx.session is not None:
        rep = ctx.session.exec(
            select(Report).where(Report.study_id == ctx.study.id).order_by(Report.updated_at.desc())
        ).first()  # type: ignore[attr-defined]
        if rep:
            return rep.sections
    return write_report(ctx.results, ctx.study)


def _no_focus(ctx: Ctx, what: str) -> Answer:
    pos = _positives(ctx.results)
    names = ", ".join(f["label"] for f in pos[:3]) or "none above threshold"
    return Answer(
        f"Which finding do you mean for {what}? Flagged in this study: {names}.",
        "clarify",
        0.0,
        suggestions=[f"{what.capitalize()} for {f['label']}" for f in pos[:3]]
        or FALLBACK_SUGGESTIONS,
    )


# ----------------------------------------------------------------------------- composers
def c_greeting(ctx: Ctx) -> Answer:
    pipe = ctx.results.get("pipeline") or "this study"
    return Answer(
        "Hi — I'm the MediVision Copilot, a self-built clinical language engine (no external AI). "
        f"I can explain the {pipe} findings, show where the models looked, give probabilities and "
        "uncertainty, measurements, urgency, comparisons with prior studies, definitions, and rewrite "
        "report sections. Everything I say comes from this study's model outputs and our knowledge base.",
        "greeting_help",
        1.0,
    )


def c_list(ctx: Ctx) -> Answer:
    pos = _positives(ctx.results)
    fs = ctx.results.get("findings", [])
    if not fs:
        return Answer(
            "No model findings are available for this study "
            f"({ctx.results.get('modality', {}).get('warning') or 'not analysed'}).",
            "list_findings",
            1.0,
        )
    lines = [f"• {_desc(f)}" for f in pos] or [
        "No finding passed its calibrated operating threshold."
    ]
    others = [f for f in fs if not f.get("positive")][:3]
    if others:
        lines.append(
            "Highest remaining scores: "
            + ", ".join(f"{f['label']} {_pct(f['probability'])}" for f in others)
            + "."
        )
    acts = [_heat_action(f) for f in pos[:3]]
    return Answer(
        "Findings for this study:\n" + "\n".join(lines) + f"\n{CORRELATE}",
        "list_findings",
        1.0,
        actions=acts,
    )


def c_explain(ctx: Ctx) -> Answer:
    f = ctx.focus
    if not f:
        return c_define(ctx) if ctx.ent["terms"] else _no_focus(ctx, "an explanation")
    e = entry(f["key"]) or entry(f["label"]) or {}
    parts = [f"In this study the model reports {_desc(f)}."]
    if e.get("definition"):
        parts.append(f"{e['name']}: {e['definition']}")
    if e.get("appearance"):
        parts.append(f"Typical appearance: {e['appearance']}")
    if f.get("uncertainty", 0) > 0.08:
        parts.append(f"Note the model uncertainty (± {f['uncertainty']:.2f}) — review carefully.")
    parts.append(CORRELATE)
    return Answer(
        " ".join(parts),
        "explain_finding",
        1.0,
        f["label"],
        [_heat_action(f)],
        sources=[e.get("name", "")] if e else [],
    )


def c_why(ctx: Ctx) -> Answer:
    f = ctx.focus
    if not f:
        return _no_focus(ctx, "the reasoning")
    parts = [
        f"{f['label']} was {'flagged' if f.get('positive') else 'not flagged'} because its "
        f"calibrated probability is {_pct(f['probability'])} versus an operating threshold of "
        f"{_pct(f['threshold'])} (chosen on validation data to maximise Youden's J)."
    ]
    mp = f.get("member_probs") or {}
    if mp:
        parts.append(
            "Ensemble members: "
            + ", ".join(f"{k.replace('chest_', '')} {_pct(v)}" for k, v in mp.items())
            + "."
        )
    if f.get("zone"):
        parts.append(f"Grad-CAM shows the strongest evidence in the {f['zone']}.")
    if f.get("detector_score"):
        parts.append(
            f"The fracture detector also proposes a box (score {f['detector_score']:.2f})."
        )
    for r in (f.get("rationale") or {}).get("reasons", []):
        if r.get("kind") in ("agreement", "measurement", "input"):
            parts.append(r["text"])
    parts.append("Heatmaps show where the network looked, not proof of disease. " + CORRELATE)
    return Answer(
        " ".join(parts),
        "why_finding",
        1.0,
        f["label"],
        [
            _heat_action(f),
            {
                "label": "Occlusion map",
                "type": "show_heatmap",
                "finding": f["key"],
                "method": "occlusion",
            },
        ],
    )


def c_where(ctx: Ctx) -> Answer:
    f = ctx.focus
    dets = ctx.results.get("detections", [])
    if not f and dets:
        f = _find(ctx.results, "Fracture")
    if not f:
        return _no_focus(ctx, "the location")
    if not f.get("zone"):
        try:
            from app.inference.pipeline import explain

            ex = explain(ctx.study, f["key"])
            f["zone"] = ex["zone"]
        except Exception:
            pass
    loc = f.get("zone") or "an undetermined region"
    txt = f"The strongest model evidence for {f['label']} is in the {loc} (Grad-CAM peak)."
    if f["label"] == "Fracture" and dets:
        txt += f" The detector marks {len(dets)} candidate box(es); the top one has score {dets[0]['score']:.2f}."
    if ctx.results.get("pipeline") == "chest":
        txt += " Sides are the patient's (radiological convention: patient right = image left)."
    return Answer(txt, "where_finding", 1.0, f["label"], [_heat_action(f)])


def c_confidence(ctx: Ctx) -> Answer:
    f = ctx.focus or (_positives(ctx.results) or ctx.results.get("findings", [None]))[0]
    if not f:
        return Answer("No findings to report confidence for.", "confidence", 1.0)
    m = _label_metrics(f["label"])
    txt = (
        f"{f['label']}: calibrated probability {_pct(f['probability'])} (category: "
        f"{f['certainty']}), uncertainty ± {f.get('uncertainty', 0):.2f}, threshold {_pct(f['threshold'])}."
    )
    if m.get("auroc"):
        txt += (
            f" On our held-out test set this label has AUROC {m['auroc']:.2f}, sensitivity "
            f"{_pct(m.get('sensitivity', 0))} and specificity {_pct(m.get('specificity', 0))} at "
            "this threshold."
        )
    return Answer(txt + " " + CORRELATE, "confidence", 1.0, f["label"], [_heat_action(f)])


def c_next(ctx: Ctx) -> Answer:
    targets = [ctx.focus] if ctx.focus else _positives(ctx.results)[:3]
    lines = []
    for f in targets:
        e = entry(f["key"]) or entry(f["label"])
        if e:
            lines += [f"• {f['label']}: {s}" for s in e.get("next_steps", [])[:2]]
    if not lines:
        lines = [
            "• No finding passed its threshold; routine review by a radiologist is still required."
        ]
    return Answer(
        "Typical next steps (general guidance from our knowledge base, not a care plan):\n"
        + "\n".join(lines)
        + f"\n{CORRELATE}",
        "next_steps",
        1.0,
        ctx.focus["label"] if ctx.focus else None,
        [{"label": "Open report", "type": "open_tab", "tab": "report"}],
    )


def c_compare(ctx: Ctx) -> Answer:
    prior = _prior(ctx)
    if prior is None:
        return Answer(
            "There is no earlier analysed study for this patient, so an interval comparison "
            "is not possible.",
            "compare_previous",
            1.0,
        )
    from app.inference.compare import compare_results

    cmp = compare_results(prior, ctx.study)
    return Answer(
        cmp["summary"],
        "compare_previous",
        1.0,
        actions=[
            {"label": "Open comparison", "type": "open_compare", "a": prior.id, "b": ctx.study.id}
        ],
        sources=[f"prior study {prior.id}"],
    )


def c_define(ctx: Ctx) -> Answer:
    term = ctx.ent["terms"][0] if ctx.ent["terms"] else None
    e = entry(term) if term else None
    if e is None:
        hits = search(ctx.question)
        e = hits[0][0] if hits else None
    if e is None:
        return fallback(ctx)
    parts = [f"{e['name']}: {e['definition']}"]
    if e.get("appearance"):
        parts.append(f"On imaging: {e['appearance']}")
    if e.get("causes"):
        parts.append("Common causes: " + ", ".join(e["causes"][:4]) + ".")
    if e.get("references"):
        parts.append("Reference: " + e["references"][0] + ".")
    acts = []
    f = _find(ctx.results, E.KB_TO_LABEL.get(e["id"], ""))
    if f:
        parts.append(f"In this study: {_desc(f)}.")
        acts.append(_heat_action(f))
    return Answer(
        " ".join(parts), "define_term", 1.0, f["label"] if f else None, acts, sources=[e["name"]]
    )


def c_heatmap(ctx: Ctx) -> Answer:
    f = ctx.focus or (_positives(ctx.results) or ctx.results.get("findings", [None]))[0]
    if not f:
        return Answer("There is nothing to explain for this study.", "show_heatmap", 1.0)
    method = ctx.ent.get("method") or "gradcam"
    names = {"gradcam": "Grad-CAM", "gradcampp": "Grad-CAM++", "occlusion": "occlusion sensitivity"}
    z = f" The hottest region maps to the {f['zone']}." if f.get("zone") else ""
    return Answer(
        f"Showing the {names[method]} map for {f['label']}.{z} Warm colours mark the regions "
        "that most increased the model's score.",
        "show_heatmap",
        1.0,
        f["label"],
        [_heat_action(f, method)],
    )


def c_measure(ctx: Ctx) -> Answer:
    m = ctx.results.get("measurements") or {}
    if not m.get("available"):
        return Answer(
            "No automatic measurements are available for this study.", "measurements", 1.0
        )
    lines = []
    if m.get("ctr"):
        lines.append(
            f"Cardiothoracic ratio ≈ {m['ctr']:.2f} ({'above' if m['ctr_flag'] else 'within'} "
            "the 0.5 PA-film cut-off; AP films magnify the heart)."
        )
    if m.get("right_lung_area_px"):
        unit = "mm²" if m.get("right_lung_area_mm2") else "px"
        lines.append(
            f"Lung areas: right {m.get('right_lung_area_mm2') or m['right_lung_area_px']:,} "
            f"{unit}, left {m.get('left_lung_area_mm2') or m['left_lung_area_px']:,} {unit}; "
            f"symmetry index {m.get('symmetry_index', 0):.2f}."
        )
    if "tumour_area_px" in m:
        lines.append(
            f"Segmented lesion area: {m['tumour_area_px']:,} px"
            + (
                f" ≈ {m['tumour_area_mm2']} mm²."
                if m.get("tumour_area_mm2")
                else " (pixel spacing unknown, so no mm² value)."
            )
        )
    lines.append("Measurements come from our segmentation models and are estimates.")
    return Answer(
        " ".join(lines),
        "measurements",
        1.0,
        actions=[
            {"label": "Show measurements", "type": "open_tab", "tab": "measurements"},
            {"label": "Toggle masks", "type": "toggle_overlay", "overlay": "mask"},
        ],
    )


def c_normal(ctx: Ctx) -> Answer:
    pos = _positives(ctx.results)
    mod = ctx.results.get("modality", {})
    if ctx.results.get("pipeline") is None:
        return Answer(f"This image was not analysed: {mod.get('warning')}", "normal_or_not", 1.0)
    if not pos:
        return Answer(
            "No finding passed its calibrated operating threshold, so the models consider this "
            "study within normal limits for the labels they know. That does not exclude "
            f"findings outside the trained label set. {CORRELATE}",
            "normal_or_not",
            1.0,
        )
    return Answer(
        "The models flag possible abnormalities: "
        + "; ".join(_desc(f) for f in pos[:4])
        + f". {CORRELATE}",
        "normal_or_not",
        1.0,
        pos[0]["label"],
        [_heat_action(f) for f in pos[:2]],
    )


def c_urgency(ctx: Ctx) -> Answer:
    t = ctx.results.get("triage", {})
    drv = ", ".join(f"{d['label']} (+{d['contribution']})" for d in t.get("drivers", [])[:3])
    txt = f"Triage: {t.get('level', 'n/a')} with urgency score {t.get('score', 0):.0f}/100." + (
        f" Main drivers: {drv}." if drv else " No finding passed its threshold."
    )
    if t.get("note"):
        txt += " " + t["note"]
    txt += (
        " The score combines severity weights, how far each calibrated probability exceeds its "
        "threshold, and model confidence (weights in the model card)."
    )
    return Answer(
        txt,
        "urgency",
        1.0,
        actions=[{"label": "Open worklist", "type": "navigate", "href": "/worklist"}],
    )


def c_limits(ctx: Ctx) -> Answer:
    cm = registry.extra.get("chest_metrics", {}).get("test", {})
    prof = ctx.results.get("model_profile", "starter")
    txt = [f"Active weights: '{prof}' profile."]
    if cm.get("macro_auroc"):
        txt.append(f"Chest ensemble macro AUROC on held-out data: {cm['macro_auroc']:.2f}.")
    txt += [
        "Main limitations: trained on public datasets that may not match your population or "
        "scanners; label noise in NIH ChestX-ray14 (labels were mined from reports); single 2-D "
        "images only; no clinical history; performance varies by sex, age and view (see the "
        "fairness panel in the Lab); heatmaps are approximate.",
        "This is a research & educational prototype — not a medical device and not for clinical use.",
    ]
    return Answer(
        " ".join(txt),
        "limitations",
        1.0,
        actions=[
            {"label": "Open Lab metrics", "type": "navigate", "href": "/lab"},
            {"label": "Model card", "type": "navigate", "href": "/model-card"},
        ],
    )


def c_rewrite(ctx: Ctx) -> Answer:
    style = ctx.ent.get("style") or ("patient" if "patient" in ctx.question.lower() else "concise")
    section = ctx.ent.get("section") or "impression"
    sec = rewrite(_report_sections(ctx), style, section)
    val = sec.get(section)
    preview = (
        " ".join(val)
        if isinstance(val, list)
        else " ".join(f"{k}: {v}" for k, v in val.items()) if isinstance(val, dict) else str(val)
    )
    return Answer(
        f"Here is the {section} rewritten in a {style} style:\n{preview}",
        "rewrite_section",
        1.0,
        actions=[
            {
                "label": f"Apply {style} style to report",
                "type": "rewrite_report",
                "style": style,
                "section": section,
            }
        ],
    )


def c_summary(ctx: Ctx) -> Answer:
    sec = _report_sections(ctx)
    t = ctx.results.get("triage", {})
    return Answer(
        "Impression: "
        + " ".join(sec.get("impression", []))
        + f" Triage {t.get('level', 'n/a')} ({t.get('score', 0):.0f}/100). {CORRELATE}",
        "summary",
        1.0,
        actions=[{"label": "Open report", "type": "open_tab", "tab": "report"}],
    )


def c_differential(ctx: Ctx) -> Answer:
    f = ctx.focus
    if not f:
        return _no_focus(ctx, "a differential")
    e = entry(f["key"]) or entry(f["label"]) or {}
    dd = e.get("differential", [])
    if not dd:
        return Answer(f"I don't have a curated differential for {f['label']}.", "differential", 1.0)
    return Answer(
        f"Differential considerations for {f['label']} (general knowledge base): "
        + "; ".join(dd)
        + f". {CORRELATE}",
        "differential",
        1.0,
        f["label"],
        [_heat_action(f)],
        sources=[e.get("name", "")],
    )


def c_uncertainty(ctx: Ctx) -> Answer:
    fs = ctx.results.get("findings", [])
    if not fs:
        return Answer("No model outputs to assess uncertainty for.", "uncertainty", 1.0)
    f = ctx.focus
    if f:
        mp = f.get("member_probs") or {}
        txt = (
            f"{f['label']}: combined uncertainty ± {f.get('uncertainty', 0):.2f} "
            f"(ensemble std {f.get('ensemble_std', 0):.2f}, MC-dropout variance "
            f"{f.get('mc_dropout_var', 0):.4f})."
        )
        if mp:
            txt += (
                " Members: "
                + ", ".join(f"{k.replace('chest_', '')} {_pct(v)}" for k, v in mp.items())
                + "."
            )
    else:
        top = sorted(fs, key=lambda x: -x.get("uncertainty", 0))[:3]
        txt = (
            "Most uncertain outputs: "
            + ", ".join(f"{x['label']} ± {x.get('uncertainty', 0):.2f}" for x in top)
            + "."
        )
    txt += " High uncertainty means the networks disagree — review those findings yourself."
    return Answer(txt, "uncertainty", 1.0, f["label"] if f else None)


def c_model_info(ctx: Ctx) -> Answer:
    st = registry.status()
    active = [f"{k} ({v['profile']}, {v['backend']})" for k, v in st.items() if v.get("loaded")]
    return Answer(
        "All models were designed and trained by this project on free public datasets "
        "(NIH ChestX-ray14 / ChestMNIST+, COVID-19 Radiography lung masks, FracAtlas, Brain "
        "Tumor MRI, LGG, IU X-ray). No external AI services are used; the Copilot is a "
        "rule-based + retrieval engine with our own intent classifier. Active models: "
        + ", ".join(active)
        + ".",
        "model_info",
        1.0,
        actions=[{"label": "Model card", "type": "navigate", "href": "/model-card"}],
    )


def c_quality(ctx: Ctx) -> Answer:
    m = ctx.results.get("modality", {})
    conf = m.get("confidence")
    txt = (
        f"Modality router: {str(m.get('label', 'unknown')).replace('_', ' ')}"
        + (f" ({_pct(conf)})" if conf is not None else "")
        + "."
    )
    ch = ctx.results.get("extra", {}).get("chest") or {}
    if ch.get("energy") is not None and ch.get("energy_threshold") is not None:
        txt += (
            f" Chest energy score {ch['energy']:.2f} vs in-distribution cut-off "
            f"{ch['energy_threshold']:.2f} ({'OUT of' if ch.get('energy_ood') else 'within'} distribution)."
        )
    txt += f" {m.get('warning') or 'No input-quality warnings.'}"
    return Answer(txt, "image_quality", 1.0)


def c_patient(ctx: Ctx) -> Answer:
    from app.copilot.rewrite import lay

    f = ctx.focus
    if f:
        e = entry(f["key"]) or entry(f["label"]) or {}
        txt = e.get("lay") or lay(e.get("definition", f["label"]))
        txt += (
            f" The computer estimated about a {round(f['probability'] * 100)}% chance for this on "
            "your image; a doctor will check the image and explain what it means for you."
        )
    else:
        sec = rewrite(_report_sections(ctx), "patient", "impression")
        txt = " ".join(sec["impression"])
    return Answer(txt, "patient_explanation", 1.0, f["label"] if f else None)


def c_check(ctx: Ctx) -> Answer:
    f = ctx.focus
    if not f:
        term = ctx.ent["terms"][0] if ctx.ent["terms"] else None
        if term:
            return Answer(
                f"The models in this pipeline do not output '{term}'.", "check_finding", 1.0
            )
        return _no_focus(ctx, "the check")
    if f.get("positive"):
        txt = (
            f"The model flags {f['label']}: {_desc(f)}. This suggests, but does not confirm, the "
            "finding."
        )
    elif f.get("ratio_to_threshold", 0) >= 0.7:
        txt = (
            f"{f['label']} is below threshold ({_pct(f['probability'])} vs {_pct(f['threshold'])}) "
            "but close to it — it cannot be confidently excluded."
        )
    else:
        txt = (
            f"{f['label']} is not flagged: calibrated probability {_pct(f['probability'])} is below "
            f"the {_pct(f['threshold'])} threshold. The AI cannot rule it out completely."
        )
    return Answer(f"{txt} {CORRELATE}", "check_finding", 1.0, f["label"], [_heat_action(f)])


def fallback(ctx: Ctx) -> Answer:
    hits = search(ctx.question, 1)
    if hits and hits[0][1] > 2.5:
        e = hits[0][0]
        return Answer(
            f"I'm not sure what you're asking, but this may help — {e['name']}: "
            f"{e['definition']}",
            "fallback",
            0.0,
            sources=[e["name"]],
            suggestions=FALLBACK_SUGGESTIONS,
        )
    return Answer(
        "I'm not confident I understood that. I can answer questions about this study's "
        "findings, locations, confidence, uncertainty, measurements, urgency, comparisons, "
        "definitions and report wording. Try one of the suggestions below.",
        "fallback",
        0.0,
        suggestions=FALLBACK_SUGGESTIONS,
    )


COMPOSERS: dict[str, Callable[[Ctx], Answer]] = {
    "greeting_help": c_greeting,
    "list_findings": c_list,
    "explain_finding": c_explain,
    "why_finding": c_why,
    "where_finding": c_where,
    "confidence": c_confidence,
    "next_steps": c_next,
    "compare_previous": c_compare,
    "define_term": c_define,
    "show_heatmap": c_heatmap,
    "measurements": c_measure,
    "normal_or_not": c_normal,
    "urgency": c_urgency,
    "limitations": c_limits,
    "rewrite_section": c_rewrite,
    "summary": c_summary,
    "differential": c_differential,
    "uncertainty": c_uncertainty,
    "model_info": c_model_info,
    "image_quality": c_quality,
    "patient_explanation": c_patient,
    "check_finding": c_check,
}


def suggestions_for(results: dict[str, Any], has_prior: bool) -> list[str]:
    pos = _positives(results)
    if pos:
        top = pos[0]["label"]
        s = [
            f"Why {top}?",
            f"Where is the {top.lower()}?",
            "How confident are you?",
            "What should we do next?",
        ]
    elif results.get("pipeline"):
        s = [
            "Is this scan normal?",
            "Show the measurements",
            "What are the limitations?",
            "Summarize the study",
        ]
    else:
        s = ["Why was this not analysed?", "What images are supported?"]
    if has_prior:
        s.insert(1, "Compare with the previous study")
    return s[:5]


def last_focus(session: Session | None, study_id: str) -> str | None:
    if session is None:
        return None
    msgs = session.exec(
        select(ChatMessage)
        .where(ChatMessage.study_id == study_id, ChatMessage.role == "assistant")
        .order_by(ChatMessage.created_at.desc())
    ).all()  # type: ignore[attr-defined]
    for m in msgs[:5]:
        meta = json.loads(m.actions_json or "{}")
        if isinstance(meta, dict) and meta.get("focus"):
            return str(meta["focus"])
    return None


def answer(question: str, study: Study, session: Session | None = None) -> Answer:
    results = study.results
    ent = E.extract(question)
    intent, conf, top3 = classify(question)
    # Entity-driven corrections for very short / ambiguous queries.
    if conf < 0.3 and ent["findings"] and len(question.split()) <= 3:
        intent, conf = "check_finding", max(conf, 0.3)
    focus_label = ent["findings"][0] if ent["findings"] else None
    if focus_label is None and intent not in ("define_term", "greeting_help"):
        focus_label = last_focus(session, study.id)
    focus = _find(results, focus_label)
    if focus is None and not ent["findings"] and intent in NEEDS_FOCUS:
        pos = _positives(results)
        focus = pos[0] if pos else None  # default to the top flagged finding
        focus_label = focus["label"] if focus else None
    if focus is None and ent["findings"]:
        focus_label = ent["findings"][0]
    ctx = Ctx(question, study, results, session, ent, focus, focus_label)
    if conf < 0.22:
        ans = fallback(ctx)
    else:
        ans = COMPOSERS.get(intent, fallback)(ctx)
        ans.confidence = conf
    ans.entities = {**ent, "intent_top3": top3}
    if not ans.suggestions:
        ans.suggestions = suggestions_for(results, _prior(ctx) is not None)
    from app.copilot.local_llm import polish

    ans.text = polish(ans.text, question)
    return ans
