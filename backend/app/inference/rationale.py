"""Plain-language reasons for every AI call (rule extraction over the model's own evidence).

Each finding gets a short list of `reasons` built from: probability vs. operating threshold,
ensemble / MC-dropout agreement, where the heatmap points, and whether independent measurements
(segmentation, CTR, detector, volumetrics) support or contradict the call. This is a transparent
summary of the evidence, not a causal explanation of the network.
"""

from __future__ import annotations

from typing import Any


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def reasons(
    f: dict[str, Any], pipeline: str | None, meas: dict[str, Any], extra: dict[str, Any]
) -> dict[str, Any]:
    p, t = float(f["probability"]), float(f.get("threshold", 0.5))
    r: list[dict[str, str]] = []
    if f.get("positive"):
        r.append(
            {
                "kind": "score",
                "text": f"Model probability {_pct(p)} is above this finding's operating threshold "
                f"{_pct(t)} ({p / max(t, 1e-6):.1f}× threshold).",
            }
        )
    else:
        r.append(
            {
                "kind": "score",
                "text": f"Model probability {_pct(p)} is below the operating threshold {_pct(t)}, "
                "so it is not flagged.",
            }
        )
    u = float(f.get("uncertainty") or 0.0)
    if u:
        r.append(
            {
                "kind": "agreement",
                "text": (
                    f"Ensemble / MC-dropout spread ±{_pct(u)} — the models agree closely."
                    if u < 0.05
                    else f"Ensemble / MC-dropout spread ±{_pct(u)} — models disagree; "
                    "treat with caution."
                ),
            }
        )
    if f.get("zone"):
        where = (
            f"Detected at {f['zone']}."
            if f.get("kind") in ("detection", "segmentation")
            else f"The heatmap's hottest region is the {f['zone']}."
        )
        r.append({"kind": "location", "text": where})
    lab = f["label"].lower()
    support = None
    if pipeline == "chest" and meas.get("ctr") is not None and "cardiomegaly" in lab:
        c = float(meas["ctr"])
        support = (
            f"Cardiothoracic ratio {c:.2f} is above 0.50, which supports the call."
            if c > 0.5
            else f"Cardiothoracic ratio {c:.2f} is ≤ 0.50, which does NOT support the call."
        )
    elif (
        pipeline == "chest"
        and meas.get("symmetry_index") is not None
        and ("effusion" in lab or "pneumothorax" in lab or "atelectasis" in lab)
    ):
        a = float(meas["symmetry_index"])
        support = (
            f"Lung-area symmetry index {a:.2f} (< 0.85) is consistent with a one-sided process."
            if a < 0.85
            else f"Lung areas are symmetric (index {a:.2f}); no volume asymmetry supports it."
        )
    elif pipeline == "brain" and "tumour_area_px" in meas and lab != "no tumour":
        support = (
            "The tumour segmenter outlines a lesion, which supports the classifier."
            if meas.get("tumour_area_px", 0) > 0
            else "The tumour segmenter finds no lesion — classifier and segmenter disagree."
        )
    elif pipeline == "fracture" and f.get("detector_score") is not None:
        support = f"The fracture detector also proposes a box (score {f['detector_score']:.2f})."
    elif pipeline == "ct_volume" and meas.get("volume_mm3") is not None:
        support = (
            f"Segmented nodule volume {meas['volume_mm3']:.0f} mm³ "
            f"(equivalent diameter {meas.get('equivalent_diameter_mm', 0):.1f} mm)."
        )
    elif pipeline == "ct_chest_scan" and (meas.get("top_nodule") or {}).get("diameter_mm"):
        tn = meas["top_nodule"]
        support = (
            f"Detected at z {tn['centre_mm_zyx'][0]} mm with an estimated diameter "
            f"{tn['diameter_mm']:.1f} mm ({meas.get('candidates_scored', 0)} candidates scored)."
        )
    elif pipeline == "petct" and meas.get("suv_max") is not None:
        support = (
            f"The fused PET+CT segmenter outlines a lesion with SUVmax {meas['suv_max']} and "
            f"metabolic volume {meas.get('mtv_ml')} ml."
        )
    elif pipeline == "mammo" and f.get("risk_category"):
        support = f"Single-view suspicion category: {f['risk_category']}."
    elif pipeline == "fundus" and f.get("grade"):
        support = f"Most probable ICDR grade: {f['grade']} (expected grade {f['expected_grade']})."
    if support:
        r.append({"kind": "measurement", "text": support})
    mod = (extra.get("modality") or {}).get("confidence")
    if mod is not None and mod < 0.6:
        r.append(
            {"kind": "input", "text": f"The modality router is unsure ({_pct(mod)}); verify input."}
        )
    summary = (
        f"Flagged because {r[0]['text'][0].lower() + r[0]['text'][1:]}"
        if f.get("positive")
        else r[0]["text"]
    )
    return {"reasons": r, "summary": summary}
