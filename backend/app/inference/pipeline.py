"""Per-study analysis pipeline:
de-identify (at upload) -> modality router / OOD -> modality pipeline -> XAI -> measurements ->
triage -> Copilot report draft. Every stage is timed."""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sqlmodel import Session, delete, select

from app.config import get_settings
from app.inference import measurements as M
from app.inference import triage, xai
from app.inference.chest import analyze_chest
from app.inference.other import analyze_brain, analyze_fracture
from app.inference.registry import registry
from app.inference.router import route
from app.models import Finding, Study
from app.services import storage
from ml.common import BRAIN_CLASSES, NIH_LABELS

log = logging.getLogger("medivision.pipeline")


class Timer:
    def __init__(self) -> None:
        self.stages: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):  # type: ignore[no-untyped-def]
        t0 = time.perf_counter()
        yield
        self.stages[name] = round((time.perf_counter() - t0) * 1000, 1)


def cam_handle_and_index(pipeline: str, key: str) -> tuple[Any, int, str]:
    """Which of our PyTorch models explains a finding, and the output index to explain."""
    if pipeline == "chest":
        h = registry.get("chest_densenet121") or (registry.chest_members() or [None])[0]
        return h, NIH_LABELS.index(key), "sigmoid"
    if pipeline == "fracture":
        return registry.get("fracture_cls"), 0, "sigmoid"
    if pipeline == "brain":
        return registry.get("brain_cls"), BRAIN_CLASSES.index(key), "softmax"
    return None, 0, "sigmoid"


def explain(
    study: Study,
    key: str,
    method: str = "gradcam",
    cmap: str = "inferno",
    lung: dict[str, Any] | None = None,
    img: np.ndarray | None = None,
) -> dict[str, Any]:
    res = study.results
    pipe = res.get("pipeline")
    h, idx, act = cam_handle_and_index(pipe or "", key)
    if h is None:
        raise ValueError("no explainable model for this study")
    img = img if img is not None else storage.load_image(study.image_path)
    heat = (
        xai.occlusion(h, img, idx, activation=act)
        if method == "occlusion"
        else xai.cam(h, img, idx, method)
    )
    rgba = xai.overlay(heat, cmap)
    relp = storage.save_rgba(study.id, f"heat_{key}_{method}_{cmap}", rgba)
    np.save(
        storage.study_dir(study.id) / "overlays" / f"heat_{key}_{method}.npy",
        heat.astype(np.float16),
    )
    hot = xai.hottest(heat)
    zone = None
    if pipe == "chest":
        if lung is None:
            lung = _lung_from_disk(study, img)
        zone = M.zone_of(*hot["peak_px"], lung, img.shape[1], img.shape[0])
    elif pipe == "brain":
        zone = _brain_region(hot["peak"])
    elif pipe == "fracture":
        zone = _bone_region(hot["peak"])
    return {
        "finding": key,
        "method": method,
        "cmap": cmap,
        "url": storage.url(relp),
        "peak": hot["peak"],
        "box": hot["box"],
        "zone": zone,
        "model": h.name,
    }


def _lung_from_disk(study: Study, img: np.ndarray) -> dict[str, Any] | None:
    lung = M.lung_measurements(img, study.pixel_spacing_mm)
    return lung if lung.get("available") else None


def _brain_region(peak: list[float]) -> str:
    x, y = peak
    side = "left" if x < 0.5 else "right"  # radiological convention: image left = patient right
    side = "right" if side == "left" else "left"
    ap = "anterior" if y < 0.4 else "posterior" if y > 0.65 else "central"
    return f"{side} {ap} region"


def _bone_region(peak: list[float]) -> str:
    x, y = peak
    v = "proximal" if y < 0.33 else "distal" if y > 0.66 else "mid"
    hz = "medial" if 0.33 < x < 0.66 else "lateral"
    return f"{v}, {hz} aspect (image coordinates)"


def analyze(session: Session, study: Study, force: bool = False) -> dict[str, Any]:
    settings = get_settings()
    T = Timer()
    with T.stage("load"):
        img = storage.load_image(study.image_path, gray=False)
        gray = img if img.ndim == 2 else np.asarray(img).mean(-1).astype(np.uint8)
        if img.ndim == 3 and np.allclose(img[..., 0], img[..., 2], atol=3):
            img = gray
    deid = json.loads(study.deid_json or "{}")
    if deid.get("ms") is not None:
        T.stages["deidentify"] = deid["ms"]
    with T.stage("router_ood"):
        modality = route(img, study.modality_hint)
    pipeline = modality["pipeline"]
    if force and pipeline is None:
        hint = {"chest": "chest", "bone": "fracture", "fracture": "fracture", "brain": "brain"}
        pipeline = hint.get(study.modality_hint or "", "chest")
        modality["forced"] = True
    findings: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []
    meas: dict[str, Any] = {}
    overlays: dict[str, Any] = {"heatmaps": {}}
    extra: dict[str, Any] = {}
    lung = None
    with T.stage("model"):
        if pipeline == "chest":
            ch = analyze_chest(gray, settings.mc_dropout_samples)
            findings = ch["findings"]
            extra["chest"] = {
                k: ch.get(k)
                for k in (
                    "members",
                    "profile",
                    "energy",
                    "energy_threshold",
                    "energy_ood",
                    "img_sizes",
                )
            }
            if ch.get("energy_ood") and not modality["ood"]:
                modality["energy_ood"] = True
                modality["warning"] = (modality.get("warning") or "") + (
                    " Energy-score OOD check: image is unlike the chest training data — "
                    "interpret with extra caution."
                ).strip()
        elif pipeline == "fracture":
            fr = analyze_fracture(gray)
            findings, detections = fr["findings"], fr["detections"]
        elif pipeline == "brain":
            br = analyze_brain(gray)
            findings = br["findings"]
            extra["brain"] = {k: br.get(k) for k in ("class_probs", "predicted", "no_tumour_prob")}
    with T.stage("measurements"):
        if pipeline == "chest":
            lung = M.lung_measurements(gray, study.pixel_spacing_mm)
            if lung.get("available") and "mask" in lung:
                overlays["lung_mask"] = storage.url(
                    storage.save_rgba(
                        study.id, "lung_mask", M.mask_overlay(lung["mask"], (34, 211, 238))
                    )
                )
            meas = {k: v for k, v in lung.items() if not k.endswith("mask") and k != "zones"}
            if lung.get("quality") != "ok":
                lung = None
        elif pipeline == "brain":
            tm = M.tumour_measurements(gray, study.pixel_spacing_mm)
            if tm.get("available"):
                overlays["tumour_mask"] = storage.url(
                    storage.save_rgba(
                        study.id, "tumour_mask", M.mask_overlay(tm["mask"], (167, 139, 250))
                    )
                )
                tm.pop("mask")
                no_tumour = (extra.get("brain") or {}).get("predicted") == "No tumour"
                tm["consistent_with_classifier"] = bool(tm["tumour_area_px"] > 0) != no_tumour
            meas = tm
    with T.stage("xai"):
        targets = [f for f in findings if f.get("positive")][:3] or findings[:1]
        for f in targets:
            try:
                ex = explain(
                    study_with(study, pipeline), f["key"], "gradcam", "inferno", lung, gray
                )
                overlays["heatmaps"][f["key"]] = {"gradcam": ex["url"]}
                f["zone"] = ex["zone"]
                f["heat_peak"] = ex["peak"]
                f["heat_box"] = ex["box"]
            except Exception as exc:  # pragma: no cover
                log.warning("xai failed for %s: %s", f["key"], exc)
    with T.stage("triage"):
        tri = triage.score(findings, ood=bool(modality.get("ood") or modality.get("energy_ood")))
    profiles = {h.name: h.profile for h in registry.handles.values()}
    active_profile = "full" if any(p == "full" for p in profiles.values()) else "starter"
    results = {
        "pipeline": pipeline,
        "modality": modality,
        "findings": findings,
        "detections": detections,
        "measurements": meas,
        "overlays": overlays,
        "triage": tri,
        "extra": extra,
        "image": {
            "url": storage.url(study.image_path),
            "width": study.width,
            "height": study.height,
        },
        "model_profile": active_profile,
        "model_profiles": profiles,
        "backend": settings.inference_backend,
    }
    # persist findings (keep prior reviews only if labels are unchanged)
    session.exec(delete(Finding).where(Finding.study_id == study.id))  # type: ignore[arg-type]
    for i, f in enumerate(findings):
        row = Finding(
            study_id=study.id,
            label=f["label"],
            kind=f.get("kind", "classification"),
            probability=f["probability"],
            uncertainty=f.get("uncertainty", 0.0),
            threshold=f.get("threshold", 0.5),
            certainty=f.get("certainty", "Unlikely"),
            positive=bool(f.get("positive")),
            zone=f.get("zone"),
            box_json=json.dumps(f.get("box")) if f.get("box") else None,
            rank=i,
        )
        session.add(row)
        f["id"] = row.id
    with T.stage("report"):
        from app.copilot.report_writer import write_report

        results["report_preview"] = write_report(results, study)["impression"]
    T.stages["total"] = round(sum(v for k, v in T.stages.items() if k != "deidentify"), 1)
    results["timings_ms"] = T.stages
    study.results_json = json.dumps(results, default=_np)
    study.modality = modality["label"]
    study.ood = bool(modality.get("ood") or modality.get("energy_ood"))
    study.urgency_score = tri["score"]
    study.urgency_level = tri["level"]
    study.status = "analyzed"
    study.model_profile = active_profile
    study.analyzed_at = datetime.now(UTC)
    session.add(study)
    session.commit()
    session.refresh(study)
    return results


def study_with(study: Study, pipeline: str | None) -> Study:
    """Study view whose results carry the pipeline name (explain() needs it during analysis)."""
    s = Study(**{k: getattr(study, k) for k in Study.model_fields})
    s.results_json = json.dumps({"pipeline": pipeline})
    return s


def _np(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def findings_for(session: Session, study_id: str) -> list[Finding]:
    return list(
        session.exec(select(Finding).where(Finding.study_id == study_id).order_by(Finding.rank))
    )  # type: ignore[arg-type]
