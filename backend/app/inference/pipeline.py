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
from sqlmodel import Session, select

from app.config import get_settings
from app.inference import measurements as M
from app.inference import modalities as MOD
from app.inference import radiomics as RAD
from app.inference import triage, xai
from app.inference import volumes3d as V3
from app.inference.chest import analyze_chest
from app.inference.other import analyze_brain, analyze_fracture
from app.inference.rationale import reasons as rationale
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
    if pipeline in ("head_ct", "breast_us", "mammo"):
        return registry.get(MOD.PIPELINE_MODEL[pipeline]), 0, "sigmoid"
    if pipeline == "fundus":
        return registry.get("retina_dr"), 2, "softmax"  # moderate NPDR = referable threshold
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
    if img is None:
        img = storage.load_image(study.image_path, gray=pipe != "fundus")
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
    elif pipe == "head_ct":
        zone = _brain_region(hot["peak"])
    elif pipe == "breast_us":
        zone = "lesion region (image coordinates)"
    elif pipe == "mammo":
        zone = _breast_region(hot["peak"])
    elif pipe == "fundus":
        zone = _fundus_region(hot["peak"])
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


def _breast_region(peak: list[float]) -> str:
    """Coarse location on a screening view: upper / middle / lower third of the image."""
    y = float(peak[1]) if len(peak) > 1 else 0.5
    third = "upper" if y < 1 / 3 else "middle" if y < 2 / 3 else "lower"
    return f"{third} third of the view (image coordinates)"


def _fundus_region(peak: list[float]) -> str:
    x, y = peak
    r = ((x - 0.5) ** 2 + (y - 0.5) ** 2) ** 0.5
    if r < 0.12:
        return "central (macular / optic-disc) region"
    v = "superior" if y < 0.5 else "inferior"
    return (
        f"{v} {'temporal/nasal (image left)' if x < 0.5 else 'temporal/nasal (image right)'} retina"
    )


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
    vol_path = storage.study_dir(study.id) / "volume.npy"
    volume = np.load(vol_path) if vol_path.exists() else None
    pet_path = storage.study_dir(study.id) / "pet.npy"
    pet = np.load(pet_path) if pet_path.exists() else None
    vsp = json.loads(study.deid_json or "{}").get("voxel_spacing_mm")
    with T.stage("router_ood"):
        if volume is not None and pet is not None:
            modality = {
                "label": "pet_ct",
                "probs": {},
                "confidence": None,
                "pipeline": "petct",
                "ood": False,
                "warning": None if vsp else "No voxel spacing supplied - 5 mm isotropic assumed.",
            }
        elif volume is not None and V3.is_whole_scan(volume):
            modality = {
                "label": "chest_ct_scan",
                "probs": {},
                "confidence": None,
                "pipeline": "ct_chest_scan",
                "ood": False,
                "warning": (
                    None
                    if vsp
                    else "No voxel spacing supplied - 1 mm isotropic assumed; "
                    "nodule sizes may be wrong."
                ),
            }
        elif volume is not None:
            modality = {
                "label": "chest_ct_volume",
                "probs": {},
                "confidence": None,
                "pipeline": "ct_volume",
                "ood": False,
                "warning": (
                    None
                    if volume.shape == (64, 64, 64)
                    else f"Volume {tuple(volume.shape)} was resampled to 64³ for the nodule model."
                ),
            }
        else:
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
        elif pipeline == "head_ct":
            findings = MOD.analyze_head_ct(gray, settings.mc_dropout_samples)["findings"]
        elif pipeline == "breast_us":
            findings = MOD.analyze_breast_us(gray, settings.mc_dropout_samples)["findings"]
        elif pipeline == "fundus":
            fu = MOD.analyze_fundus(img)
            findings = fu["findings"]
            extra["fundus"] = {k: fu.get(k) for k in ("grade_probs", "predicted_grade")}
        elif pipeline == "mammo":
            findings = MOD.analyze_mammo(gray, settings.mc_dropout_samples)["findings"]
        elif pipeline == "ct_volume" and volume is not None:
            cv = MOD.analyze_volume(volume, tuple(vsp) if vsp else None)
            findings = cv["findings"]
        elif pipeline == "ct_chest_scan" and volume is not None:
            cs = V3.analyze_chest_scan(volume, tuple(vsp) if vsp else None)
            findings = cs["findings"]
            extra["chest_scan"] = {"detections": cs.get("detections", []), "error": cs.get("error")}
        elif pipeline == "petct" and volume is not None and pet is not None:
            pc = V3.analyze_petct(volume, pet, tuple(vsp) if vsp else None)
            findings = pc["findings"]
            if pc.get("error"):
                extra["petct"] = {"error": pc["error"]}
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
        elif pipeline == "ct_volume" and volume is not None:
            meas = cv["volumetrics"]
            v8, mask = cv["volume"], cv["mask"]
            np.save(storage.study_dir(study.id) / "volume64.npy", v8)
            np.save(storage.study_dir(study.id) / "mask64.npy", mask.astype(np.uint8))
            mpr = {}
            for plane, sl in MOD.mpr_slices(v8).items():
                rgb = np.stack([sl] * 3, -1)
                big = np.kron(rgb, np.ones((4, 4, 1), np.uint8))
                mpr[plane] = storage.url(
                    storage.save_rgba(
                        study.id,
                        f"mpr_{plane}",
                        np.dstack([big, np.full(big.shape[:2], 255, np.uint8)]),
                    )
                )
            frames = []
            for i, fr in enumerate(MOD.mip_frames(v8, mask)):
                big = np.kron(fr, np.ones((4, 4, 1), np.uint8))
                frames.append(
                    storage.url(
                        storage.save_rgba(
                            study.id,
                            f"mip_{i:02d}",
                            np.dstack([big, np.full(big.shape[:2], 255, np.uint8)]),
                        )
                    )
                )
            overlays["volume"] = {"mpr": mpr, "mip_frames": frames, "shape": list(v8.shape)}
            overlays["tumour_mask"] = storage.url(
                storage.save_rgba(
                    study.id,
                    "tumour_mask",
                    M.mask_overlay(
                        np.kron(mask[32].astype(np.uint8), np.ones((16, 16), np.uint8)),
                        (167, 139, 250),
                    ),
                )
            )
        elif pipeline == "ct_chest_scan" and volume is not None and "axial" in cs:
            meas = cs["measurements"]
            overlays["volume"] = _volume_overlay(study.id, cs["mpr"], cs["mip_frames"], 2)
            overlays["volume"]["key_image"] = storage.url(
                storage.save_rgba(study.id, "scan_key_axial", _rgba(cs["axial"]))
            )
        elif pipeline == "petct" and volume is not None and pet is not None and "mpr" in pc:
            meas = pc["measurements"]
            overlays["volume"] = _volume_overlay(study.id, pc["mpr"], [], 4)
            overlays["volume"]["pet_mip"] = storage.url(
                storage.save_rgba(
                    study.id, "pet_mip", _rgba(np.kron(pc["pet_mip"], np.ones((3, 3, 1), np.uint8)))
                )
            )
    with T.stage("radiomics"):
        radiomics: dict[str, Any] | None = None
        try:
            if pipeline == "ct_volume" and volume is not None:
                radiomics = RAD.features_3d(
                    cv["volume"], cv["mask"], tuple(meas.get("spacing_mm") or (1.0, 1.0, 1.0))
                )
            elif pipeline == "ct_chest_scan" and volume is not None and cs.get("top_crop"):
                crop, cmask = cs["top_crop"]
                radiomics = RAD.features_3d(crop, cmask, (1.0, 1.0, 1.0))
            elif pipeline == "petct" and pet is not None and pc.get("mask") is not None:
                z = pc["slice"]
                from ml.petct_fusion import norm_pet

                pet_u8 = (np.clip(norm_pet(pet[z]), 0, 1) * 255).astype(np.uint8)
                m = pc["mask"][z]
                radiomics = RAD.features(
                    pet_u8, m if m.any() else None, "FDG-avid lesion (PET)", None
                )
            elif pipeline == "chest" and lung is not None and "mask" in lung:
                radiomics = RAD.features(
                    gray, lung["mask"] > 0, "both lungs", study.pixel_spacing_mm
                )
            elif pipeline == "brain":
                tmask = M.tumour_measurements(gray, study.pixel_spacing_mm).get("mask")
                radiomics = RAD.features(
                    gray,
                    tmask > 0 if tmask is not None and tmask.any() else None,
                    "tumour (Attention U-Net mask)",
                    study.pixel_spacing_mm,
                )
            elif pipeline is not None:
                radiomics = RAD.features(
                    gray, None, pipeline.replace("_", " "), study.pixel_spacing_mm
                )
        except Exception as exc:  # pragma: no cover
            log.warning("radiomics failed: %s", exc)
    with T.stage("xai"):
        targets = [f for f in findings if f.get("positive")][:3] or findings[:1]
        if pipeline in ("ct_chest_scan", "petct"):
            targets = []
        if pipeline == "ct_volume":
            targets = []
            if volume is not None and cv.get("cam") is not None and findings:
                cam_sl = cv["cam"][32]
                big = np.kron(cam_sl, np.ones((16, 16), np.float32))
                url = storage.url(storage.save_rgba(study.id, "heat_nodule", xai.overlay(big)))
                overlays["heatmaps"][findings[0]["key"]] = {"gradcam": url}
                hot = xai.hottest(big)
                findings[0].update(
                    zone="axial mid-slice", heat_peak=hot["peak"], heat_box=hot["box"]
                )
        for f in targets:
            try:
                ex = explain(
                    study_with(study, pipeline),
                    f["key"],
                    "gradcam",
                    "inferno",
                    lung,
                    img if pipeline == "fundus" else gray,
                )
                overlays["heatmaps"][f["key"]] = {"gradcam": ex["url"]}
                f["zone"] = ex["zone"]
                f["heat_peak"] = ex["peak"]
                f["heat_box"] = ex["box"]
            except Exception as exc:  # pragma: no cover
                log.warning("xai failed for %s: %s", f["key"], exc)
    with T.stage("triage"):
        tri = triage.score(findings, ood=bool(modality.get("ood") or modality.get("energy_ood")))
    for f in findings:
        f["rationale"] = rationale(f, pipeline, meas, {"modality": modality})
    used = _models_used(pipeline)
    versions = [h.version_info() for n in used if (h := registry.get(n)) is not None]
    vmap = {v["name"]: v for v in versions}
    diag = MODELS_FOR.get(pipeline or "", [None])[0]
    if pipeline == "chest":
        members = [vmap[m]["version"] for m in MODELS_FOR["chest"][:3] if m in vmap]
        import hashlib

        diag = "chest_ensemble"
        vmap[diag] = {"version": hashlib.sha256("|".join(members).encode()).hexdigest()[:12]}
    for f in findings:
        mname = f.get("model") or diag
        f["provenance"] = {
            "source": "ai",
            "model": mname,
            "model_version": (vmap.get(mname) or {}).get("version"),
        }
    autonomy = _autonomy(settings, tri, findings, modality)
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
        "model_versions": versions,
        "radiomics": radiomics,
        "autonomy": autonomy,
        "backend": settings.inference_backend,
    }
    # persist findings (keep prior reviews only if labels are unchanged)
    # Findings a human already reviewed are kept (the reviews reference them — audit history) but
    # marked superseded; unreviewed findings of the previous run are replaced.
    from app.models import Review

    reviewed = set(session.exec(select(Review.finding_id).where(Review.study_id == study.id)).all())
    for old in session.exec(select(Finding).where(Finding.study_id == study.id)).all():
        if old.id in reviewed:
            old.status, old.positive, old.rank = "superseded", False, -1
            session.add(old)
        else:
            session.delete(old)
    session.flush()
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
            source="ai",
            model_name=f["provenance"]["model"],
            model_version=f["provenance"]["model_version"],
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


MODELS_FOR = {
    "chest": ["chest_densenet121", "chest_efficientnet_b2", "chest_convnext_tiny", "lung_unet"],
    "fracture": ["fracture_cls", "fracture_det"],
    "brain": ["brain_cls", "brain_seg"],
    "head_ct": ["head_ct_cls"],
    "breast_us": ["breast_us"],
    "fundus": ["retina_dr"],
    "ct_volume": ["nodule3d"],
    "mammo": ["mammo_cls"],
    "ct_chest_scan": ["lung_nodule_det", "nodule3d"],
    "petct": ["petct_seg"],
}
VOLUME_PIPELINES = {"ct_volume", "ct_chest_scan", "petct"}


def _rgba(rgb: np.ndarray) -> np.ndarray:
    return np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])


def _volume_overlay(
    study_id: str, mpr: dict[str, np.ndarray], frames: list[np.ndarray], scale: int
) -> dict[str, Any]:
    up = np.ones((scale, scale, 1), np.uint8)
    out: dict[str, Any] = {
        "mpr": {
            k: storage.url(storage.save_rgba(study_id, f"mpr_{k}", _rgba(np.kron(v, up))))
            for k, v in mpr.items()
        },
        "mip_frames": [
            storage.url(storage.save_rgba(study_id, f"mip_{i:02d}", _rgba(np.kron(f, up))))
            for i, f in enumerate(frames)
        ],
    }
    first = next(iter(mpr.values()), None)
    out["shape"] = list(first.shape[:2]) if first is not None else []
    return out


def _models_used(pipeline: str | None) -> list[str]:
    return ["router"] * (pipeline not in VOLUME_PIPELINES) + MODELS_FOR.get(pipeline or "", [])


def _autonomy(
    settings: Any, tri: dict[str, Any], findings: list[dict[str, Any]], modality: dict[str, Any]
) -> dict[str, Any]:
    """Autonomous mode (C10) for two well-defined tasks; a human must always confirm."""
    if not settings.autonomous_triage:
        return {"enabled": False}
    ood = bool(modality.get("ood") or modality.get("energy_ood"))
    max_unc = max((float(f.get("uncertainty") or 0) for f in findings), default=0.0)
    # Pre-clear only CLEAR negatives: every output far below its threshold (<= 40 %) AND every model
    # produced a real uncertainty estimate (uncertainty 0.0 = "not estimated", never "certain").
    clear = all(
        float(f["probability"]) <= 0.4 * max(float(f.get("threshold", 0.5)), 0.2)
        and float(f.get("uncertainty") or 0) > 0
        for f in findings
    )
    if (
        tri["level"] == "NORMAL"
        and not ood
        and findings
        and clear
        and max_unc <= settings.autonomous_max_uncertainty
    ):
        return {
            "enabled": True,
            "action": "pre_cleared_normal",
            "text": "AI pre-cleared as normal (no finding above threshold, low uncertainty). "
            "Moved to the confirm-normal queue — a radiologist must still confirm.",
            "requires_human_confirmation": True,
        }
    if tri["level"] == "CRITICAL":
        return {
            "enabled": True,
            "action": "critical_alert",
            "text": "Critical-finding alert raised automatically; the study is pinned to the top "
            "of the worklist until a radiologist acknowledges it.",
            "requires_human_confirmation": True,
        }
    return {"enabled": True, "action": "none", "requires_human_confirmation": True}


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
