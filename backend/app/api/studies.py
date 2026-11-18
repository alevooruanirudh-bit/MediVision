"""Studies: upload (PNG/JPG/DICOM, de-identified), list, get, delete, analyze, explain."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from sqlmodel import Session, col, select

from app.api.deps import APIError, get_study, study_out
from app.config import get_settings
from app.db import get_session
from app.inference.pipeline import analyze, explain
from app.models import ChatMessage, Finding, Patient, Report, Review, Study
from app.services import storage
from app.services.audit import audit
from app.services.auth import Principal, current_user, require
from app.services.deid import ingest

router = APIRouter(prefix="/api/studies", tags=["studies"])
ALLOWED = (".png", ".jpg", ".jpeg", ".dcm", ".dicom", ".npy", ".npz")
CONSENT_BASES = ("uploader_attested", "public_research_dataset", "waived_ethics_approval")


def create_study(
    session: Session,
    data: bytes,
    filename: str,
    *,
    patient_ref: str | None = None,
    modality_hint: str | None = None,
    indication: str | None = None,
    study_date: str | None = None,
    age: int | None = None,
    sex: str | None = None,
    description: str | None = None,
    consent_basis: str | None = None,
    user: Principal | None = None,
) -> Study:
    settings = get_settings()
    t0 = time.perf_counter()
    try:
        ing = ingest(data, filename, settings.pseudonym_salt, patient_ref)
    except Exception as exc:
        raise APIError(422, "unreadable_image", f"Could not read image: {exc}") from exc
    deid_ms = round((time.perf_counter() - t0) * 1000, 1)
    patient = session.exec(select(Patient).where(Patient.pseudonym == ing.pseudonym)).first()
    if patient is None:
        patient = Patient(pseudonym=ing.pseudonym, sex=sex or ing.sex, age=age or ing.age)
        session.add(patient)
        session.commit()
        session.refresh(patient)
    study = Study(
        patient_id=patient.id,
        filename=filename.rsplit("/", 1)[-1],
        original_format=ing.fmt,
        modality_hint=modality_hint,
        indication=indication,
        study_date=study_date,
        description=description,
        consent_basis=consent_basis,
        uploaded_by=user.username if user else "system",
        input_sha256=hashlib.sha256(data).hexdigest(),
    )
    relp, w, h, scale = storage.save_image(study.id, ing.pixels)
    study.image_path, study.width, study.height = relp, w, h
    if ing.pixel_spacing_mm:
        study.pixel_spacing_mm = ing.pixel_spacing_mm / scale
    study.deid_json = json.dumps(
        {
            "format": ing.fmt,
            "removed_tags": ing.removed_tags,
            "pseudonym": ing.pseudonym,
            "ms": deid_ms,
            "dicom_modality": ing.modality_tag,
            "view": ing.view,
            "exif_stripped": ing.fmt not in ("dcm", "npy"),
            "volume_shape": list(ing.volume.shape) if ing.volume is not None else None,
            "voxel_spacing_mm": ing.voxel_spacing_mm,
        }
    )
    if ing.volume is not None:
        np.save(storage.study_dir(study.id) / "volume.npy", ing.volume)
    if ing.pet is not None:
        np.save(storage.study_dir(study.id) / "pet.npy", ing.pet.astype(np.float32))
    if ing.deid_dataset is not None:
        ing.deid_dataset.save_as(str(storage.study_dir(study.id) / "deidentified.dcm"))
    session.add(study)
    session.commit()
    session.refresh(study)
    audit(
        session,
        "study.upload",
        "study",
        study.id,
        actor=user.username if user else "system",
        role=user.role if user else None,
        format=ing.fmt,
        phi_tags_removed=len(ing.removed_tags),
        pseudonym=ing.pseudonym,
        input_sha256=study.input_sha256,
        consent_basis=consent_basis,
    )
    return study


@router.post("", summary="Upload a study (PNG/JPG/DICOM ≤ 20 MB). DICOM is de-identified.")
async def upload(
    file: UploadFile = File(...),
    patient_ref: str | None = Form(None, description="Free-text patient reference (hashed)"),
    modality_hint: str | None = Form(
        None, description="chest | bone | brain | head_ct | breast | fundus"
    ),
    indication: str | None = Form(None),
    study_date: str | None = Form(None),
    age: int | None = Form(None),
    sex: str | None = Form(None),
    analyze_now: bool = Form(True),
    consent_basis: str | None = Form(
        None,
        description="Lawful basis for research use: uploader_attested | public_research_dataset "
        "| waived_ethics_approval (required when REQUIRE_CONSENT=true)",
    ),
    session: Session = Depends(get_session),
    user: Principal = Depends(require("study.write")),
) -> dict[str, Any]:
    name = (file.filename or "upload").lower()
    if not name.endswith(ALLOWED):
        raise APIError(415, "unsupported_type", f"Allowed: {', '.join(ALLOWED)}")
    if consent_basis is not None and consent_basis not in CONSENT_BASES:
        raise APIError(422, "bad_consent", f"consent_basis must be one of {CONSENT_BASES}")
    if get_settings().require_consent and not consent_basis:
        raise APIError(
            428,
            "consent_required",
            "Confirm that you may use this image for research (consent / lawful basis) and that "
            "it contains no identifiable data beyond what de-identification removes.",
        )
    limit = get_settings().max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise APIError(413, "too_large", f"Max upload size is {get_settings().max_upload_mb} MB")
    study = create_study(
        session,
        data,
        file.filename or "upload",
        patient_ref=patient_ref,
        modality_hint=modality_hint,
        indication=indication,
        study_date=study_date,
        age=age,
        sex=sex,
        consent_basis=consent_basis,
        user=user,
    )
    if analyze_now:
        run_analysis(session, study)
    return study_out(session, study)


def run_analysis(session: Session, study: Study, force: bool = False) -> dict[str, Any]:
    try:
        results = analyze(session, study, force)
    except Exception as exc:
        study.status = "error"
        session.add(study)
        session.commit()
        raise APIError(500, "analysis_failed", str(exc)) from exc
    from app.copilot.report_writer import write_report

    rep = Report(
        study_id=study.id, engine="rules", sections_json=json.dumps(write_report(results, study))
    )
    session.add(rep)
    session.commit()
    audit(
        session,
        "study.analyze",
        "study",
        study.id,
        actor="ai",
        role="ai",
        input_sha256=study.input_sha256,
        pipeline=results.get("pipeline"),
        models={v["name"]: v["version"] for v in results.get("model_versions", [])},
        outputs={
            f["label"]: round(float(f["probability"]), 4)
            for f in results.get("findings", [])
            if f.get("positive")
        },
        urgency=study.urgency_level,
        score=study.urgency_score,
        autonomy=(results.get("autonomy") or {}).get("action"),
        ms=results["timings_ms"].get("total"),
    )
    if (results.get("autonomy") or {}).get("action") == "critical_alert":
        audit(
            session,
            "alert.raised",
            "study",
            study.id,
            actor="ai",
            role="ai",
            drivers=[d["label"] for d in results["triage"]["drivers"][:3]],
        )
    return results


@router.get("", summary="List studies")
def list_studies(
    q: str | None = None,
    level: str | None = None,
    limit: int = 200,
    session: Session = Depends(get_session),
) -> list[dict[str, Any]]:
    rows = session.exec(select(Study).order_by(col(Study.created_at).desc()).limit(limit)).all()
    out = [study_out(session, s, full=False) for s in rows]
    if level:
        out = [s for s in out if (s["urgency_level"] or "").upper() == level.upper()]
    if q:
        ql = q.lower()
        out = [s for s in out if ql in json.dumps(s).lower()]
    return out


@router.get("/{study_id}", summary="Get a study with full results")
def get_one(study_id: str, session: Session = Depends(get_session)) -> dict[str, Any]:
    return study_out(session, get_study(session, study_id))


@router.delete("/{study_id}", summary="Delete a study and its files")
def delete(
    study_id: str,
    session: Session = Depends(get_session),
    user: Principal = Depends(require("study.delete")),
) -> dict[str, Any]:
    s = get_study(session, study_id)
    for model in (Finding, Review, Report, ChatMessage):
        for row in session.exec(select(model).where(model.study_id == study_id)).all():  # type: ignore[attr-defined]
            session.delete(row)
    session.delete(s)
    session.commit()
    storage.delete_study_files(study_id)
    audit(session, "study.delete", "study", study_id, actor=user.username, role=user.role)
    return {"deleted": study_id}


@router.post("/{study_id}/analyze", summary="(Re)run the full AI pipeline")
def analyze_study(
    study_id: str,
    force: bool = Query(False, description="analyse even if OOD"),
    session: Session = Depends(get_session),
    user: Principal = Depends(require("study.write")),
) -> dict[str, Any]:
    s = get_study(session, study_id)
    s.alert_ack_by, s.alert_ack_at = None, None
    run_analysis(session, s, force)
    audit(
        session, "study.reanalyze", "study", s.id, actor=user.username, role=user.role, force=force
    )
    return study_out(session, s)


@router.post("/{study_id}/alert/ack", summary="Acknowledge a critical-finding alert")
def ack_alert(
    study_id: str,
    session: Session = Depends(get_session),
    user: Principal = Depends(require("alert.ack")),
) -> dict[str, Any]:
    s = get_study(session, study_id)
    s.alert_ack_by, s.alert_ack_at = user.username, datetime.now(UTC)
    session.add(s)
    session.commit()
    audit(session, "alert.ack", "study", s.id, actor=user.username, role=user.role)
    return study_out(session, s)


@router.get("/{study_id}/dicom-sc", summary="Export image + AI overlay as DICOM Secondary Capture")
def dicom_sc(
    study_id: str,
    finding: str | None = None,
    session: Session = Depends(get_session),
    user: Principal = Depends(current_user),
):  # type: ignore[no-untyped-def]
    from fastapi.responses import Response

    from app.services.dicom_export import secondary_capture

    s = get_study(session, study_id)
    if s.status != "analyzed":
        raise APIError(409, "not_analyzed", "Analyse the study first")
    data = secondary_capture(session, s, finding)
    audit(session, "export.dicom_sc", "study", s.id, actor=user.username, role=user.role)
    return Response(
        data,
        media_type="application/dicom",
        headers={"Content-Disposition": f'attachment; filename="medivision_{s.id}_sc.dcm"'},
    )


@router.get("/{study_id}/explain", summary="Explainability overlay for one finding")
def explain_study(
    study_id: str,
    finding: str,
    method: str = Query("gradcam", pattern="^(gradcam|gradcampp|occlusion)$"),
    cmap: str = "inferno",
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    s = get_study(session, study_id)
    if s.status != "analyzed":
        raise APIError(409, "not_analyzed", "Analyse the study first")
    keys = {f["key"]: f for f in s.results.get("findings", [])}
    key = (
        finding
        if finding in keys
        else next((k for k, f in keys.items() if f["label"].lower() == finding.lower()), None)
    )
    if key is None:
        raise APIError(404, "finding_not_found", f"No finding '{finding}' in this study")
    t0 = time.perf_counter()
    try:
        out = explain(s, key, method, cmap)
    except ValueError as exc:
        raise APIError(422, "not_explainable", str(exc)) from exc
    out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    audit(session, "study.explain", "study", s.id, actor="ai", finding=key, method=method)
    return out
