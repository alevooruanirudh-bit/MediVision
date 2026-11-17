"""Studies: upload (PNG/JPG/DICOM, de-identified), list, get, delete, analyze, explain."""

from __future__ import annotations

import json
import time
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from sqlmodel import Session, col, select

from app.api.deps import APIError, get_study, study_out
from app.config import get_settings
from app.db import get_session
from app.inference.pipeline import analyze, explain
from app.models import ChatMessage, Finding, Patient, Report, Review, Study
from app.services import storage
from app.services.audit import audit
from app.services.deid import ingest

router = APIRouter(prefix="/api/studies", tags=["studies"])
ALLOWED = (".png", ".jpg", ".jpeg", ".dcm", ".dicom")


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
            "exif_stripped": ing.fmt != "dcm",
        }
    )
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
        format=ing.fmt,
        phi_tags_removed=len(ing.removed_tags),
        pseudonym=ing.pseudonym,
    )
    return study


@router.post("", summary="Upload a study (PNG/JPG/DICOM ≤ 20 MB). DICOM is de-identified.")
async def upload(
    file: UploadFile = File(...),
    patient_ref: str | None = Form(None, description="Free-text patient reference (hashed)"),
    modality_hint: str | None = Form(None, description="chest | bone | brain"),
    indication: str | None = Form(None),
    study_date: str | None = Form(None),
    age: int | None = Form(None),
    sex: str | None = Form(None),
    analyze_now: bool = Form(True),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    name = (file.filename or "upload").lower()
    if not name.endswith(ALLOWED):
        raise APIError(415, "unsupported_type", f"Allowed: {', '.join(ALLOWED)}")
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
        urgency=study.urgency_level,
        ms=results["timings_ms"].get("total"),
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
def delete(study_id: str, session: Session = Depends(get_session)) -> dict[str, Any]:
    s = get_study(session, study_id)
    for model in (Finding, Review, Report, ChatMessage):
        for row in session.exec(select(model).where(model.study_id == study_id)).all():  # type: ignore[attr-defined]
            session.delete(row)
    session.delete(s)
    session.commit()
    storage.delete_study_files(study_id)
    audit(session, "study.delete", "study", study_id)
    return {"deleted": study_id}


@router.post("/{study_id}/analyze", summary="(Re)run the full AI pipeline")
def analyze_study(
    study_id: str,
    force: bool = Query(False, description="analyse even if OOD"),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    s = get_study(session, study_id)
    run_analysis(session, s, force)
    return study_out(session, s)


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
