"""Shared API helpers: errors, lookups, serialisers."""

from __future__ import annotations

import json
from typing import Any

from fastapi import HTTPException
from sqlmodel import Session, select

from app.models import Finding, Patient, Report, Study
from app.services import storage


class APIError(HTTPException):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(status_code=status, detail={"code": code, "message": message})


def get_study(session: Session, study_id: str) -> Study:
    s = session.get(Study, study_id)
    if s is None:
        raise APIError(404, "study_not_found", f"Study {study_id} not found")
    return s


def get_report(session: Session, report_id: str) -> Report:
    r = session.get(Report, report_id)
    if r is None:
        raise APIError(404, "report_not_found", f"Report {report_id} not found")
    return r


def study_out(session: Session, s: Study, full: bool = True) -> dict[str, Any]:
    p = session.get(Patient, s.patient_id)
    res = s.results
    rows = session.exec(
        select(Finding).where(Finding.study_id == s.id).order_by(Finding.rank)
    ).all()  # type: ignore[arg-type]
    status = {
        f.id: {"status": f.status, "edited_label": f.edited_label, "note": f.note} for f in rows
    }
    findings = res.get("findings", [])
    for f in findings:
        if f.get("id") in status:
            f["review"] = status[f["id"]]
    top = next((f for f in findings if f.get("positive")), None)
    out: dict[str, Any] = {
        "id": s.id,
        "patient": (
            {"id": p.id, "pseudonym": p.pseudonym, "sex": p.sex, "age": p.age} if p else None
        ),
        "filename": s.filename,
        "format": s.original_format,
        "modality": s.modality,
        "modality_hint": s.modality_hint,
        "status": s.status,
        "urgency_score": s.urgency_score,
        "urgency_level": s.urgency_level,
        "ood": s.ood,
        "model_profile": s.model_profile,
        "indication": s.indication,
        "study_date": s.study_date,
        "description": s.description,
        "image_url": storage.url(s.image_path),
        "width": s.width,
        "height": s.height,
        "pixel_spacing_mm": s.pixel_spacing_mm,
        "created_at": s.created_at.isoformat(),
        "analyzed_at": s.analyzed_at.isoformat() if s.analyzed_at else None,
        "top_finding": (
            {
                "label": top["label"],
                "probability": top["probability"],
                "certainty": top["certainty"],
            }
            if top
            else None
        ),
        "n_positive": sum(1 for f in findings if f.get("positive")),
        "pipeline": res.get("pipeline"),
        "warning": (res.get("modality") or {}).get("warning"),
    }
    if full:
        out["results"] = res
        out["deid"] = json.loads(s.deid_json or "{}")
        rep = session.exec(
            select(Report).where(Report.study_id == s.id).order_by(Report.updated_at.desc())
        ).first()  # type: ignore[attr-defined]
        out["report_id"] = rep.id if rep else None
        others = session.exec(
            select(Study).where(Study.patient_id == s.patient_id, Study.id != s.id)
        ).all()
        out["other_studies"] = [
            {
                "id": o.id,
                "created_at": o.created_at.isoformat(),
                "study_date": o.study_date,
                "modality": o.modality,
                "status": o.status,
            }
            for o in others
        ]
    return out


def report_out(r: Report) -> dict[str, Any]:
    return {
        "id": r.id,
        "study_id": r.study_id,
        "engine": r.engine,
        "style": r.style,
        "sections": r.sections,
        "status": r.status,
        "signed_by": r.signed_by,
        "signed_at": r.signed_at.isoformat() if r.signed_at else None,
        "version": r.version,
        "created_at": r.created_at.isoformat(),
        "updated_at": r.updated_at.isoformat(),
    }
