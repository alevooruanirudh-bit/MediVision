"""Reports: generate (rules | experimental), edit, rewrite, sign, PDF, FHIR."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from pydantic import BaseModel
from sqlmodel import Session, select

from app.api.deps import APIError, get_report, get_study, report_out
from app.copilot.report_writer import write_report
from app.copilot.rewrite import STYLES, rewrite
from app.db import get_session
from app.models import Patient, Report
from app.services import storage
from app.services.audit import audit
from app.services.auth import Principal, current_user, require
from app.services.exports import report_fhir, report_pdf

router = APIRouter(prefix="/api/reports", tags=["reports"])


class ReportUpdate(BaseModel):
    sections: dict[str, Any]


class SignRequest(BaseModel):
    signed_by: str | None = None  # display name; defaults to the signed-in user
    attest_reviewed: bool = True  # "I have reviewed the images and every AI finding"


class RewriteRequest(BaseModel):
    style: str = "concise"
    section: str | None = None


@router.post("/{study_id}/generate", summary="Generate a report draft")
def generate(
    study_id: str,
    engine: str = Query("rules", pattern="^(rules|experimental)$"),
    style: str = Query("standard"),
    session: Session = Depends(get_session),
    user: Principal = Depends(require("report.edit")),
) -> dict[str, Any]:
    s = get_study(session, study_id)
    if s.status != "analyzed":
        raise APIError(409, "not_analyzed", "Analyse the study first")
    if engine == "experimental":
        from app.inference import report_gen

        if s.results.get("pipeline") != "chest":
            raise APIError(422, "unsupported", "The experimental engine only supports chest X-rays")
        if not report_gen.available():
            raise APIError(503, "model_missing", "Experimental report model not trained")
        sections = report_gen.generate(storage.load_image(s.image_path), s.results)
    else:
        sections = write_report(s.results, s)
    if style != "standard":
        sections = rewrite(sections, style)
    rep = Report(
        study_id=s.id,
        engine=engine,
        style=style,
        sections_json=json.dumps(sections),
        drafted_by=f"ai:{'report-generator' if engine == 'experimental' else 'copilot-rules'}",
    )
    session.add(rep)
    session.commit()
    session.refresh(rep)
    audit(
        session,
        "report.generate",
        "report",
        rep.id,
        actor="ai",
        role="ai",
        requested_by=user.username,
        engine=engine,
        style=style,
    )
    return report_out(rep)


@router.get("/by-study/{study_id}", summary="Latest report for a study")
def latest(study_id: str, session: Session = Depends(get_session)) -> dict[str, Any]:
    rep = session.exec(
        select(Report).where(Report.study_id == study_id).order_by(Report.updated_at.desc())
    ).first()  # type: ignore[attr-defined]
    if rep is None:
        raise APIError(404, "report_not_found", "No report for this study")
    return report_out(rep)


@router.get("/{report_id}", summary="Get a report")
def get(report_id: str, session: Session = Depends(get_session)) -> dict[str, Any]:
    return report_out(get_report(session, report_id))


@router.put("/{report_id}", summary="Edit report sections")
def update(
    report_id: str,
    body: ReportUpdate,
    session: Session = Depends(get_session),
    user: Principal = Depends(require("report.edit")),
) -> dict[str, Any]:
    rep = get_report(session, report_id)
    if rep.status == "signed":
        raise APIError(409, "signed", "Signed reports are read-only")
    rep.sections_json = json.dumps(body.sections)
    rep.version += 1
    rep.updated_at = datetime.now(UTC)
    rep.edited_by = user.username
    session.add(rep)
    session.commit()
    session.refresh(rep)
    audit(
        session,
        "report.edit",
        "report",
        rep.id,
        actor=user.username,
        role=user.role,
        version=rep.version,
    )
    return report_out(rep)


@router.post("/{report_id}/rewrite", summary="Rewrite (concise / detailed / patient-friendly)")
def rewrite_report(
    report_id: str,
    body: RewriteRequest,
    session: Session = Depends(get_session),
    user: Principal = Depends(require("report.edit")),
) -> dict[str, Any]:
    rep = get_report(session, report_id)
    if body.style not in STYLES:
        raise APIError(422, "bad_style", f"style must be one of {STYLES}")
    if rep.status == "signed":
        raise APIError(409, "signed", "Signed reports are read-only")
    base = rep.sections
    if body.style == "standard" or body.section:
        s = get_study(session, rep.study_id)
        fresh = write_report(s.results, s) if rep.engine == "rules" else base
        if body.style == "standard":
            base = fresh if not body.section else {**base, body.section: fresh.get(body.section)}
    new = rewrite(base, body.style, body.section) if body.style != "standard" else base
    rep.sections_json = json.dumps(new)
    rep.style = body.style if not body.section else rep.style
    rep.version += 1
    rep.updated_at = datetime.now(UTC)
    session.add(rep)
    session.commit()
    session.refresh(rep)
    audit(
        session,
        "report.rewrite",
        "report",
        rep.id,
        actor=user.username,
        role=user.role,
        style=body.style,
        section=body.section,
    )
    return report_out(rep)


@router.post("/{report_id}/sign", summary="Sign off a report")
def sign(
    report_id: str,
    body: SignRequest,
    session: Session = Depends(get_session),
    user: Principal = Depends(require("report.sign")),
) -> dict[str, Any]:
    rep = get_report(session, report_id)
    if rep.status == "signed":
        raise APIError(409, "signed", "Report is already signed")
    if not body.attest_reviewed:
        raise APIError(422, "attestation_required", "Confirm you reviewed the images and findings")
    rep.status = "signed"
    rep.signed_by = body.signed_by or user.display_name or user.username
    rep.signed_role = user.role
    rep.signed_at = datetime.now(UTC)
    session.add(rep)
    session.commit()
    session.refresh(rep)
    audit(
        session,
        "report.sign",
        "report",
        rep.id,
        actor=user.username,
        role=user.role,
        display_name=rep.signed_by,
        version=rep.version,
        drafted_by=rep.drafted_by,
    )
    return report_out(rep)


def _ctx(session: Session, report_id: str) -> tuple[Report, Any, Patient]:
    rep = get_report(session, report_id)
    s = get_study(session, rep.study_id)
    p = session.get(Patient, s.patient_id)
    assert p is not None
    return rep, s, p


@router.get("/{report_id}/pdf", summary="Export PDF")
def pdf(
    report_id: str,
    session: Session = Depends(get_session),
    user: Principal = Depends(current_user),
) -> Response:
    rep, s, p = _ctx(session, report_id)
    audit(session, "report.export_pdf", "report", rep.id, actor=user.username, role=user.role)
    return Response(
        report_pdf(rep, s, p),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="medivision_{s.id}.pdf"'},
    )


@router.get("/{report_id}/fhir", summary="Export FHIR R4 DiagnosticReport JSON")
def fhir(
    report_id: str,
    session: Session = Depends(get_session),
    user: Principal = Depends(current_user),
) -> dict[str, Any]:
    rep, s, p = _ctx(session, report_id)
    audit(session, "report.export_fhir", "report", rep.id, actor=user.username, role=user.role)
    return report_fhir(rep, s, p, session)


@router.get("/{report_id}/dicom", summary="Export the report as a DICOM Encapsulated PDF")
def dicom_pdf(
    report_id: str,
    session: Session = Depends(get_session),
    user: Principal = Depends(current_user),
) -> Response:
    from app.services.dicom_export import encapsulated_pdf

    rep, s, p = _ctx(session, report_id)
    data = encapsulated_pdf(report_pdf(rep, s, p), rep, s, p)
    audit(session, "report.export_dicom", "report", rep.id, actor=user.username, role=user.role)
    return Response(
        data,
        media_type="application/dicom",
        headers={"Content-Disposition": f'attachment; filename="medivision_{s.id}_report.dcm"'},
    )
