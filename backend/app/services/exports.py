"""Report exports: PDF (reportlab) and FHIR R4-style DiagnosticReport JSON."""

from __future__ import annotations

import io
from datetime import UTC, datetime
from typing import Any

from app.config import APP_VERSION, DISCLAIMER
from app.copilot.report_writer import SECTION_ORDER, SECTION_TITLES
from app.models import Patient, Report, Study
from app.services import storage

LOINC_BY_PIPE = {
    "chest": ("36643-5", "XR Chest 2V"),
    "fracture": ("37637-5", "XR Bone"),
    "brain": ("24590-2", "MR Brain"),
}


def report_pdf(report: Report, study: Study, patient: Patient) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"MediVision report {study.id}",
    )
    ss = getSampleStyleSheet()
    h = ParagraphStyle(
        "h", parent=ss["Heading3"], textColor=colors.HexColor("#0e7490"), spaceAfter=3
    )
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13)
    warn = ParagraphStyle("w", parent=body, textColor=colors.HexColor("#b45309"), fontSize=8.5)
    els: list[Any] = [
        Paragraph("<b>MediVision AI</b> — AI-assisted imaging report (draft)", ss["Title"]),
        Paragraph(DISCLAIMER, warn),
        Spacer(1, 4 * mm),
    ]
    info = [
        ["Patient (pseudonym)", patient.pseudonym, "Study", study.id],
        [
            "Modality",
            (study.modality or "").replace("_", " "),
            "Urgency",
            f"{study.urgency_level or '-'} ({study.urgency_score or 0:.0f})",
        ],
        [
            "Status",
            report.status + (f" by {report.signed_by}" if report.signed_by else ""),
            "Engine",
            f"{report.engine} / {report.style}",
        ],
    ]
    t = Table(info, colWidths=[35 * mm, 50 * mm, 25 * mm, 50 * mm])
    t.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.grey),
                ("TEXTCOLOR", (2, 0), (2, -1), colors.grey),
                ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.lightgrey),
            ]
        )
    )
    els += [t, Spacer(1, 4 * mm)]
    img_path = storage.abs_path(study.image_path) if study.image_path else None
    if img_path and img_path.exists():
        ratio = (study.height or 1) / max(study.width or 1, 1)
        els += [Image(str(img_path), width=60 * mm, height=60 * mm * ratio), Spacer(1, 3 * mm)]
    sec = report.sections
    for key in SECTION_ORDER:
        if key not in sec or key == "disclaimer":
            continue
        els.append(Paragraph(SECTION_TITLES[key], h))
        v = sec[key]
        if isinstance(v, dict):
            for k, val in v.items():
                els.append(Paragraph(f"<b>{k}:</b> {val}", body))
        elif isinstance(v, list):
            for val in v:
                els.append(Paragraph(str(val), body))
        else:
            els.append(Paragraph(str(v), body))
    els += [
        Spacer(1, 5 * mm),
        Paragraph(sec.get("disclaimer", DISCLAIMER), warn),
        Paragraph(
            f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M UTC} · MediVision AI "
            f"v{APP_VERSION} · all models self-trained, fully local inference",
            warn,
        ),
    ]
    doc.build(els)
    return buf.getvalue()


def report_fhir(report: Report, study: Study, patient: Patient) -> dict[str, Any]:
    res = study.results
    pipe = res.get("pipeline") or "chest"
    code, display = LOINC_BY_PIPE.get(pipe, ("18748-4", "Diagnostic imaging study"))
    observations = []
    for i, f in enumerate(res.get("findings", [])):
        observations.append(
            {
                "resourceType": "Observation",
                "id": f"obs-{i}",
                "status": "preliminary",
                "code": {"text": f["label"]},
                "valueQuantity": {"value": f["probability"], "unit": "probability"},
                "interpretation": [{"text": f["certainty"]}],
                "note": [
                    {
                        "text": f"threshold {f.get('threshold')}; uncertainty {f.get('uncertainty')}"
                        + (f"; location {f['zone']}" if f.get("zone") else "")
                    }
                ],
                "method": {"text": "MediVision self-trained model (AI-generated, requires review)"},
            }
        )
    sec = report.sections
    return {
        "resourceType": "DiagnosticReport",
        "id": report.id,
        "meta": {"tag": [{"system": "urn:medivision", "code": "ai-generated-draft"}]},
        "status": "final" if report.status == "signed" else "preliminary",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                        "code": "RAD",
                        "display": "Radiology",
                    }
                ]
            }
        ],
        "code": {"coding": [{"system": "http://loinc.org", "code": code, "display": display}]},
        "subject": {"reference": f"Patient/{patient.pseudonym}", "display": patient.pseudonym},
        "effectiveDateTime": (study.study_date or study.created_at.isoformat()),
        "issued": report.updated_at.isoformat(),
        "performer": [{"display": report.signed_by or "MediVision AI (unsigned draft)"}],
        "imagingStudy": [{"reference": f"ImagingStudy/{study.id}"}],
        "contained": observations,
        "result": [{"reference": f"#{o['id']}"} for o in observations],
        "conclusion": " ".join(sec.get("impression", [])),
        "extension": [
            {
                "url": "urn:medivision:urgency",
                "valueString": f"{study.urgency_level} ({study.urgency_score})",
            },
            {"url": "urn:medivision:model-profile", "valueString": res.get("model_profile", "")},
            {"url": "urn:medivision:disclaimer", "valueString": DISCLAIMER},
        ],
    }
