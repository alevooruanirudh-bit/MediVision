"""Seed demo patients/studies (already analysed) from sample_data/.  `make seed` or POST /api/demo/load.

Creates ~12 studies: chest X-rays (incl. a follow-up pair for one patient so Compare works), bone
X-rays, brain MRIs and one non-medical photo to demonstrate the OOD guard.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlmodel import Session, select

from app.config import ROOT
from app.models import Finding, Study

SAMPLES = ROOT / "sample_data"
INDICATIONS = {
    "chest_followup_prior": "Cough and fever for one week.",
    "chest_followup_current": "Follow-up after treatment for pneumonia.",
    "chest_pneumothorax": "Sudden-onset pleuritic chest pain and breathlessness.",
    "chest_cardiomegaly_effusion": "Progressive dyspnoea and ankle swelling.",
    "chest_effusion": "Shortness of breath; reduced breath sounds at the right base.",
    "chest_cardiomegaly_edema": "Orthopnoea; known heart failure.",
    "bone_fracture": "Fall onto outstretched hand; pain and swelling.",
    "bone_no_fracture": "Hand pain after minor trauma.",
    "brain_glioma": "Headaches and new seizure.",
    "brain_meningioma": "Chronic headaches; incidental finding work-up.",
    "brain_notumor": "Dizziness; rule out structural lesion.",
    "not_medical_photo": "Uploaded by mistake (demonstrates the OOD guard).",
    "headct_hemorrhage": "Sudden severe headache; reduced consciousness.",
    "headct_normal": "Minor head injury; GCS 15.",
    "breast_us_malignant": "Palpable breast lump.",
    "fundus_dr": "Type 2 diabetes; annual retinal screening.",
    "ct_nodule_volume": "Incidental lung nodule on staging CT; characterise.",
    "mammo_cancer": "Screening mammogram, age 50-74 programme.",
    "mammo_normal": "Screening mammogram, age 50-74 programme.",
    "petct_sarcoma": "Biopsy-proven soft-tissue sarcoma of the extremity; staging FDG-PET/CT.",
    "chest_ct_scan": "Former smoker; low-dose CT lung-cancer screening.",
}
HINT = {
    "chest": "chest",
    "bone": "bone",
    "brain": "brain",
    "head_ct": "head_ct",
    "breast_us": "breast",
    "fundus": "fundus",
    "ct_volume": None,
    "mammography": "mammo",
    "petct": None,
    "ct_chest_scan": None,
    "none": None,
}


def seed(session: Session, reset: bool = False) -> dict[str, Any]:
    from app.api.studies import create_study, run_analysis

    manifest = json.loads((SAMPLES / "manifest.json").read_text())
    existing = {s.filename for s in session.exec(select(Study)).all()}
    created, skipped = [], []
    base = datetime.now(UTC) - timedelta(days=3)
    for i, m in enumerate(manifest):
        if m["file"] in existing and not reset:
            skipped.append(m["file"])
            continue
        role = m["file"].rsplit(".", 1)[0]
        date = base - timedelta(days=60) if role.endswith("prior") else base + timedelta(hours=i)
        s = create_study(
            session,
            (SAMPLES / m["file"]).read_bytes(),
            m["file"],
            patient_ref=m.get("patient_ref"),
            modality_hint=HINT.get(m["modality"]),
            indication=INDICATIONS.get(role),
            study_date=date.date().isoformat(),
            age=m.get("age"),
            sex=m.get("sex"),
            description=f"Demo: {m['source']} — ground truth: {m['ground_truth']}",
            consent_basis="public_research_dataset",
        )
        s.created_at = date
        session.add(s)
        session.commit()
        run_analysis(session, s)
        created.append(s.id)
    _demo_reviews(session)
    return {"created": len(created), "skipped": len(skipped), "study_ids": created}


def _demo_reviews(session: Session) -> None:
    """A few illustrative human reviews so the agreement KPI is populated (marked as demo)."""
    from app.models import Review

    if session.exec(select(Review)).first() is not None:
        return
    rows = session.exec(select(Finding).where(Finding.positive == True)).all()  # noqa: E712
    for j, f in enumerate(rows[:6]):
        action = "reject" if j % 4 == 3 else "accept"
        f.status = "rejected" if action == "reject" else "accepted"
        session.add(f)
        session.add(
            Review(
                finding_id=f.id,
                study_id=f.study_id,
                action=action,
                reviewer="demo radiologist",
                reviewer_role="radiologist",
                original_label=f.label,
                comment="Demo review (seed data)",
                model_probability=f.probability,
            )
        )
    session.commit()


def main() -> None:
    from app.db import get_engine, init_db
    from app.inference.registry import registry

    init_db()
    registry.load()
    with Session(get_engine()) as s:
        print(seed(s))


if __name__ == "__main__":
    main()
