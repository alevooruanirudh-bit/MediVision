"""Tests for the paper-requirement additions (D-21 … D-30): new modalities, rationale, radiomics,
provenance / versioning, consent, roles + auth, critical alerts, hash-chained audit, DICOM export,
progression timeline, fusion, super-resolution, autonomy."""

from __future__ import annotations

import io

import numpy as np
import pytest
from conftest import (
    SAMPLES,
    needs_breast,
    needs_chest,
    needs_headct,
    needs_nodule,
    needs_retina,
)


# ---------------------------------------------------------------------------- data governance
def test_consent_required(client) -> None:
    p = SAMPLES / "chest_effusion.png"
    r = client.post("/api/studies", files={"file": (p.name, p.read_bytes())})
    assert r.status_code == 428 and r.json()["error"]["code"] == "consent_required"
    r = client.post(
        "/api/studies",
        files={"file": (p.name, p.read_bytes())},
        data={"consent_basis": "because", "analyze_now": "false"},
    )
    assert r.status_code == 422


def test_upload_records_hash_consent_and_uploader(client, uploaded) -> None:
    s = next(iter(uploaded.values()))
    full = client.get(f"/api/studies/{s['id']}").json()
    assert len(full["input_sha256"]) == 64
    assert full["consent_basis"] == "public_research_dataset"
    assert full["uploaded_by"]


# ---------------------------------------------------------------------------- roles / auth
def test_demo_roles_enforced(client, uploaded) -> None:
    sid = next(s["id"] for s in uploaded.values() if s["pipeline"])
    rep = client.get(f"/api/reports/by-study/{sid}").json()
    r = client.post(f"/api/reports/{rep['id']}/sign", json={}, headers={"X-Demo-Role": "viewer"})
    assert r.status_code == 403 and r.json()["error"]["code"] == "forbidden"
    r = client.post(f"/api/reports/{rep['id']}/sign", json={}, headers={"X-Demo-Role": "resident"})
    assert r.status_code == 403
    me = client.get("/api/auth/me", headers={"X-Demo-Role": "resident"}).json()
    assert me["role"] == "resident" and "report.sign" not in me["permissions"]
    assert "finding.review" in me["permissions"]


def test_auth_enabled_login_flow(client) -> None:
    from app.config import get_settings

    s = get_settings()
    s.auth_enabled = True
    try:
        assert client.get("/api/auth/me").status_code == 401
        bad = client.post("/api/auth/login", json={"username": "dr.lee", "password": "nope"})
        assert bad.status_code == 401
        ok = client.post(
            "/api/auth/login", json={"username": "dr.lee", "password": s.demo_password}
        ).json()
        h = {"Authorization": f"Bearer {ok['token']}"}
        me = client.get("/api/auth/me", headers=h).json()
        assert me["username"] == "dr.lee" and me["role"] == "radiologist" and me["authenticated"]
        # the demo header is ignored when auth is on
        assert client.get("/api/auth/me", headers={"X-Demo-Role": "admin"}).status_code == 401
        assert client.get("/api/auth/users", headers=h).status_code == 403  # admin only
        client.post("/api/auth/logout", headers=h)
        assert client.get("/api/auth/me", headers=h).status_code == 401
    finally:
        s.auth_enabled = False


# ---------------------------------------------------------------------------- audit chain
def test_audit_hash_chain_detects_tampering(client, uploaded) -> None:
    v = client.get("/api/audit/verify").json()
    assert v["valid"] and v["checked"] > 0
    from sqlmodel import Session, col, select

    from app.db import get_engine
    from app.models import AuditLog

    with Session(get_engine()) as s:
        row = s.exec(
            select(AuditLog).where(col(AuditLog.hash).is_not(None)).order_by(col(AuditLog.id))
        ).first()
        assert row is not None
        original = row.action
        row.action = "tampered"
        s.add(row)
        s.commit()
        bad = client.get("/api/audit/verify").json()
        assert not bad["valid"] and bad["broken_at"] == row.id
        row.action = original
        s.add(row)
        s.commit()
    assert client.get("/api/audit/verify").json()["valid"]


def test_analysis_audit_has_versions_and_outputs(client, uploaded) -> None:
    rows = client.get("/api/audit?action=study.analyze").json()
    d = rows[0]["details"]
    assert d["input_sha256"] and d["models"] and "outputs" in d and rows[0]["role"] == "ai"


# ---------------------------------------------------------------------------- transparency
@needs_chest
def test_rationale_versions_radiomics_chest(client, uploaded) -> None:
    r = client.get(f"/api/studies/{uploaded['chest_effusion.png']['id']}").json()["results"]
    f = r["findings"][0]
    assert f["rationale"]["reasons"] and f["rationale"]["reasons"][0]["kind"] == "score"
    assert f["provenance"]["source"] == "ai" and len(f["provenance"]["model_version"]) == 12
    assert f["provenance"]["model"] == "chest_ensemble"  # the diagnostic model, not the router
    names = {v["name"] for v in r["model_versions"]}
    assert {"router", "chest_densenet121"} <= names
    rad = r["radiomics"]
    assert rad["roi"] == "both lungs" and rad["texture_glcm"]["contrast"] >= 0
    assert set(rad["first_order"]) >= {"mean", "std", "entropy_bits"}


@needs_chest
def test_review_records_provenance_and_fhir(client, uploaded) -> None:
    sid = uploaded["chest_followup_prior.png"]["id"]
    st = client.get(f"/api/studies/{sid}").json()
    f = st["results"]["findings"][0]
    r = client.post(
        f"/api/findings/{f['id']}/review",
        json={"action": "edit", "new_label": "Mass"},
        headers={"X-Demo-Role": "resident"},
    ).json()
    assert r["reviewed_role"] == "resident"
    st = client.get(f"/api/studies/{sid}").json()
    rv = next(x for x in st["results"]["findings"] if x["id"] == f["id"])["review"]
    assert rv["source"] == "ai+radiologist" and rv["reviewed_by"]
    rep = client.get(f"/api/reports/by-study/{sid}").json()
    signed = client.post(f"/api/reports/{rep['id']}/sign", json={"signed_by": "Dr Test"}).json()
    assert signed["signed_role"] == "radiologist" and signed["drafted_by"].startswith("ai:")
    fhir = client.get(f"/api/reports/{rep['id']}/fhir").json()
    assert fhir["resultsInterpreter"][0]["display"].startswith("Dr Test")
    assert any(e["url"] == "urn:medivision:model-versions" for e in fhir["extension"])


# ---------------------------------------------------------------------------- new modalities
@needs_headct
def test_head_ct_hemorrhage(client, uploaded) -> None:
    s = uploaded["headct_hemorrhage.png"]
    r = client.get(f"/api/studies/{s['id']}").json()["results"]
    assert r["pipeline"] == "head_ct"
    f = r["findings"][0]
    assert f["label"] == "Intracranial hemorrhage" and 0 <= f["probability"] <= 1
    assert r["overlays"]["heatmaps"]  # Grad-CAM for the new model
    assert r["report_preview"]


@needs_breast
def test_breast_ultrasound(client, uploaded) -> None:
    r = client.get(f"/api/studies/{uploaded['breast_us_malignant.png']['id']}").json()["results"]
    assert r["pipeline"] == "breast_us"
    assert r["findings"][0]["label"] == "Suspicious breast lesion"
    assert "risk_category" in r["findings"][0]


@needs_retina
def test_fundus_grading(client, uploaded) -> None:
    r = client.get(f"/api/studies/{uploaded['fundus_dr.png']['id']}").json()["results"]
    assert r["pipeline"] == "fundus", r["modality"]
    f = r["findings"][0]
    assert f["label"] == "Referable diabetic retinopathy" and f["grade"]
    assert abs(sum(r["extra"]["fundus"]["grade_probs"].values()) - 1) < 1e-3


@needs_nodule
def test_ct_volume_nodule_volumetrics_and_3d(client, uploaded) -> None:
    s = uploaded["ct_nodule_volume.npy"]
    r = client.get(f"/api/studies/{s['id']}").json()["results"]
    assert r["pipeline"] == "ct_volume"
    m = r["measurements"]
    assert m["available"] and m["volume_mm3"] > 0 and m["equivalent_diameter_mm"] > 0
    assert len(r["overlays"]["volume"]["mip_frames"]) == 24
    assert set(r["overlays"]["volume"]["mpr"]) == {"axial", "coronal", "sagittal"}
    assert r["radiomics"]["shape"]["sphericity"] > 0
    assert r["findings"][0]["label"] == "Malignancy-suspicious lung nodule"


def test_volume_rejects_pickles_and_bad_shapes(client) -> None:
    buf = io.BytesIO()
    np.save(buf, np.zeros((4, 4)))
    r = client.post(
        "/api/studies",
        files={"file": ("flat.npy", buf.getvalue())},
        data={"consent_basis": "uploader_attested"},
    )
    assert r.status_code == 422
    buf = io.BytesIO()
    np.save(buf, np.array([{"a": 1}], dtype=object), allow_pickle=True)
    r = client.post(
        "/api/studies",
        files={"file": ("evil.npy", buf.getvalue())},
        data={"consent_basis": "uploader_attested"},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------- alerts, autonomy
def test_alerts_and_acknowledgement(client, uploaded) -> None:
    a = client.get("/api/alerts").json()
    wl = client.get("/api/triage/worklist").json()["items"]
    crit = [s for s in wl if s["urgency_level"] == "CRITICAL"]
    assert a["count"] == sum(1 for s in crit if s["alert"]["active"])
    if crit:
        sid = crit[0]["id"]
        assert wl[0]["alert"]["active"]  # pinned to the top
        r = client.post(f"/api/studies/{sid}/alert/ack", headers={"X-Demo-Role": "viewer"})
        assert r.status_code == 403
        ok = client.post(f"/api/studies/{sid}/alert/ack").json()
        assert not ok["alert"]["active"] and ok["alert"]["acknowledged_by"]


def test_autonomy_flags(client, uploaded) -> None:
    for s in uploaded.values():
        r = client.get(f"/api/studies/{s['id']}").json()
        au = r["results"].get("autonomy") or {}
        if not au.get("enabled"):
            continue
        assert au["requires_human_confirmation"] is True
        if r["urgency_level"] == "CRITICAL":
            assert au["action"] == "critical_alert"
        if au["action"] == "pre_cleared_normal":
            assert r["urgency_level"] == "NORMAL"


# ---------------------------------------------------------------------------- interoperability
@needs_chest
def test_dicom_exports(client, uploaded) -> None:
    import pydicom

    sid = uploaded["chest_effusion.png"]["id"]
    r = client.get(f"/api/studies/{sid}/dicom-sc")
    assert r.status_code == 200
    ds = pydicom.dcmread(io.BytesIO(r.content))
    assert ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.7" and ds.PatientIdentityRemoved == "YES"
    assert ds.Rows > 0 and ds.SamplesPerPixel == 3
    rep = client.get(f"/api/reports/by-study/{sid}").json()
    d = pydicom.dcmread(io.BytesIO(client.get(f"/api/reports/{rep['id']}/dicom").content))
    assert d.MIMETypeOfEncapsulatedDocument == "application/pdf"
    assert bytes(d.EncapsulatedDocument)[:4] == b"%PDF"


# ---------------------------------------------------------------------------- progression
@needs_chest
def test_patient_timeline(client, uploaded) -> None:
    pid = client.get(f"/api/studies/{uploaded['chest_followup_current.png']['id']}").json()[
        "patient_id"
    ]
    t = client.get(f"/api/patients/{pid}/timeline").json()
    assert t["n_studies"] >= 2 and t["trends"] and t["call"]
    assert all(len(x["series"]) == t["n_studies"] for x in t["trends"])


def test_progression_logic_unit() -> None:
    from app.inference.compare import _slope

    assert _slope([0, 30, 60], [0.1, 0.2, 0.3]) == pytest.approx(0.1 / 30)
    assert _slope([0], [0.5]) == 0.0


# ---------------------------------------------------------------------------- imaging
def test_fusion_and_superres(client, uploaded) -> None:
    ids = [s["id"] for s in uploaded.values() if s["pipeline"] in ("chest", "brain")]
    if len(ids) < 2:
        pytest.skip("need two images")
    f = client.post("/api/fusion", json={"base_id": ids[0], "overlay_id": ids[1]}).json()
    assert f["image"].startswith("data:image/png") and len(f["shift_px"]) == 2
    e = client.post("/api/enhance", json={"study_id": ids[0], "ops": {"superres": True}}).json()
    assert any("x2" in s for s in e["steps"]) and e["size"][0] > 0


def test_radiomics_unit_disk() -> None:
    import cv2

    from app.inference.radiomics import features

    img = np.zeros((128, 128), np.uint8)
    cv2.circle(img, (64, 64), 30, 200, -1)
    r = features(img, img > 0, "disk", spacing_mm=0.5)
    assert r["shape"]["circularity"] > 0.85 and r["shape"]["area_mm2"] > 0
    assert r["first_order"]["std"] < 1  # uniform inside the ROI


def test_rationale_unit() -> None:
    from app.inference.rationale import reasons

    f = {
        "label": "Cardiomegaly",
        "probability": 0.7,
        "threshold": 0.3,
        "positive": True,
        "uncertainty": 0.12,
        "zone": "cardiomediastinal region",
    }
    r = reasons(f, "chest", {"ctr": 0.45}, {"modality": {"confidence": 0.9}})
    texts = " ".join(x["text"] for x in r["reasons"])
    assert "above" in texts and "disagree" in texts and "does NOT support" in texts


def test_migration_is_idempotent() -> None:
    from app.db import migrate

    assert migrate() == []


@needs_chest
def test_reanalyse_after_review_keeps_history(client, uploaded) -> None:
    sid = uploaded["chest_effusion.png"]["id"]
    f = client.get(f"/api/studies/{sid}").json()["results"]["findings"][0]
    client.post(f"/api/findings/{f['id']}/review", json={"action": "accept"})
    r = client.post(f"/api/studies/{sid}/analyze")
    assert r.status_code == 200, r.text  # previously: FOREIGN KEY constraint failed
    assert client.get("/api/findings/reviews").json()["n"] >= 1


def test_preclear_requires_clear_margin_and_uncertainty() -> None:
    from app.config import get_settings
    from app.inference.pipeline import _autonomy

    s, tri, mod = get_settings(), {"level": "NORMAL"}, {"ood": False}
    near = [{"probability": 0.90, "threshold": 0.915, "uncertainty": 0.03}]
    no_unc = [{"probability": 0.05, "threshold": 0.6, "uncertainty": 0.0}]
    clear = [{"probability": 0.01, "threshold": 0.5, "uncertainty": 0.003}]
    assert _autonomy(s, tri, near, mod)["action"] == "none"
    assert _autonomy(s, tri, no_unc, mod)["action"] == "none"
    assert _autonomy(s, tri, clear, mod)["action"] == "pre_cleared_normal"
