"""Every API endpoint + end-to-end analysis of each sample modality (with the trained models)."""

from __future__ import annotations

import json
import time

import pytest
from conftest import SAMPLES, needs_brain, needs_chest, needs_fracture


def test_health_and_root(client) -> None:
    h = client.get("/api/system/health").json()
    assert h["status"] == "ok" and h["external_ai_apis"] == "none"
    assert "not a medical device" in h["disclaimer"]
    assert client.get("/").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    assert "X-MediVision-Disclaimer" in client.get("/").headers


def test_upload_validation(client) -> None:
    r = client.post("/api/studies", files={"file": ("notes.txt", b"hello")})
    assert r.status_code == 415 and r.json()["error"]["code"] == "unsupported_type"
    r = client.post(
        "/api/studies",
        files={"file": ("bad.png", b"not an image")},
        data={"consent_basis": "uploader_attested"},
    )
    assert r.status_code == 422
    assert client.get("/api/studies/does-not-exist").status_code == 404


@needs_chest
def test_chest_end_to_end(client, uploaded) -> None:
    s = uploaded["chest_effusion.png"]
    r = s["results"]
    assert s["status"] == "analyzed" and r["pipeline"] == "chest"
    assert r["modality"]["label"] == "chest_xray"
    assert len(r["findings"]) == 14
    f = r["findings"][0]
    assert {"probability", "uncertainty", "threshold", "certainty", "member_probs"} <= set(f)
    assert len(f["member_probs"]) == 3
    assert r["overlays"]["heatmaps"] and r["overlays"].get("lung_mask")
    assert r["measurements"]["available"] and r["measurements"].get("ctr")
    assert r["triage"]["level"] in ("CRITICAL", "URGENT", "ROUTINE", "NORMAL")
    assert r["timings_ms"]["total"] > 0


@needs_chest
def test_chest_latency_after_warmup(client, uploaded) -> None:
    sid = uploaded["chest_effusion.png"]["id"]
    t0 = time.perf_counter()
    r = client.post(f"/api/studies/{sid}/analyze")
    elapsed = time.perf_counter() - t0
    assert r.status_code == 200
    print(
        f"\n[latency] chest X-ray end-to-end (warm): {elapsed:.2f}s; stages {r.json()['results']['timings_ms']}"
    )
    assert elapsed < 5.0


@needs_fracture
def test_fracture_end_to_end(client, uploaded) -> None:
    r = uploaded["bone_fracture.jpg"]["results"]
    assert r["pipeline"] == "fracture" and r["findings"][0]["label"] == "Fracture"
    assert isinstance(r["detections"], list)


@needs_brain
def test_brain_end_to_end(client, uploaded) -> None:
    r = uploaded["brain_glioma.png"]["results"]
    assert r["pipeline"] == "brain"
    assert r["extra"]["brain"]["predicted"] in (
        "Glioma",
        "Meningioma",
        "No tumour",
        "Pituitary tumour",
    )
    assert "tumour_area_px" in r["measurements"]


def test_non_medical_photo_triggers_ood(client, uploaded) -> None:
    if "not_medical_photo.png" not in uploaded:
        pytest.skip("sample missing")
    s = uploaded["not_medical_photo.png"]
    assert s["ood"] is True
    assert s["results"]["pipeline"] is None and s["results"]["modality"]["warning"]
    assert s["results"]["findings"] == []


def test_list_get_worklist_dashboard(client, uploaded) -> None:
    lst = client.get("/api/studies").json()
    assert len(lst) >= len(uploaded)
    wl = client.get("/api/triage/worklist").json()
    order = {"CRITICAL": 0, "URGENT": 1, "ROUTINE": 2, "NORMAL": 3}
    lv = [order.get(i["urgency_level"], 4) for i in wl["items"]]
    assert lv == sorted(lv)
    d = client.get("/api/dashboard").json()
    assert d["total_studies"] >= len(uploaded)
    one = client.get(f"/api/studies/{lst[0]['id']}").json()
    assert "results" in one and "other_studies" in one


@needs_chest
def test_explain_methods(client, uploaded) -> None:
    sid = uploaded["chest_effusion.png"]["id"]
    for method in ("gradcam", "gradcampp", "occlusion"):
        r = client.get(
            f"/api/studies/{sid}/explain", params={"finding": "Effusion", "method": method}
        )
        assert r.status_code == 200, r.text
        e = r.json()
        assert e["url"].startswith("/files/") and len(e["box"]) == 4 and e["zone"]
        assert client.get(e["url"]).status_code == 200
    assert client.get(f"/api/studies/{sid}/explain", params={"finding": "Nope"}).status_code == 404


@needs_chest
def test_reports_lifecycle(client, uploaded) -> None:
    sid = uploaded["chest_effusion.png"]["id"]
    rep = client.get(f"/api/reports/by-study/{sid}").json()
    assert rep["engine"] == "rules" and rep["sections"]["impression"]
    gen = client.post(
        f"/api/reports/{sid}/generate", params={"engine": "rules", "style": "concise"}
    ).json()
    rid = gen["id"]
    secs = gen["sections"]
    secs["impression"] = ["1. Edited by reviewer."]
    upd = client.put(f"/api/reports/{rid}", json={"sections": secs}).json()
    assert upd["version"] == 2 and upd["sections"]["impression"] == ["1. Edited by reviewer."]
    rw = client.post(f"/api/reports/{rid}/rewrite", json={"style": "patient"}).json()
    assert rw["sections"]["impression"][0].startswith("In simple terms")
    regen = client.post(
        f"/api/reports/{rid}/rewrite", json={"style": "standard", "section": "impression"}
    ).json()
    assert regen["sections"]["impression"][0].startswith("1.")
    pdf = client.get(f"/api/reports/{rid}/pdf")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
    fhir = client.get(f"/api/reports/{rid}/fhir").json()
    assert fhir["resourceType"] == "DiagnosticReport" and fhir["status"] == "preliminary"
    assert len(fhir["contained"]) == 14
    signed = client.post(f"/api/reports/{rid}/sign", json={"signed_by": "Dr Test"}).json()
    assert signed["status"] == "signed"
    assert client.put(f"/api/reports/{rid}", json={"sections": secs}).status_code == 409
    assert client.get(f"/api/reports/{rid}/fhir").json()["status"] == "final"


@needs_chest
def test_experimental_report_engine(client, uploaded) -> None:
    from conftest import have

    if not have("report_gen"):
        pytest.skip("report generator not trained")
    sid = uploaded["chest_effusion.png"]["id"]
    r = client.post(f"/api/reports/{sid}/generate", params={"engine": "experimental"})
    assert r.status_code == 200, r.text
    s = r.json()["sections"]
    assert s["meta"]["engine"] == "experimental" and "consistency" in s["meta"]


@needs_chest
def test_copilot_chat_stream_and_history(client, uploaded) -> None:
    sid = uploaded["chest_effusion.png"]["id"]
    r = client.post(
        f"/api/copilot/{sid}/chat", json={"message": "what did you find?", "delay_ms": 0}
    )
    assert r.status_code == 200
    body = r.text
    assert "event: token" in body and "event: done" in body
    done = json.loads(
        body.split("event: done")[1].split("data: ", 1)[1].split("\r\n")[0].split("\n")[0]
    )
    assert done["intent"] == "list_findings" and done["suggestions"]
    plain = client.post(
        f"/api/copilot/{sid}/chat", json={"message": "how urgent is this", "stream": False}
    ).json()
    assert plain["intent"] == "urgency"
    h = client.get(f"/api/copilot/{sid}/history").json()
    assert len(h["messages"]) >= 4 and h["suggestions"]


@needs_chest
def test_findings_review_and_export(client, uploaded) -> None:
    s = client.get(f"/api/studies/{uploaded['chest_effusion.png']['id']}").json()
    fid = s["results"]["findings"][0]["id"]
    assert (
        client.post(f"/api/findings/{fid}/review", json={"action": "accept"}).json()["status"]
        == "accepted"
    )
    fid2 = s["results"]["findings"][1]["id"]
    assert (
        client.post(
            f"/api/findings/{fid2}/review", json={"action": "edit", "new_label": "Atelectasis"}
        ).json()["status"]
        == "edited"
    )
    assert client.post(f"/api/findings/{fid2}/review", json={"action": "edit"}).status_code == 422
    rv = client.get("/api/findings/reviews").json()
    assert rv["n"] >= 2 and 0 <= rv["agreement_rate"] <= 1
    csv = client.get("/api/findings/reviews/export").text
    assert csv.startswith("study_id,image_path") and "Atelectasis" in csv


@needs_chest
def test_compare(client, uploaded) -> None:
    a = uploaded["chest_followup_prior.png"]
    b = uploaded["chest_followup_current.png"]
    assert a["patient"]["pseudonym"] == b["patient"]["pseudonym"]
    r = client.post("/api/compare", json={"prior_id": a["id"], "current_id": b["id"]})
    assert r.status_code == 200, r.text
    c = r.json()
    assert len(c["deltas"]) == 14 and c["summary"].startswith("Compared with the prior study")
    assert c["heatmap_diff"].get("url")
    sid = uploaded["brain_glioma.png"]["id"] if "brain_glioma.png" in uploaded else None
    if sid:
        assert (
            client.post("/api/compare", json={"prior_id": a["id"], "current_id": sid}).status_code
            == 422
        )


@needs_chest
def test_enhance(client, uploaded) -> None:
    sid = uploaded["chest_effusion.png"]["id"]
    r = client.post(
        "/api/enhance",
        json={"study_id": sid, "ops": {"simulate_low_dose": {"photons": 60}, "denoise": True}},
    ).json()
    assert r["image"].startswith("data:image/png;base64,")
    assert r["metrics"]["psnr"] > r["metrics"]["psnr_noisy"]  # our denoiser improves PSNR
    r2 = client.post(
        "/api/enhance", json={"study_id": sid, "ops": {"clahe": {"clip": 2}, "gamma": 1.2}}
    ).json()
    assert "CLAHE" in r2["steps"]


def test_federated_stream(client) -> None:
    from ml.data.datasets import _medmnist_npz

    try:
        _medmnist_npz("pneumoniamnist", 64)
    except FileNotFoundError:
        pytest.skip("PneumoniaMNIST not downloaded")
    r = client.post("/api/federated/run", json={"rounds": 2})
    assert r.status_code == 200
    assert "event: setup" in r.text and "event: round" in r.text and "event: done" in r.text
    assert client.get("/api/federated/last").json()["rounds"] == 2


def test_metrics_model_card_audit(client, uploaded) -> None:
    m = client.get("/api/metrics").json()
    assert m["models"] and m["datasets"]
    card = client.get("/api/model-card").json()
    assert card["triage"]["weights"]["Pneumothorax"] == 1.0
    audit = client.get("/api/audit").json()
    actions = {a["action"] for a in audit}
    assert "study.upload" in actions and "study.analyze" in actions


def test_delete_study(client) -> None:
    p = SAMPLES / "brain_notumor.png"
    if not p.exists():
        pytest.skip("sample missing")
    s = client.post(
        "/api/studies",
        files={"file": (p.name, p.read_bytes())},
        data={"analyze_now": "false", "consent_basis": "uploader_attested"},
    ).json()
    assert s["status"] == "uploaded"
    assert client.delete(f"/api/studies/{s['id']}").json() == {"deleted": s["id"]}
    assert client.get(f"/api/studies/{s['id']}").status_code == 404
