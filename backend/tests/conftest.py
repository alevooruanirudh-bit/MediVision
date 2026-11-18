"""Test fixtures: isolated SQLite DB + storage per session, real trained models from artifacts/."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]
_TMP = Path(tempfile.mkdtemp(prefix="medivision-test-"))
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP / 'test.db'}"
os.environ["STORAGE_DIR"] = str(_TMP / "storage")
os.environ.setdefault("MC_DROPOUT_SAMPLES", "4")

SAMPLES = ROOT / "sample_data"
ARTIFACTS = ROOT / "artifacts"


def have(model: str) -> bool:
    return any(
        (ARTIFACTS / model / p / f).exists()
        for p in ("starter", "full")
        for f in ("model.onnx", "model.pt")
    )


needs_chest = pytest.mark.skipif(not have("chest_densenet121"), reason="chest models not trained")
needs_brain = pytest.mark.skipif(not have("brain_cls"), reason="brain model not trained")
needs_fracture = pytest.mark.skipif(not have("fracture_cls"), reason="fracture model not trained")
needs_headct = pytest.mark.skipif(not have("head_ct_cls"), reason="head-CT model not trained")
needs_nodule = pytest.mark.skipif(not have("nodule3d"), reason="nodule model not trained")
needs_breast = pytest.mark.skipif(not have("breast_us"), reason="breast model not trained")
needs_retina = pytest.mark.skipif(not have("retina_dr"), reason="retina model not trained")


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def uploaded(client):
    """Upload each sample once; returns {name: study_json}."""
    out = {}
    for name, hint in [
        ("chest_effusion.png", "chest"),
        ("bone_fracture.jpg", "bone"),
        ("brain_glioma.png", "brain"),
        ("not_medical_photo.png", None),
        ("chest_followup_prior.png", "chest"),
        ("chest_followup_current.png", "chest"),
        ("headct_hemorrhage.png", "head_ct"),
        ("breast_us_malignant.png", "breast"),
        ("fundus_dr.png", "fundus"),
        ("ct_nodule_volume.npy", None),
    ]:
        p = SAMPLES / name
        if not p.exists():
            continue
        data = {"modality_hint": hint} if hint else {}
        data["consent_basis"] = "public_research_dataset"
        if name.startswith("chest_followup"):
            data["patient_ref"] = "NIH-246"
        r = client.post("/api/studies", files={"file": (name, p.read_bytes())}, data=data)
        assert r.status_code == 200, r.text
        out[name] = r.json()
    return out
