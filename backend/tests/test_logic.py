"""Ensemble + uncertainty math, triage scoring, DICOM PHI removal, report writer (golden), rewrite."""

from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pytest

from app.copilot.report_writer import to_text, write_report
from app.copilot.rewrite import rewrite
from app.inference.chest import certainty, joint_energy, mc_dropout_head
from app.inference.triage import score


def _f(label: str, p: float, t: float = 0.2, u: float = 0.02, zone: str | None = None) -> dict:
    return {
        "label": label,
        "key": label,
        "probability": p,
        "threshold": t,
        "uncertainty": u,
        "positive": p >= t,
        "certainty": certainty(p, t),
        "ratio_to_threshold": p / t,
        "zone": zone,
        "kind": "classification",
    }


# ----------------------------------------------------------------------------- ensemble math
def test_mc_dropout_zero_rate_has_no_variance() -> None:
    rng = np.random.default_rng(0)
    head = {
        "w1": rng.normal(size=(8, 4)),
        "b1": np.zeros(8),
        "w2": rng.normal(size=(3, 8)),
        "b2": np.zeros(3),
        "dropout": np.array(0.0),
    }
    out = mc_dropout_head(rng.normal(size=4), head, 6, rng)
    assert out.shape == (6, 3)
    assert np.allclose(out.var(0), 0)
    head["dropout"] = np.array(0.5)
    assert mc_dropout_head(rng.normal(size=4), head, 16, rng).var(0).sum() > 0


def test_certainty_wording_and_energy() -> None:
    assert certainty(0.7, 0.3) == "Likely"
    assert certainty(0.35, 0.3) == "Possible"
    assert certainty(0.1, 0.3) == "Unlikely"
    assert joint_energy(np.array([5.0, 5.0])) > joint_energy(np.array([-5.0, -5.0]))


# ----------------------------------------------------------------------------- triage
def test_triage_levels() -> None:
    assert score([_f("Effusion", 0.05)])["level"] == "NORMAL"
    assert score([_f("Effusion", 0.05)], ood=True)["level"] == "ROUTINE"
    assert score([_f("Pneumothorax", 0.95, 0.2, 0.0)])["level"] == "CRITICAL"
    low = score([_f("Hernia", 0.25)])
    assert low["level"] in ("ROUTINE", "URGENT") and low["score"] < 45


def test_triage_monotonic_in_probability_and_uncertainty() -> None:
    a = score([_f("Mass", 0.4)])["score"]
    b = score([_f("Mass", 0.8)])["score"]
    c = score([_f("Mass", 0.8, u=0.3)])["score"]
    assert b > a and c < b


# ----------------------------------------------------------------------------- DICOM de-identification
def _dicom_bytes() -> bytes:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset("x.dcm", {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientName = "Doe^Jane"
    ds.PatientID = "MRN-123456"
    ds.PatientBirthDate = "19700101"
    ds.InstitutionName = "General Hospital"
    ds.ReferringPhysicianName = "Smith^John"
    ds.AccessionNumber = "ACC42"
    ds.PatientSex = "F"
    ds.PatientAge = "054Y"
    ds.Modality = "CR"
    ds.PixelSpacing = [0.15, 0.15]
    ds.Rows, ds.Columns = 64, 64
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.PixelData = (np.random.rand(64, 64) * 4000).astype(np.uint16).tobytes()
    ds.add_new(0x00091001, "LO", "private vendor data")
    buf = io.BytesIO()
    pydicom.dcmwrite(buf, ds, enforce_file_format=True)
    return buf.getvalue()


def test_dicom_phi_removed_and_pseudonymised() -> None:
    from app.services.deid import ingest

    data = _dicom_bytes()
    a = ingest(data, "scan.dcm", "salt")
    b = ingest(data, "scan.dcm", "salt")
    assert a.fmt == "dcm" and a.pixels.shape == (64, 64) and a.pixels.dtype == np.uint8
    assert a.pseudonym == b.pseudonym and a.pseudonym.startswith("PT-")
    assert "MRN-123456" not in a.pseudonym
    for tag in (
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "InstitutionName",
        "ReferringPhysicianName",
        "AccessionNumber",
    ):
        assert tag in a.removed_tags
        assert tag not in a.deid_dataset
    assert a.deid_dataset.PatientIdentityRemoved == "YES"
    assert not any(e.tag.is_private for e in a.deid_dataset)
    assert a.age == 54 and a.sex == "F" and a.pixel_spacing_mm == pytest.approx(0.15)
    assert ingest(data, "scan.dcm", "other-salt").pseudonym != a.pseudonym


# ----------------------------------------------------------------------------- report writer (golden)
STUDY = SimpleNamespace(id="golden-study", indication="Breathlessness.")


def _results(findings: list[dict], ctr: float | None = 0.46) -> dict:
    return {
        "pipeline": "chest",
        "findings": findings,
        "modality": {"warning": None},
        "measurements": {
            "available": True,
            "ctr": ctr,
            "ctr_flag": bool(ctr and ctr > 0.5),
            "right_lung_area_px": 1000,
            "left_lung_area_px": 900,
            "symmetry_index": 0.9,
        },
        "extra": {"chest": {"members": ["a", "b", "c"]}},
        "model_profile": "starter",
        "backend": "onnx",
        "detections": [],
    }


def test_report_normal_chest_golden() -> None:
    rep = write_report(_results([_f("Effusion", 0.05), _f("Pneumothorax", 0.01)]), STUDY)
    assert rep["indication"] == "Breathlessness."
    assert rep["impression"][0].startswith("1. No acute cardiopulmonary")
    assert "pneumothorax" in rep["findings"]["Pleura"].lower()
    assert "CTR 0.46" in rep["findings"]["Heart and mediastinum"]
    assert rep["measurements"][0] == "Cardiothoracic ratio (estimate): 0.46 — within normal range"
    assert "not a medical device" in rep["disclaimer"]


def test_report_positive_findings_are_urgency_ordered_and_located() -> None:
    fs = [
        _f("Atelectasis", 0.5, 0.2, zone="left lower zone"),
        _f("Pneumothorax", 0.9, 0.2, zone="right upper zone"),
        _f("Mass", 0.15, 0.2),
    ]  # below threshold but >= 0.7x -> "cannot be excluded"
    rep = write_report(_results(fs, ctr=0.56), STUDY)
    imp = rep["impression"]
    assert "pneumothorax" in imp[0].lower() and "right upper zone" in imp[0]
    assert "atelectasis" in imp[1].lower()
    assert any("cannot be excluded" in i for i in imp)
    assert any("0.56" in i for i in imp)  # CTR > 0.5 surfaced
    assert "AI p = 0.90" in rep["findings"]["Pleura"]
    assert any("Pneumothorax" in r for r in rep["recommendations"])
    assert write_report(_results(fs, ctr=0.56), STUDY) == {
        **rep,
        "meta": {
            **rep["meta"],
            "generated_at": write_report(_results(fs, 0.56), STUDY)["meta"]["generated_at"],
        },
    }
    assert "IMPRESSION" in to_text(rep)


def test_rewrite_styles() -> None:
    rep = write_report(_results([_f("Effusion", 0.6, 0.2, zone="left lower zone")]), STUDY)
    concise = rewrite(rep, "concise")
    assert "AI p =" not in " ".join(concise["findings"].values())
    patient = rewrite(rep, "patient")
    assert patient["impression"][0].startswith("In simple terms:")
    assert "fluid" in " ".join(patient["impression"]).lower()
    detailed = rewrite(rep, "detailed")
    assert len(" ".join(detailed["findings"].values())) > len(" ".join(rep["findings"].values()))
    with pytest.raises(ValueError):
        rewrite(rep, "poetic")
