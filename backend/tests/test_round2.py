"""Round-2 requirements (D-33..D-36): mammography (C1), PET-CT fusion + fMRI (C7), whole-scan nodule
detection (M5). Unit tests run on synthetic data; API tests use the held-out demo samples."""

from __future__ import annotations

import io

import numpy as np
import pytest
from conftest import SAMPLES, needs_lungdet, needs_mammo, needs_petct


# ----------------------------------------------------------------------------- unit
def test_letterbox_roundtrip_maps_back_to_image_frame() -> None:
    from ml.data.transforms import letterbox_square, unletterbox

    img = np.zeros((300, 150), np.uint8)
    img[200:260, 20:60] = 255  # bright blob lower-left
    sq, box = letterbox_square(img, 128)
    assert sq.shape == (128, 128) and box[2] == 128 and box[3] == 64  # tall image -> pillarbox
    back = unletterbox(sq.astype(np.float32) / 255, box, 128, img.shape)
    assert back.shape == img.shape
    y, x = np.unravel_index(int(back.argmax()), back.shape)
    assert 190 <= y <= 270 and 10 <= x <= 70


def test_volume_bundle_reads_petct_and_spacing() -> None:
    from app.services.deid import load_volume_bundle

    buf = io.BytesIO()
    np.savez(
        buf,
        ct=np.zeros((20, 30, 30), np.int16),
        pet=np.ones((20, 30, 30), np.float16),
        spacing=np.array([5.0, 4.0, 4.0]),
    )
    b = load_volume_bundle(buf.getvalue(), "x.npz")
    assert b["volume"].shape == b["pet"].shape == (20, 30, 30) and b["spacing"] == [5.0, 4.0, 4.0]
    buf = io.BytesIO()
    np.savez(buf, ct=np.zeros((20, 30, 30)), pet=np.zeros((20, 30, 31)))
    with pytest.raises(ValueError):
        load_volume_bundle(buf.getvalue(), "x.npz")


def _phantom() -> tuple[np.ndarray, np.ndarray]:
    """Body (0 HU) with two air 'lungs' (-850 HU) and a 10 mm solid nodule in the right lung."""
    vol = np.full((80, 120, 120), -1000, np.int16)  # outside air
    zz, yy, xx = np.mgrid[:80, :120, :120]
    vol[((yy - 60) ** 2 / 50**2 + (xx - 60) ** 2 / 55**2) < 1] = 0
    for cx in (35, 85):
        vol[((yy - 60) ** 2 / 30**2 + (xx - cx) ** 2 / 18**2) < 1] = -850
    c = np.array([40, 60, 35])
    vol[((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) <= 25] = 20
    return vol, c


def test_lung_mask_and_candidates_find_phantom_nodule() -> None:
    from ml.data import luna as L

    vol, c = _phantom()
    mask = L.lung_mask(vol)
    assert 0.05 < mask.mean() < 0.4 and mask[tuple(c)]
    cands = L.candidates(vol, mask)
    assert len(cands) and np.linalg.norm(cands - c, axis=1).min() <= 3


def test_froc_counts_each_nodule_once() -> None:
    from ml.lung_detect import froc

    scans = [
        {
            "probs": np.array([0.9, 0.8, 0.3, 0.2]),
            "hit": np.array([0, 0, -1, -1]),
            "n_nodules": 1,
            "diam": [6],
        },
        {"probs": np.array([0.7, 0.1]), "hit": np.array([-1, 0]), "n_nodules": 1, "diam": [8]},
    ]
    f = froc(scans)
    assert f["candidate_recall"] == 1.0
    assert f["sens"]["0.125"] == 0.5  # nodule 1 found before any false positive
    assert f["sens"]["8"] == 1.0


def test_fmri_glm_recovers_synthetic_activation() -> None:
    from ml.fmri_glm import CONDITIONS, design, glm_t

    events = [
        {"onset": 10.0 + 30 * k, "duration": 15.0, "trial_type": CONDITIONS[k % 3]}
        for k in range(12)
    ]
    X, _ = design(events, 160, 2.5)
    rng = np.random.default_rng(0)
    Y = rng.normal(0, 1, (160, 50))
    Y[:, :5] += 3 * X[:, 0:1]  # voxels 0-4 respond to Finger only
    c = np.zeros(X.shape[1])
    c[0], c[1], c[2] = 1, -0.5, -0.5
    t = glm_t(Y, X, {"finger": c})["finger"]
    assert t[:5].min() > 5 and np.abs(t[5:]).max() < 4.5


def test_pet_readout_units() -> None:
    from ml.petct_fusion import VOXEL_ML, pet_readout

    pet = np.zeros((10, 10, 10), np.float32)
    mask = np.zeros_like(pet, bool)
    mask[2:4, 2:4, 2:4] = True  # 8 voxels
    pet[mask] = 5.0
    r = pet_readout(pet, mask)
    assert (
        r["suv_max"] == 5.0
        and r["mtv_ml"] == round(8 * VOXEL_ML, 1)
        and r["tlg"] == round(5 * 8 * VOXEL_ML, 1)
    )


# ----------------------------------------------------------------------------- API
@needs_mammo
def test_mammogram_pipeline(client, uploaded) -> None:
    if "mammo_cancer.png" not in uploaded:
        pytest.skip("mammography sample not generated")
    r = client.get(f"/api/studies/{uploaded['mammo_cancer.png']['id']}").json()["results"]
    assert r["pipeline"] == "mammo", r["modality"]
    f = r["findings"][0]
    assert f["label"] == "Suspicious for breast cancer (mammography)" and "risk_category" in f
    assert r["overlays"]["heatmaps"]  # Grad-CAM through the letterbox
    assert any(v["name"] == "mammo_cls" for v in r["model_versions"])


@needs_petct
def test_petct_fusion_pipeline(client, uploaded) -> None:
    if "petct_sarcoma.npz" not in uploaded:
        pytest.skip("PET-CT sample not generated")
    r = client.get(f"/api/studies/{uploaded['petct_sarcoma.npz']['id']}").json()["results"]
    assert r["pipeline"] == "petct"
    m = r["measurements"]
    assert m["suv_max"] > 0 and m["mtv_ml"] > 0 and not m["spacing_assumed"]
    assert {"ct", "pet", "fused"} <= set(r["overlays"]["volume"]["mpr"])
    assert r["overlays"]["volume"]["pet_mip"]
    assert r["findings"][0]["provenance"]["model"] == "petct_seg"


def test_petct_shape_mismatch_rejected(client) -> None:
    buf = io.BytesIO()
    np.savez(buf, ct=np.zeros((20, 30, 30)), pet=np.zeros((21, 30, 30)))
    r = client.post(
        "/api/studies",
        files={"file": ("bad.npz", buf.getvalue())},
        data={"consent_basis": "uploader_attested"},
    )
    assert r.status_code == 422


@needs_lungdet
def test_whole_scan_nodule_detection(client, uploaded) -> None:
    if "chest_ct_scan.npz" not in uploaded:
        pytest.skip("chest CT sample not generated")
    r = client.get(f"/api/studies/{uploaded['chest_ct_scan.npz']['id']}").json()["results"]
    assert r["pipeline"] == "ct_chest_scan"
    m = r["measurements"]
    assert m["candidates_scored"] > 0 and not m["spacing_assumed"]
    dets = r["extra"]["chest_scan"]["detections"]
    assert 1 <= len(dets) <= 5 and all(0 <= d["probability"] <= 1 for d in dets)
    assert r["findings"][0]["label"] == "Lung nodule (whole-scan detection)"
    assert r["overlays"]["volume"]["key_image"] and r["overlays"]["volume"]["mip_frames"]


def test_fmri_endpoint(client) -> None:
    r = client.get("/api/fmri")
    if r.status_code == 404:
        pytest.skip("fMRI maps not computed")
    d = r.json()
    sub = next(iter(d["subjects"].values()))
    assert set(sub["peaks"]) == {"Finger > others", "Foot > others", "Lips > others"}
    assert all(v.startswith("data:image/png;base64,") for v in sub["images"].values())
    assert "somatotopy" in sub and d["uncorrected_t"] > 0


def test_samples_are_held_out() -> None:
    src = (SAMPLES / "SOURCES.md").read_text() if (SAMPLES / "SOURCES.md").exists() else ""
    if "petct_sarcoma.npz" in src:
        assert "held out" in src


def test_local_nodule_size_detaches_vessel_and_flags_wall() -> None:
    from app.inference.volumes3d import local_nodule_size

    v = np.full((80, 80, 80), -850, np.int16)
    zz, yy, xx = np.mgrid[:80, :80, :80]
    v[(zz - 40) ** 2 + (yy - 40) ** 2 + (xx - 40) ** 2 <= 25] = 30  # 10 mm nodule
    v[38:42, 38:42, :] = 40  # 4 mm vessel running through it
    r = local_nodule_size(v, np.array([40, 40, 40]))
    assert r["equivalent_diameter_mm"] is not None and 8 <= r["equivalent_diameter_mm"] <= 14
    wall = np.full((80, 80, 80), -850, np.int16)
    wall[:, :, 40:] = 30  # solid wall: never a <= 30 mm nodule
    assert local_nodule_size(wall, np.array([40, 40, 42]))["equivalent_diameter_mm"] is None
