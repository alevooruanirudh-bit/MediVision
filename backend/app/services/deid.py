"""Image ingestion + DICOM de-identification (PS3.15 Basic Profile subset) + pseudonymisation."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageOps

# Tags removed or blanked (PHI). Age, sex, modality, pixel spacing and view are kept for analysis.
PHI_TAGS: list[str] = [
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientBirthTime",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "OtherPatientIDs",
    "OtherPatientNames",
    "OtherPatientIDsSequence",
    "PatientMotherBirthName",
    "MilitaryRank",
    "EthnicGroup",
    "Occupation",
    "MedicalRecordLocator",
    "InstitutionName",
    "InstitutionAddress",
    "InstitutionalDepartmentName",
    "ReferringPhysicianName",
    "ReferringPhysicianAddress",
    "ReferringPhysicianTelephoneNumbers",
    "PerformingPhysicianName",
    "OperatorsName",
    "PhysiciansOfRecord",
    "NameOfPhysiciansReadingStudy",
    "RequestingPhysician",
    "AccessionNumber",
    "StudyID",
    "StationName",
    "DeviceSerialNumber",
    "StudyDate",
    "SeriesDate",
    "AcquisitionDate",
    "ContentDate",
    "StudyTime",
    "SeriesTime",
    "AcquisitionTime",
    "ContentTime",
    "PatientComments",
    "AdditionalPatientHistory",
    "RequestAttributesSequence",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "FrameOfReferenceUID",
]


@dataclass
class Ingested:
    pixels: np.ndarray  # uint8 grayscale (H,W) or RGB (H,W,3)
    fmt: str
    pseudonym: str
    age: int | None = None
    sex: str | None = None
    pixel_spacing_mm: float | None = None
    modality_tag: str | None = None
    view: str | None = None
    removed_tags: list[str] = field(default_factory=list)
    deid_dataset: Any = None
    volume: np.ndarray | None = None  # (D,H,W) for CT volumes (.npy/.npz or multi-frame DICOM)
    voxel_spacing_mm: list[float] | None = None
    pet: np.ndarray | None = None  # co-registered PET (SUV) for PET-CT uploads (.npz ct + pet)


def pseudonymize(identifier: str, salt: str) -> str:
    h = hashlib.sha256(f"{salt}:{identifier}".encode()).hexdigest()
    return f"PT-{h[:6].upper()}"


def _parse_age(v: str | None) -> int | None:
    if not v:
        return None
    try:
        n = int("".join(ch for ch in str(v) if ch.isdigit()) or 0)
        unit = str(v)[-1].upper()
        return n // 12 if unit == "M" else 0 if unit in ("W", "D") else n
    except ValueError:
        return None


def deidentify_dicom(ds: Any) -> list[str]:
    removed = []
    for tag in PHI_TAGS:
        if tag in ds:
            removed.append(tag)
            delattr(ds, tag)
    ds.remove_private_tags()
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = "MediVision basic profile (PHI tags removed, private tags removed)"
    return removed


def dicom_to_uint8(ds: Any) -> np.ndarray:
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
        arr = arr[arr.shape[0] // 2]  # middle frame/slice
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    inter = float(getattr(ds, "RescaleIntercept", 0) or 0)
    arr = arr * slope + inter
    wc, ww = getattr(ds, "WindowCenter", None), getattr(ds, "WindowWidth", None)
    if wc is not None and ww is not None:
        wc = float(wc[0] if hasattr(wc, "__len__") and not isinstance(wc, str) else wc)
        ww = float(ww[0] if hasattr(ww, "__len__") and not isinstance(ww, str) else ww)
        lo, hi = wc - ww / 2, wc + ww / 2
    else:
        lo, hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = 255 - arr
    return arr.astype(np.uint8)


def _check3d(arr: np.ndarray) -> np.ndarray:
    arr = np.squeeze(np.asarray(arr))
    if arr.ndim != 3 or min(arr.shape) < 8 or max(arr.shape) > 512:
        raise ValueError(f"expected a 3-D volume (8..512 per axis), got shape {arr.shape}")
    return arr


def load_volume_bundle(data: bytes, name: str) -> dict[str, Any]:
    """.npy / .npz -> {"volume", "pet"?, "spacing"?}. Pickles are refused (allow_pickle=False).

    .npz keys understood: `ct` + `pet` (PET-CT, same grid), `volume` / first array (CT), optional
    `spacing` (z, y, x in mm).
    """
    obj = np.load(io.BytesIO(data), allow_pickle=False)
    if not name.endswith(".npz"):
        return {"volume": _check3d(obj)}
    files = set(obj.files)
    out: dict[str, Any] = {}
    if {"ct", "pet"} <= files:
        out["volume"], out["pet"] = _check3d(obj["ct"]), _check3d(obj["pet"])
        if out["volume"].shape != out["pet"].shape:
            raise ValueError("ct and pet arrays must have the same shape (co-registered)")
    else:
        key = "volume" if "volume" in files else next(f for f in obj.files if f != "spacing")
        out["volume"] = _check3d(obj[key])
    if "spacing" in files:
        sp = [float(v) for v in np.asarray(obj["spacing"]).ravel()[:3]]
        if len(sp) == 3 and all(0.1 <= v <= 20 for v in sp):
            out["spacing"] = sp
    return out


def load_volume_array(data: bytes, name: str) -> np.ndarray:
    """.npy / .npz -> the CT volume (see load_volume_bundle)."""
    return load_volume_bundle(data, name)["volume"]


def ingest(data: bytes, filename: str, salt: str, patient_ref: str | None = None) -> Ingested:
    name = filename.lower()
    if name.endswith((".npy", ".npz")):
        b = load_volume_bundle(data, name)
        vol = b["volume"]
        mid = vol[vol.shape[0] // 2].astype(np.float32)
        lo, hi = float(np.percentile(mid, 0.5)), float(np.percentile(mid, 99.5))
        px = (np.clip((mid - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
        ident = patient_ref or hashlib.sha256(data[:65536]).hexdigest()
        return Ingested(
            px,
            "npy",
            pseudonymize(ident, salt),
            modality_tag="PT/CT" if "pet" in b else "CT",
            volume=vol,
            voxel_spacing_mm=b.get("spacing"),
            pet=b.get("pet"),
        )
    is_dicom = name.endswith((".dcm", ".dicom")) or data[128:132] == b"DICM"
    if is_dicom:
        import pydicom

        ds = pydicom.dcmread(io.BytesIO(data), force=True)
        ident = str(
            getattr(ds, "PatientID", "")
            or getattr(ds, "PatientName", "")
            or patient_ref
            or hashlib.sha256(data[:4096]).hexdigest()
        )
        age = _parse_age(getattr(ds, "PatientAge", None))
        sex = str(getattr(ds, "PatientSex", "") or "") or None
        spacing = getattr(ds, "PixelSpacing", None) or getattr(ds, "ImagerPixelSpacing", None)
        px = dicom_to_uint8(ds)
        vol, vsp = None, None
        raw = ds.pixel_array
        if raw.ndim == 3 and raw.shape[-1] not in (3, 4) and raw.shape[0] >= 8:
            slope = float(getattr(ds, "RescaleSlope", 1) or 1)
            inter = float(getattr(ds, "RescaleIntercept", 0) or 0)
            vol = raw.astype(np.float32) * slope + inter
            thk = float(getattr(ds, "SliceThickness", 0) or 0) or None
            if spacing and thk:
                vsp = [thk, float(spacing[0]), float(spacing[1])]
        removed = deidentify_dicom(ds)
        return Ingested(
            px,
            "dcm",
            pseudonymize(ident, salt),
            age,
            sex,
            float(spacing[0]) if spacing else None,
            str(getattr(ds, "Modality", "") or "") or None,
            str(getattr(ds, "ViewPosition", "") or "") or None,
            removed,
            ds,
            vol,
            vsp,
        )
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)  # also drops EXIF metadata on re-save
    arr = np.asarray(img.convert("RGB"))
    if np.allclose(arr[..., 0], arr[..., 1], atol=2) and np.allclose(
        arr[..., 1], arr[..., 2], atol=2
    ):
        arr = arr[..., 0]
    ident = patient_ref or hashlib.sha256(data[:65536]).hexdigest()
    return Ingested(
        arr.astype(np.uint8),
        "jpg" if name.endswith((".jpg", ".jpeg")) else "png",
        pseudonymize(ident, salt),
    )
