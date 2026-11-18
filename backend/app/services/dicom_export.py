"""DICOM outputs for PACS (M21): Secondary Capture of the image + AI overlay, and the report as an
Encapsulated PDF. Pseudonymised identifiers only; every object is tagged as AI-generated."""

from __future__ import annotations

import io
from datetime import UTC, datetime
from typing import Any

import cv2
import numpy as np

from app.config import APP_VERSION, DISCLAIMER
from app.models import Patient, Report, Study
from app.services import storage

SC_SOP = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture Image Storage
PDF_SOP = "1.2.840.10008.5.1.4.1.1.104.1"  # Encapsulated PDF Storage


def _base(sop_class: str, study: Study, pseudonym: str, modality: str) -> Any:
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    now = datetime.now(UTC)
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid(entropy_srcs=[study.id])
    ds.SeriesInstanceUID = generate_uid()
    ds.PatientName = pseudonym
    ds.PatientID = pseudonym
    ds.PatientIdentityRemoved = "YES"
    ds.Modality = modality
    ds.StudyID = study.id[:16]
    ds.SeriesNumber = 900
    ds.InstanceNumber = 1
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S")
    ds.Manufacturer = "MediVision AI (research prototype)"
    ds.SoftwareVersions = APP_VERSION
    ds.SeriesDescription = "MediVision AI output — NOT FOR CLINICAL USE"
    ds.ImageComments = DISCLAIMER
    ds.BurnedInAnnotation = "YES"
    ds.is_little_endian, ds.is_implicit_VR = True, False
    return ds


def _pseudonym(session: Any, study: Study) -> str:
    p = session.get(Patient, study.patient_id)
    return p.pseudonym if p else "ANON"


def secondary_capture(session: Any, study: Study, finding: str | None = None) -> bytes:
    res = study.results
    img = storage.load_image(study.image_path, gray=False)
    heat = res.get("overlays", {}).get("heatmaps", {})
    key = finding or next(iter(heat), None)
    if key and heat.get(key, {}).get("gradcam"):
        rel = heat[key]["gradcam"].removeprefix("/files/")
        ov = cv2.imread(str(storage.abs_path(rel)), cv2.IMREAD_UNCHANGED)
        if ov is not None and ov.shape[2] == 4:
            ov = cv2.resize(ov, (img.shape[1], img.shape[0]))
            a = ov[..., 3:4].astype(np.float32) / 255
            img = (img * (1 - a) + cv2.cvtColor(ov[..., :3], cv2.COLOR_BGR2RGB) * a).astype(
                np.uint8
            )
    lines = [f"MediVision AI - {res.get('triage', {}).get('level', '')} - NOT FOR CLINICAL USE"]
    lines += [
        f"{f['label']}: {f['probability']:.0%} ({f['certainty']})"
        for f in res.get("findings", [])
        if f.get("positive")
    ][:4]
    y = 22
    for t in lines:
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(
            img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 230, 120), 1, cv2.LINE_AA
        )
        y += 22
    ds = _base(SC_SOP, study, _pseudonym(session, study), "OT")
    ds.ConversionType = "WSD"
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB"
    ds.PlanarConfiguration = 0
    ds.Rows, ds.Columns = img.shape[:2]
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.ImageType = ["DERIVED", "SECONDARY", "AI_OVERLAY"]
    ds.DerivationDescription = f"AI heatmap ({key}) + findings burned in"
    ds.PixelData = np.ascontiguousarray(img).tobytes()
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


def encapsulated_pdf(pdf: bytes, report: Report, study: Study, patient: Patient) -> bytes:
    ds = _base(PDF_SOP, study, patient.pseudonym, "DOC")
    ds.MIMETypeOfEncapsulatedDocument = "application/pdf"
    ds.DocumentTitle = "MediVision AI-assisted report"
    ds.BurnedInAnnotation = "NO"
    ds.ConceptNameCodeSequence = []
    ds.VerificationFlag = "VERIFIED" if report.status == "signed" else "UNVERIFIED"
    if report.signed_by:
        ds.VerifyingObserverName = report.signed_by
    ds.EncapsulatedDocument = pdf + (b"\0" if len(pdf) % 2 else b"")
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()
