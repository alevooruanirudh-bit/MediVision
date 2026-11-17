"""SQLModel tables: Patient, Study, Finding, Review, Report, ChatMessage, AuditLog."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlmodel import Field, SQLModel


def _id() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> datetime:
    return datetime.now(UTC)


class Patient(SQLModel, table=True):
    id: str = Field(default_factory=_id, primary_key=True)
    pseudonym: str = Field(index=True, unique=True)
    sex: str | None = None
    age: int | None = None
    created_at: datetime = Field(default_factory=_now)


class Study(SQLModel, table=True):
    id: str = Field(default_factory=_id, primary_key=True)
    patient_id: str = Field(foreign_key="patient.id", index=True)
    filename: str = ""
    original_format: str = "png"  # png | jpg | dcm
    modality: str | None = None  # router decision
    modality_hint: str | None = None
    image_path: str = ""  # relative to storage dir (de-identified PNG)
    width: int = 0
    height: int = 0
    pixel_spacing_mm: float | None = None
    status: str = "uploaded"  # uploaded | analyzed | error
    urgency_score: float | None = None
    urgency_level: str | None = None
    ood: bool = False
    model_profile: str | None = None
    indication: str | None = None
    study_date: str | None = None
    description: str | None = None
    results_json: str | None = None
    deid_json: str | None = None
    created_at: datetime = Field(default_factory=_now, index=True)
    analyzed_at: datetime | None = None

    @property
    def results(self) -> dict[str, Any]:
        return json.loads(self.results_json) if self.results_json else {}


class Finding(SQLModel, table=True):
    id: str = Field(default_factory=_id, primary_key=True)
    study_id: str = Field(foreign_key="study.id", index=True)
    label: str
    kind: str = "classification"  # classification | detection | segmentation
    probability: float = 0.0
    uncertainty: float = 0.0
    threshold: float = 0.5
    certainty: str = "Unlikely"  # Likely | Possible | Unlikely
    positive: bool = False
    zone: str | None = None
    box_json: str | None = None
    status: str = "pending"  # pending | accepted | rejected | edited
    edited_label: str | None = None
    note: str | None = None
    rank: int = 0


class Review(SQLModel, table=True):
    id: str = Field(default_factory=_id, primary_key=True)
    finding_id: str = Field(foreign_key="finding.id", index=True)
    study_id: str = Field(index=True)
    action: str  # accept | reject | edit
    reviewer: str = "radiologist"
    original_label: str = ""
    new_label: str | None = None
    comment: str | None = None
    model_probability: float = 0.0
    created_at: datetime = Field(default_factory=_now)


class Report(SQLModel, table=True):
    id: str = Field(default_factory=_id, primary_key=True)
    study_id: str = Field(foreign_key="study.id", index=True)
    engine: str = "rules"  # rules | experimental
    style: str = "standard"  # standard | concise | detailed | patient
    sections_json: str = "{}"
    status: str = "draft"  # draft | signed
    signed_by: str | None = None
    signed_at: datetime | None = None
    version: int = 1
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    @property
    def sections(self) -> dict[str, Any]:
        return json.loads(self.sections_json or "{}")


class ChatMessage(SQLModel, table=True):
    id: str = Field(default_factory=_id, primary_key=True)
    study_id: str = Field(index=True)
    role: str  # user | assistant
    content: str
    intent: str | None = None
    actions_json: str | None = None
    created_at: datetime = Field(default_factory=_now)


class AuditLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    ts: datetime = Field(default_factory=_now, index=True)
    actor: str = "system"
    action: str
    entity: str = ""
    entity_id: str | None = None
    details_json: str | None = None
