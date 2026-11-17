"""Append-only audit log helper."""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlmodel import Session

from app.models import AuditLog

log = logging.getLogger("medivision.audit")


def audit(
    session: Session,
    action: str,
    entity: str = "",
    entity_id: str | None = None,
    actor: str = "user",
    **details: Any,
) -> None:
    entry = AuditLog(
        actor=actor,
        action=action,
        entity=entity,
        entity_id=entity_id,
        details_json=json.dumps(details, default=str) if details else None,
    )
    session.add(entry)
    session.commit()
    log.info("audit %s %s %s %s", actor, action, entity, entity_id)
