"""Append-only, hash-chained audit log (C5: tamper-evident, blockchain-style linking).

Each entry stores `prev_hash` and `hash = sha256(prev_hash | canonical entry)`. Editing or deleting
any past row breaks every later hash, which `verify_chain()` detects.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime
from typing import Any

from sqlmodel import Session, col, select

from app.models import AuditLog

log = logging.getLogger("medivision.audit")
_lock = threading.Lock()
GENESIS = "0" * 64


def _ts(ts: datetime) -> str:
    return ts.replace(tzinfo=None).isoformat(timespec="microseconds")


def entry_hash(prev: str, e: AuditLog) -> str:
    canon = json.dumps(
        [_ts(e.ts), e.actor, e.role, e.action, e.entity, e.entity_id, e.details_json],
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(f"{prev}|{canon}".encode()).hexdigest()


def audit(
    session: Session,
    action: str,
    entity: str = "",
    entity_id: str | None = None,
    actor: str = "user",
    role: str | None = None,
    **details: Any,
) -> AuditLog:
    with _lock:
        last = session.exec(
            select(AuditLog)
            .where(col(AuditLog.hash).is_not(None))
            .order_by(col(AuditLog.id).desc())
        ).first()
        entry = AuditLog(
            actor=actor,
            role=role,
            action=action,
            entity=entity,
            entity_id=entity_id,
            details_json=json.dumps(details, default=str, sort_keys=True) if details else None,
        )
        entry.prev_hash = last.hash if last and last.hash else GENESIS
        entry.hash = entry_hash(entry.prev_hash, entry)
        session.add(entry)
        session.commit()
    log.info("audit %s %s %s %s", actor, action, entity, entity_id)
    return entry


def verify_chain(session: Session) -> dict[str, Any]:
    rows = session.exec(select(AuditLog).order_by(col(AuditLog.id))).all()
    legacy = 0
    prev = None
    checked = 0
    for r in rows:
        if r.hash is None:
            legacy += 1
            continue
        expect_prev = prev if prev is not None else GENESIS
        if r.prev_hash != expect_prev or entry_hash(r.prev_hash or GENESIS, r) != r.hash:
            return {
                "valid": False,
                "broken_at": r.id,
                "checked": checked,
                "legacy_unchained": legacy,
                "message": f"Audit entry #{r.id} was altered or an earlier entry was removed.",
            }
        prev = r.hash
        checked += 1
    return {
        "valid": True,
        "checked": checked,
        "legacy_unchained": legacy,
        "head": prev,
        "message": f"{checked} chained entries verified"
        + (f"; {legacy} legacy entries pre-date the chain" if legacy else ""),
    }
