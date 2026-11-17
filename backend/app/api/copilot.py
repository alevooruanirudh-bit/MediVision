"""Copilot chat over Server-Sent Events (streamed word by word) + history."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlmodel import Session, select
from sse_starlette.sse import EventSourceResponse

from app.api.deps import get_study
from app.copilot.chat import answer, suggestions_for
from app.db import get_session
from app.models import ChatMessage
from app.services.audit import audit

router = APIRouter(prefix="/api/copilot", tags=["copilot"])


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    stream: bool = True
    delay_ms: int = Field(18, ge=0, le=200)


@router.post("/{study_id}/chat", summary="Ask the Copilot (SSE stream: token… then done)")
async def chat(study_id: str, body: ChatRequest, session: Session = Depends(get_session)) -> Any:
    study = get_study(session, study_id)
    session.add(ChatMessage(study_id=study_id, role="user", content=body.message))
    session.commit()
    ans = answer(body.message, study, session)
    meta = {"focus": ans.focus, "actions": ans.actions, "suggestions": ans.suggestions}
    session.add(
        ChatMessage(
            study_id=study_id,
            role="assistant",
            content=ans.text,
            intent=ans.intent,
            actions_json=json.dumps(meta),
        )
    )
    session.commit()
    audit(
        session,
        "copilot.chat",
        "study",
        study_id,
        actor="copilot",
        intent=ans.intent,
        confidence=round(ans.confidence, 3),
    )
    payload = ans.as_dict()
    if not body.stream:
        return payload

    async def gen() -> AsyncIterator[dict[str, str]]:
        yield {
            "event": "meta",
            "data": json.dumps(
                {"intent": ans.intent, "confidence": ans.confidence, "focus": ans.focus}
            ),
        }
        words = ans.text.split(" ")
        for i, w in enumerate(words):
            yield {"event": "token", "data": json.dumps(w + (" " if i < len(words) - 1 else ""))}
            await asyncio.sleep(body.delay_ms / 1000)
        yield {"event": "done", "data": json.dumps(payload, default=str)}

    return EventSourceResponse(gen())


@router.get("/{study_id}/history", summary="Chat history + starter suggestions")
def history(study_id: str, session: Session = Depends(get_session)) -> dict[str, Any]:
    study = get_study(session, study_id)
    msgs = session.exec(
        select(ChatMessage).where(ChatMessage.study_id == study_id).order_by(ChatMessage.created_at)
    ).all()  # type: ignore[arg-type]
    from app.models import Study

    has_prior = (
        session.exec(
            select(Study).where(Study.patient_id == study.patient_id, Study.id != study.id)
        ).first()
        is not None
    )
    return {
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "intent": m.intent,
                **(json.loads(m.actions_json) if m.actions_json else {}),
                "created_at": m.created_at.isoformat(),
            }
            for m in msgs
        ],
        "suggestions": suggestions_for(study.results, has_prior),
    }
