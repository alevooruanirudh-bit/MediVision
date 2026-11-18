"""Auth endpoints (local users only; see app/services/auth.py)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlmodel import Session, select

from app.api.deps import APIError
from app.config import get_settings
from app.db import get_session
from app.models import User
from app.services import auth as A
from app.services.audit import audit

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/login", summary="Sign in (returns a bearer token)")
def login(body: LoginRequest, session: Session = Depends(get_session)) -> dict[str, Any]:
    res = A.login(session, body.username, body.password)
    if res is None:
        audit(session, "auth.login_failed", "user", None, actor=body.username)
        raise APIError(401, "bad_credentials", "Unknown user or wrong password")
    u, tok = res
    audit(session, "auth.login", "user", u.id, actor=u.username, role=u.role)
    return {"token": tok, "user": {"username": u.username, "role": u.role, "name": u.display_name}}


@router.post("/logout", summary="Invalidate the current token")
def logout(
    user: A.Principal = Depends(A.current_user), session: Session = Depends(get_session)
) -> dict[str, Any]:
    u = session.exec(select(User).where(User.username == user.username)).first()
    if u is not None:
        u.token_hash = None
        session.add(u)
        session.commit()
    audit(session, "auth.logout", "user", None, actor=user.username, role=user.role)
    return {"ok": True}


@router.get("/me", summary="Who am I, and what may I do")
def me(user: A.Principal = Depends(A.current_user)) -> dict[str, Any]:
    return {
        "username": user.username,
        "role": user.role,
        "name": user.display_name or user.username,
        "authenticated": user.authenticated,
        "auth_enabled": get_settings().auth_enabled,
        "permissions": [p for p in A.PERMISSIONS if user.can(p)],
        "roles": list(A.ROLES),
    }


@router.get("/users", summary="List users (admin)")
def users(
    _: A.Principal = Depends(A.require("users.manage")), session: Session = Depends(get_session)
) -> list[dict[str, Any]]:
    return [
        {"username": u.username, "name": u.display_name, "role": u.role}
        for u in session.exec(select(User)).all()
    ]
