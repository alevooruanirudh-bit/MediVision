"""Access control (M31): local users, roles and bearer tokens. No external identity provider.

AUTH_ENABLED=false (single-user local demo): every request acts as `demo.radiologist`; the UI may
switch the acting demo role with the `X-Demo-Role` header so role checks can be tried out.
AUTH_ENABLED=true: a bearer token from POST /api/auth/login is required; roles are enforced.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass

from fastapi import Depends, Header
from sqlmodel import Session, select

from app.config import get_settings
from app.db import get_session
from app.models import User

ROLES = ("admin", "radiologist", "resident", "viewer")
PERMISSIONS: dict[str, tuple[str, ...]] = {
    "study.write": ("admin", "radiologist", "resident"),  # upload, analyse
    "study.delete": ("admin", "radiologist"),
    "finding.review": ("admin", "radiologist", "resident"),
    "report.edit": ("admin", "radiologist", "resident"),
    "report.sign": ("admin", "radiologist"),
    "alert.ack": ("admin", "radiologist", "resident"),
    "users.manage": ("admin",),
}
DEMO_USERS = [
    ("admin", "Admin", "admin"),
    ("dr.lee", "Dr. Lee (radiologist)", "radiologist"),
    ("dr.patel", "Dr. Patel (resident)", "resident"),
    ("student", "Student viewer", "viewer"),
]


@dataclass
class Principal:
    username: str
    role: str
    display_name: str = ""
    authenticated: bool = False

    def can(self, perm: str) -> bool:
        return self.role in PERMISSIONS.get(perm, ())


def hash_password(pw: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(8)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 120_000).hex()
    return f"{salt}${dk}"


def verify_password(pw: str, stored: str) -> bool:
    try:
        salt, _ = stored.split("$", 1)
    except ValueError:
        return False
    return hmac.compare_digest(hash_password(pw, salt), stored)


def token_hash(tok: str) -> str:
    return hashlib.sha256(tok.encode()).hexdigest()


def seed_users(session: Session) -> None:
    if session.exec(select(User)).first() is not None:
        return
    pw = get_settings().demo_password
    for u, name, role in DEMO_USERS:
        session.add(User(username=u, display_name=name, role=role, password_hash=hash_password(pw)))
    session.commit()


def login(session: Session, username: str, password: str) -> tuple[User, str] | None:
    u = session.exec(select(User).where(User.username == username)).first()
    if u is None or not verify_password(password, u.password_hash):
        return None
    tok = secrets.token_urlsafe(32)
    u.token_hash = token_hash(tok)
    session.add(u)
    session.commit()
    return u, tok


def current_user(
    authorization: str | None = Header(None),
    x_demo_role: str | None = Header(None),
    session: Session = Depends(get_session),
) -> Principal:
    from app.api.deps import APIError

    if not get_settings().auth_enabled:
        role = x_demo_role if x_demo_role in ROLES else "radiologist"
        name = {r: u for u, _, r in DEMO_USERS}.get(role, "demo")
        return Principal(f"demo.{name}" if role != "radiologist" else "demo.radiologist", role)
    if not authorization or not authorization.lower().startswith("bearer "):
        raise APIError(401, "unauthenticated", "Sign in required (Authorization: Bearer <token>)")
    th = token_hash(authorization.split(" ", 1)[1].strip())
    u = session.exec(select(User).where(User.token_hash == th)).first()
    if u is None:
        raise APIError(401, "invalid_token", "Session expired — sign in again")
    return Principal(u.username, u.role, u.display_name, True)


def require(perm: str):  # type: ignore[no-untyped-def]
    """FastAPI dependency: the acting user must hold `perm`."""

    def dep(user: Principal = Depends(current_user)) -> Principal:
        from app.api.deps import APIError

        if not user.can(perm):
            raise APIError(
                403,
                "forbidden",
                f"Role '{user.role}' may not {perm.replace('.', ' ')} "
                f"(allowed: {', '.join(PERMISSIONS[perm])}).",
            )
        return user

    return dep
