"""Database engine + session dependency (SQLite by default)."""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import event
from sqlmodel import Session, SQLModel, create_engine

from app.config import get_settings

_engine = None


def get_engine():  # type: ignore[no-untyped-def]
    global _engine
    if _engine is None:
        url = get_settings().database_url
        _engine = create_engine(
            url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {}
        )
        if url.startswith("sqlite"):

            @event.listens_for(_engine, "connect")
            def _pragma(conn, _):  # type: ignore[no-untyped-def]
                cur = conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()

    return _engine


def reset_engine(url: str | None = None) -> None:
    """Testing helper: point the app at another database URL."""
    global _engine
    if url:
        get_settings().database_url = url
    _engine = None


def init_db() -> None:
    from app import models  # noqa: F401  (register tables)

    SQLModel.metadata.create_all(get_engine())
    migrate()


def migrate() -> list[str]:
    """Additive SQLite migration: ADD COLUMN for model fields missing in an older database."""
    eng = get_engine()
    if not str(eng.url).startswith("sqlite"):
        return []
    added = []
    with eng.begin() as conn:
        for table in SQLModel.metadata.sorted_tables:
            have = {r[1] for r in conn.exec_driver_sql(f"PRAGMA table_info('{table.name}')")}
            for col in table.columns:
                if col.name in have:
                    continue
                typ = col.type.compile(dialect=eng.dialect)
                conn.exec_driver_sql(f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {typ}')
                added.append(f"{table.name}.{col.name}")
    return added


def get_session() -> Iterator[Session]:
    with Session(get_engine()) as s:
        yield s
