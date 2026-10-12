"""Database engine. SQLite by default; set RETENTIONIQ_DATABASE_URL for PostgreSQL
(e.g. postgresql+psycopg://user:pass@host/db)."""

from __future__ import annotations

import os
from functools import lru_cache

from sqlalchemy import Engine, create_engine, event

from retentioniq import config


def database_url() -> str:
    url = os.environ.get("RETENTIONIQ_DATABASE_URL")
    if url:
        return url
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{config.DB_PATH}"


@lru_cache(maxsize=4)
def get_engine(url: str | None = None) -> Engine:
    url = url or database_url()
    if url.startswith("sqlite"):
        engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30},
                               pool_pre_ping=True)

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _):  # pragma: no cover - driver hook
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA busy_timeout=30000")
            cur.close()
    else:
        engine = create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)
    from retentioniq.storage.tables import metadata
    metadata.create_all(engine)
    return engine


def is_sqlite(engine: Engine) -> bool:
    return engine.dialect.name == "sqlite"
