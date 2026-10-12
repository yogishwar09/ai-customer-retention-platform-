"""Database-backed job queue (portable: SQLite and PostgreSQL).

Claiming uses an optimistic compare-and-set update, so several workers can run
safely. Failed jobs retry with exponential backoff until ``max_attempts``, then
become ``dead`` for inspection. Swap for Celery/RQ + Redis when throughput needs it.
"""

from __future__ import annotations

import json
import os
import socket
from typing import Any

import pandas as pd
from sqlalchemy import Engine, text

from retentioniq.storage.engine import get_engine
from retentioniq.storage.util import new_id, now_iso

WORKER_ID = f"{socket.gethostname()}:{os.getpid()}"


class JobQueue:
    def __init__(self, engine: Engine | None = None):
        self.engine = engine or get_engine()

    def enqueue(self, job_type: str, payload: dict[str, Any] | None = None, *, org_id: str | None = None,
                created_by: str = "system", max_attempts: int = 3, delay_seconds: float = 0,
                job_id: str | None = None) -> str:
        job_id = job_id or new_id("job")
        with self.engine.begin() as c:
            c.execute(text(
                "INSERT INTO jobs (job_id, org_id, job_type, payload, status, attempts, max_attempts, run_after, "
                "created_by, created_at) VALUES (:i, :o, :t, :p, 'queued', 0, :m, :r, :b, :c)"),
                dict(i=job_id, o=org_id, t=job_type, p=json.dumps(payload or {}), m=max_attempts,
                     r=now_iso(delay_seconds), b=created_by, c=now_iso()))
        return job_id

    def claim(self, worker_id: str = WORKER_ID, job_types: list[str] | None = None) -> dict[str, Any] | None:
        for _ in range(5):
            with self.engine.connect() as c:
                sql = "SELECT job_id FROM jobs WHERE status = 'queued' AND run_after <= :now"
                params: dict[str, Any] = {"now": now_iso()}
                if job_types:
                    sql += " AND job_type IN (" + ", ".join(f":t{i}" for i in range(len(job_types))) + ")"
                    params.update({f"t{i}": t for i, t in enumerate(job_types)})
                row = c.execute(text(sql + " ORDER BY run_after LIMIT 1"), params).first()
            if row is None:
                return None
            with self.engine.begin() as c:
                won = c.execute(text(
                    "UPDATE jobs SET status = 'running', locked_by = :w, locked_at = :t, attempts = attempts + 1 "
                    "WHERE job_id = :i AND status = 'queued'"), dict(w=worker_id, t=now_iso(), i=row[0])).rowcount
                if won:
                    job = c.execute(text("SELECT * FROM jobs WHERE job_id = :i"), dict(i=row[0])).mappings().first()
                    out = dict(job)
                    out["payload"] = json.loads(out["payload"] or "{}")
                    return out
        return None

    def progress(self, job_id: str, fraction: float) -> None:
        with self.engine.begin() as c:
            c.execute(text("UPDATE jobs SET progress = :p WHERE job_id = :i"), dict(p=float(fraction), i=job_id))

    def complete(self, job_id: str, result: dict[str, Any] | None = None) -> None:
        with self.engine.begin() as c:
            c.execute(text("UPDATE jobs SET status = 'succeeded', progress = 1, result = :r, finished_at = :t, "
                           "locked_by = NULL WHERE job_id = :i"),
                      dict(r=json.dumps(result or {}, default=str), t=now_iso(), i=job_id))

    def fail(self, job_id: str, error: str) -> str:
        with self.engine.begin() as c:
            job = c.execute(text("SELECT attempts, max_attempts FROM jobs WHERE job_id = :i"),
                            dict(i=job_id)).mappings().first()
            dead = job["attempts"] >= job["max_attempts"]
            c.execute(text(
                "UPDATE jobs SET status = CAST(:s AS VARCHAR), error = :e, run_after = :r, locked_by = NULL, "
                "finished_at = CASE WHEN CAST(:s AS VARCHAR) = 'dead' THEN CAST(:t AS VARCHAR) ELSE NULL END "
                "WHERE job_id = :i"),
                dict(s="dead" if dead else "queued", e=error[:2000], r=now_iso(30 * 2 ** job["attempts"]),
                     t=now_iso(), i=job_id))
        return "dead" if dead else "queued"

    def retry(self, job_id: str) -> None:
        with self.engine.begin() as c:
            n = c.execute(text("UPDATE jobs SET status = 'queued', attempts = 0, run_after = :t, error = NULL "
                               "WHERE job_id = :i AND status IN ('dead', 'failed')"),
                          dict(t=now_iso(), i=job_id)).rowcount
        if not n:
            raise KeyError(job_id)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as c:
            row = c.execute(text("SELECT * FROM jobs WHERE job_id = :i"), dict(i=job_id)).mappings().first()
        if row is None:
            return None
        out = dict(row)
        out["payload"] = json.loads(out["payload"] or "{}")
        out["result"] = json.loads(out["result"]) if out["result"] else None
        return out

    def list(self, org_id: str | None = None, limit: int = 200, include_system: bool = True) -> pd.DataFrame:
        sql = "SELECT job_id, org_id, job_type, status, attempts, max_attempts, progress, created_by, created_at, " \
              "run_after, finished_at, error FROM jobs"
        params: dict[str, Any] = {"n": limit}
        if org_id:
            sql += " WHERE (org_id = :o" + (" OR org_id IS NULL)" if include_system else ")")
            params["o"] = org_id
        with self.engine.connect() as c:
            return pd.read_sql_query(text(sql + " ORDER BY created_at DESC LIMIT :n"), c, params=params)

    def depth(self) -> dict[str, int]:
        with self.engine.connect() as c:
            rows = c.execute(text("SELECT status, COUNT(*) FROM jobs GROUP BY status")).all()
        return {r[0]: int(r[1]) for r in rows}

    # ------------------------------------------------------------ scheduler
    def schedule_tick(self, schedules: dict[str, int]) -> list[str]:
        """Enqueue periodic system jobs whose interval has elapsed (idempotent per interval)."""
        enqueued = []
        now = now_iso()
        with self.engine.begin() as c:
            existing = {r[0]: r for r in c.execute(text("SELECT name, interval_seconds, enabled, last_enqueued_at "
                                                        "FROM schedules")).all()}
            for name, interval in schedules.items():
                if name not in existing:
                    # First run happens one interval after registration (no thundering herd on deploy);
                    # short-interval tasks such as webhook dispatch start right away.
                    first_last = None if interval <= 60 else now
                    c.execute(text("INSERT INTO schedules (name, interval_seconds, enabled, last_enqueued_at) "
                                   "VALUES (:n, :i, 1, :l)"), dict(n=name, i=int(interval), l=first_last))
                    last, enabled = first_last, 1
                else:
                    _, _, enabled, last = existing[name]
                    c.execute(text("UPDATE schedules SET interval_seconds = :i WHERE name = :n"),
                              dict(n=name, i=int(interval)))
                if not enabled:
                    continue
                due = last is None or (pd.Timestamp(now) - pd.Timestamp(last)).total_seconds() >= interval
                if not due:
                    continue
                guard = "last_enqueued_at IS NULL" if last is None else "last_enqueued_at = :last"
                won = c.execute(text(f"UPDATE schedules SET last_enqueued_at = :t WHERE name = :n AND {guard}"),
                                dict(t=now, n=name, last=last)).rowcount
                if won:
                    enqueued.append(name)
        for name in enqueued:
            self.enqueue(name, {}, created_by="scheduler", max_attempts=1)
        return enqueued

    def schedules(self) -> pd.DataFrame:
        with self.engine.connect() as c:
            return pd.read_sql_query(text("SELECT * FROM schedules ORDER BY name"), c)

    def set_schedule_enabled(self, name: str, enabled: bool) -> None:
        with self.engine.begin() as c:
            c.execute(text("UPDATE schedules SET enabled = :e WHERE name = :n"), dict(e=int(enabled), n=name))
