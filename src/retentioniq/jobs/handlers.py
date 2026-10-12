"""Job handlers. Each receives (job, ctx) and returns a JSON-serialisable result."""

from __future__ import annotations

import io
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from retentioniq import config
from retentioniq.data.schema import FEATURE_COLUMNS, ID_COLUMN, SchemaError
from retentioniq.storage.accounts import Accounts
from retentioniq.storage.blobs import store
from retentioniq.storage.repository import Repository
from retentioniq.storage.util import now_iso


class JobContext:
    def __init__(self, queue, service_factory: Callable[[], Any]):
        self.queue = queue
        self._service_factory = service_factory

    @property
    def service(self):
        return self._service_factory()


def _read_csv(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8-sig")


def batch_score(job: dict, ctx: JobContext) -> dict[str, Any]:
    """Asynchronous batch scoring of an uploaded file stored in blob storage."""
    p = job["payload"]
    repo = Repository(job["org_id"])
    svc = ctx.service
    blobs = store()
    batch_id = p.get("batch_job_id") or job["job_id"]
    if repo.get_batch_job(batch_id) is None:
        repo.start_batch_job(p["filename"], p.get("actor", "system"), svc.model.version,
                             source=p.get("source", "upload"), job_id=batch_id)
    try:
        df = _read_csv(blobs.get(p["upload_uri"]))
        if len(df) > config.settings()["uploads"]["max_rows"]:
            raise SchemaError(f"{len(df):,} rows exceeds the configured limit")
        batch = svc.score(df, source=p.get("source", "batch"), repo=repo, persist=bool(p.get("persist", True)),
                          batch_job_id=batch_id, chunk_size=500,
                          progress=lambda f: ctx.queue.progress(job["job_id"], f * 0.95))
    except Exception as exc:
        repo.finish_batch_job(batch_id, status="failed", error=f"{type(exc).__name__}: {exc}"[:500],
                              actor=p.get("actor", "system"))
        if isinstance(exc, SchemaError):
            return {"status": "failed", "error": str(exc)}  # bad input: don't retry
        raise
    stem = Path(p["filename"]).stem[:60]
    result_uri = blobs.put(f"{job['org_id']}/results/{batch_id}_{stem}_scored.csv",
                           batch.to_frame().to_csv(index=False).encode())
    error_uri = None
    if not batch.validation.rejected.empty:
        error_uri = blobs.put(f"{job['org_id']}/results/{batch_id}_{stem}_errors.csv",
                              batch.validation.rejected.to_csv(index=False).encode())
    high = sum(1 for r in batch.records if r["risk_tier"] == "high")
    repo.finish_batch_job(batch_id, status="completed", summary=batch.validation.summary,
                          actor=p.get("actor", "system"), result_uri=result_uri, error_report_uri=error_uri,
                          high_risk=high)
    return {"batch_job_id": batch_id, **batch.validation.summary, "high_risk": high, "result_uri": result_uri,
            "error_report_uri": error_uri}


def dispatch_webhooks(job: dict, ctx: JobContext) -> dict[str, Any]:
    from retentioniq.integrations.webhooks import dispatch_due
    return dispatch_due(ctx.queue.engine)


def inbox_import(job: dict, ctx: JobContext) -> dict[str, Any]:
    """Scheduled import: <inbox_dir>/<org-slug>/*.csv -> async batch scoring jobs."""
    root = config.PROJECT_ROOT / config.settings()["jobs"]["inbox_dir"]
    blobs = store()
    picked = []
    for org in Accounts(ctx.queue.engine).list_orgs():
        folder = root / org["slug"]
        if not folder.is_dir():
            continue
        for f in sorted(folder.glob("*.csv")):
            processed = folder / "processed"
            processed.mkdir(exist_ok=True)
            uri = blobs.put(f"{org['org_id']}/uploads/{now_iso().replace(':', '')}_{f.name}", f.read_bytes())
            shutil.move(str(f), processed / f"{now_iso().replace(':', '')}_{f.name}")
            jid = ctx.queue.enqueue("batch_score", {"upload_uri": uri, "filename": f.name, "persist": True,
                                                    "source": "scheduled_import", "actor": "scheduler"},
                                    org_id=org["org_id"], created_by="scheduler")
            Repository(org["org_id"]).audit("scheduler", "import.picked_up", "file", f.name, {"job_id": jid})
            picked.append({"org": org["slug"], "file": f.name, "job_id": jid})
    return {"files": picked}


def scheduled_rescore(job: dict, ctx: JobContext) -> dict[str, Any]:
    """Re-score every customer's latest snapshot with the production model so score
    history, rising-risk events and drift stay current."""
    svc = ctx.service
    orgs = [job["org_id"]] if job.get("org_id") else [o["org_id"] for o in Accounts(ctx.queue.engine).list_orgs()]
    out = {}
    for org_id in orgs:
        repo = Repository(org_id)
        latest = repo.latest_scores()
        if latest.empty:
            continue
        frame = latest[[ID_COLUMN if ID_COLUMN in latest else "customer_id", *FEATURE_COLUMNS]].rename(
            columns={"customer_id": ID_COLUMN})
        batch = svc.score(frame, source="scheduled_rescore", repo=repo, persist=True)
        repo.audit("scheduler", "rescore.completed", "predictions", None, {"customers": len(batch.records)})
        out[org_id] = len(batch.records)
    return {"rescored": out}


def monitoring_snapshot(job: dict, ctx: JobContext) -> dict[str, Any]:
    from retentioniq.monitoring.summary import monitoring_summary
    svc = ctx.service
    out = {}
    for org in Accounts(ctx.queue.engine).list_orgs():
        repo = Repository(org["org_id"])
        if repo.count("predictions") == 0:
            continue
        s = monitoring_summary(repo, svc)
        high = [a for a in s["alerts"] if a["severity"] == "high"]
        if high:
            repo.emit("alert.raised", {"alerts": high, "window": s["window"], "model_version": svc.model.version})
        out[org["slug"]] = len(s["alerts"])
    return {"alerts_by_org": out}


def retrain_model(job: dict, ctx: JobContext) -> dict[str, Any]:
    from retentioniq.models.train import train
    p = job["payload"]
    logs: list[str] = []
    result = train(log=logs.append, tune=bool(p.get("tune")), feedback_org=job.get("org_id") if p.get(
        "use_feedback", True) else None, actor=p.get("actor", "system"))
    if job.get("org_id"):
        Repository(job["org_id"]).audit(p.get("actor", "system"), "model.trained", "model", result["version"],
                                        {"status": result["status"]})
    return {"version": result["version"], "status": result["status"], "log": logs[-12:]}


HANDLERS: dict[str, Callable[[dict, JobContext], dict[str, Any]]] = {
    "batch_score": batch_score,
    "dispatch_webhooks": dispatch_webhooks,
    "inbox_import": inbox_import,
    "scheduled_rescore": scheduled_rescore,
    "monitoring_snapshot": monitoring_snapshot,
    "retrain_model": retrain_model,
}
