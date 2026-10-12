"""RetentionIQ REST API (v1).

    uvicorn api.main:app --reload            # OpenAPI docs at /docs

Authentication: ``Authorization: Bearer riq_...`` (organisation API keys, created by an
admin in the app). Every request is scoped to the key's organisation and checked
against its role. ``/health`` and ``/metrics`` are unauthenticated.
"""

import json
import logging
import time
import uuid

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from api import schemas
from api.deps import get_service, repo_for, require
from retentioniq import __version__, config
from retentioniq.auth.rbac import Principal
from retentioniq.data.schema import SchemaError
from retentioniq.decisioning import workflow
from retentioniq.experiments.registry import Experiments
from retentioniq.integrations.webhooks import WebhookError, Webhooks, verify_signature
from retentioniq.jobs.queue import JobQueue
from retentioniq.models import registry
from retentioniq.models.registry import ModelNotAvailable, PromotionError
from retentioniq.storage.blobs import store
from retentioniq.storage.engine import get_engine
from retentioniq.storage.util import now_iso


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": self.formatTime(record), "level": record.levelname, "msg": record.getMessage()}
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload)


logger = logging.getLogger("retentioniq.api")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(JsonFormatter())
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

REQUESTS = Counter("retentioniq_http_requests_total", "HTTP requests", ["method", "route", "status"])
LATENCY = Histogram("retentioniq_http_request_seconds", "HTTP latency", ["route"],
                    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10))
PREDICTIONS = Counter("retentioniq_predictions_total", "Customers scored", ["channel", "risk_tier"])
QUEUE_DEPTH = Gauge("retentioniq_job_queue", "Jobs by status", ["status"])
MODEL_INFO = Gauge("retentioniq_model_info", "Loaded model version", ["version", "role"])

app = FastAPI(
    title="RetentionIQ API", version=__version__,
    description="Explainable churn-risk scoring, governed retention actions, experiments, integrations and "
                "monitoring. Scores are estimates; recommendations are advisory.",
)
_origins = config.settings().get("api", {}).get("cors_origins") or []
if _origins:
    app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["GET", "POST", "PATCH"],
                       allow_headers=["Authorization", "Content-Type", "X-Request-ID"])

_latencies: list[float] = []


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:  # pragma: no cover
        logger.exception("unhandled", extra={"extra_fields": {"request_id": request_id}})
        response = JSONResponse({"detail": "Internal error", "request_id": request_id}, status_code=500)
    elapsed = time.perf_counter() - started
    route = request.scope.get("route")
    route_path = getattr(route, "path", "unmatched")
    REQUESTS.labels(request.method, route_path, str(response.status_code)).inc()
    LATENCY.labels(route_path).observe(elapsed)
    _latencies.append(elapsed * 1000)
    del _latencies[:-1000]
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    principal = getattr(request.state, "principal", None)
    logger.info("request", extra={"extra_fields": {
        "request_id": request_id, "method": request.method, "route": route_path, "status": response.status_code,
        "duration_ms": round(elapsed * 1000, 1), "org_id": getattr(principal, "org_id", None),
        "actor": getattr(principal, "actor", None)}})
    return response


@app.exception_handler(SchemaError)
async def _schema_error(_: Request, exc: SchemaError):
    return JSONResponse({"detail": str(exc)}, status_code=422)


@app.exception_handler(ModelNotAvailable)
async def _model_missing(_: Request, exc: ModelNotAvailable):
    return JSONResponse({"detail": str(exc)}, status_code=503)


def _out(record: dict) -> dict:
    return {**{k: record[k] for k in (
        "prediction_id", "customer_id", "model_version", "scored_at", "churn_probability", "risk_tier",
        "above_operating_threshold", "borderline", "explanation_method", "explanations", "recommendation",
        "data_quality")}, "revenue_at_risk_estimate": round(record["revenue_at_risk"], 2)}


def _count(records: list[dict], channel: str) -> None:
    for r in records:
        PREDICTIONS.labels(channel, r["risk_tier"]).inc()


# ---------------------------------------------------------------- ops
@app.get("/health", response_model=schemas.Health, tags=["ops"])
def health():
    db_ok, version, challenger, detail, depth = True, None, None, None, None
    try:
        depth = JobQueue().depth()
    except Exception as exc:  # pragma: no cover
        db_ok, detail = False, f"database: {exc}"
    try:
        svc = get_service()
        version = svc.model.version
        challenger = svc.challenger.version if svc.challenger else None
    except Exception as exc:
        detail = f"model: {exc}"
    ok = db_ok and version is not None
    body = schemas.Health(status="ok" if ok else "degraded", model_loaded=version is not None,
                          model_version=version, challenger_version=challenger, database=db_ok, queue=depth,
                          detail=detail).model_dump()
    return JSONResponse(body, status_code=200 if ok else 503)


@app.get("/metrics", include_in_schema=False)
def metrics():
    try:
        for s, n in JobQueue().depth().items():
            QUEUE_DEPTH.labels(s).set(n)
        MODEL_INFO.clear()
        svc = get_service()
        MODEL_INFO.labels(svc.model.version, "champion").set(1)
        if svc.challenger:
            MODEL_INFO.labels(svc.challenger.version, "challenger").set(1)
    except Exception:  # metrics must never fail the scrape
        pass
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/model-info", response_model=schemas.ModelInfo, tags=["model"])
def model_info(_: Principal = Depends(require("read"))):
    return get_service().model_info()


# -------------------------------------------------------- predictions
@app.post("/v1/predictions", response_model=schemas.PredictionResponse, tags=["predictions"])
def predict(customer: schemas.CustomerFeatures, persist: bool = False, p: Principal = Depends(require("score"))):
    if persist and not p.can("persist_scores"):
        raise HTTPException(403, "Role cannot persist scores")
    repo = repo_for(p)
    batch = get_service().score(customer.model_dump(), source="api", repo=repo, persist=persist)
    if not batch.records:
        raise HTTPException(422, batch.validation.rejected["errors"].iloc[0])
    _count(batch.records, "api")
    if persist:
        repo.audit(p.actor, "prediction.created", "prediction", batch.records[0]["prediction_id"])
    return _out(batch.records[0])


@app.post("/v1/predictions/batch", response_model=schemas.BatchResponse, tags=["predictions"])
def predict_batch(req: schemas.BatchRequest, p: Principal = Depends(require("score"))):
    max_rows = min(config.settings()["uploads"]["max_rows"], 5000)
    if len(req.customers) > max_rows:
        raise HTTPException(413, f"At most {max_rows} customers per synchronous request; use /v1/jobs/batch-scoring")
    svc = get_service()
    batch = svc.score([c.model_dump() for c in req.customers], source="api_batch", repo=repo_for(p),
                      persist=req.persist and p.can("persist_scores"))
    _count(batch.records, "api_batch")
    rejected = batch.validation.rejected
    return {"model_version": svc.model.version, "rows_total": batch.validation.summary["rows_total"],
            "rows_accepted": len(batch.records), "rows_rejected": len(rejected),
            "latency_ms": round(batch.latency_ms, 1), "predictions": [_out(r) for r in batch.records],
            "rejected": [] if rejected.empty else rejected[["row_number", "errors"]].to_dict(orient="records")}


@app.get("/v1/predictions/{prediction_id}", tags=["predictions"])
def get_prediction(prediction_id: str, p: Principal = Depends(require("read"))):
    record = repo_for(p).get_prediction(prediction_id)
    if record is None:
        raise HTTPException(404, "Prediction not found")
    return record


@app.get("/v1/customers/{customer_id}/retention-summary", tags=["customers"])
def retention_summary(customer_id: str, p: Principal = Depends(require("read"))):
    repo = repo_for(p)
    history = repo.customer_history(customer_id)
    if history.empty:
        raise HTTPException(404, "No scores stored for this customer")
    return {"customer_id": customer_id, "latest": repo.get_prediction(history.iloc[-1]["prediction_id"]),
            "score_history": json.loads(history.to_json(orient="records")),
            "actions": json.loads(repo.actions(customer_id).to_json(orient="records"))}


# ------------------------------------------------------------ actions
@app.post("/v1/actions/recommend", response_model=schemas.ActionOut, tags=["actions"])
def recommend_action(req: schemas.RecommendRequest, p: Principal = Depends(require("read"))):
    repo = repo_for(p)
    record = repo.get_prediction(req.prediction_id)
    if record is None:
        raise HTTPException(404, "Prediction not found (score with persist=true first)")
    rec = workflow.recommendation_for(record).as_dict()
    if not req.create_action:
        return {"action_id": None, "prediction_id": req.prediction_id, "customer_id": record["customer_id"],
                "recommendation": rec}
    if not p.can("log_action"):
        raise HTTPException(403, "Role cannot log actions")
    try:
        act = workflow.create_action(repo, record, p.actor, req.note, recommendation=rec)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None
    return {"action_id": act["action_id"], "prediction_id": req.prediction_id, "customer_id": record["customer_id"],
            "experiment_id": act["experiment_id"], "arm": act["arm"], "recommendation": rec}


@app.patch("/v1/actions/{action_id}", tags=["actions"])
def update_action(action_id: str, req: schemas.ActionStatusRequest, p: Principal = Depends(require("log_action"))):
    repo = repo_for(p)
    action = repo.get_action(action_id)
    if action is None:
        raise HTTPException(404, "Action not found")
    needs_manager = req.status in ("approved", "rejected") or (
        req.status == "in_progress" and action["requires_approval"] and action["status"] == "recommended")
    if needs_manager and not p.can("approve_action"):
        raise HTTPException(403, "Manager approval required")
    if action["status"] == "held_out":
        raise HTTPException(409, "Control-group actions are held out and cannot be actioned")
    repo.update_action_status(action_id, req.status, p.actor, req.note)
    return repo.get_action(action_id)


@app.post("/v1/actions/{action_id}/outcome", status_code=status.HTTP_201_CREATED, tags=["actions"])
def record_outcome(action_id: str, req: schemas.OutcomeRequest, p: Principal = Depends(require("record_outcome"))):
    try:
        outcome_id = repo_for(p).record_outcome(action_id, req.outcome_type, p.actor, req.outcome_window_days,
                                                label_source="api")
    except KeyError:
        raise HTTPException(404, "Action not found") from None
    return {"outcome_id": outcome_id, "action_id": action_id}


# --------------------------------------------------------------- jobs
@app.post("/v1/jobs/batch-scoring", response_model=schemas.JobOut, status_code=202, tags=["jobs"])
async def submit_batch_job(file: UploadFile = File(...), persist: bool = Form(True),
                           p: Principal = Depends(require("score"))):
    max_bytes = config.settings()["uploads"]["max_megabytes"] * 1024 * 1024
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, "File too large")
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(415, "Only .csv files are accepted")
    queue = JobQueue()
    job_id = f"job_{uuid.uuid4().hex[:16]}"
    uri = store().put(f"{p.org_id}/uploads/{job_id}.csv", data)
    queue.enqueue("batch_score", {"upload_uri": uri, "filename": file.filename, "persist": persist and p.can(
        "persist_scores"), "source": "api_job", "actor": p.actor, "batch_job_id": job_id},
        org_id=p.org_id, created_by=p.actor, job_id=job_id)
    repo_for(p).audit(p.actor, "job.submitted", "job", job_id, {"type": "batch_score"})
    return {"job_id": job_id, "job_type": "batch_score", "status": "queued"}


@app.get("/v1/jobs/{job_id}", response_model=schemas.JobOut, tags=["jobs"])
def get_job(job_id: str, p: Principal = Depends(require("read"))):
    job = JobQueue().get(job_id)
    if job is None or job["org_id"] != p.org_id:
        raise HTTPException(404, "Job not found")
    return {k: job.get(k) for k in ("job_id", "job_type", "status", "progress", "result", "error")}


@app.get("/v1/jobs/{job_id}/results", tags=["jobs"])
def job_results(job_id: str, kind: str = "scored", p: Principal = Depends(require("export"))):
    job = JobQueue().get(job_id)
    if job is None or job["org_id"] != p.org_id or job["status"] != "succeeded":
        raise HTTPException(404, "No results for this job")
    uri = (job["result"] or {}).get("error_report_uri" if kind == "errors" else "result_uri")
    if not uri:
        raise HTTPException(404, "No such result file")
    return Response(store().get(uri), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{job_id}_{kind}.csv"'})


# ------------------------------------------------------- integrations
@app.get("/v1/webhooks", tags=["integrations"])
def list_webhooks(p: Principal = Depends(require("manage_integrations"))):
    return json.loads(Webhooks(repo_for(p)).list().to_json(orient="records"))


@app.post("/v1/webhooks", status_code=201, tags=["integrations"])
def create_webhook(req: schemas.WebhookCreate, p: Principal = Depends(require("manage_integrations"))):
    try:
        webhook_id, secret = Webhooks(repo_for(p)).create(req.name, req.url, req.event_types, p.actor)
    except WebhookError as exc:
        raise HTTPException(422, str(exc)) from None
    return {"webhook_id": webhook_id, "signing_secret": secret,
            "note": "Store the signing secret now; it is not shown again."}


@app.get("/v1/webhooks/deliveries", tags=["integrations"])
def deliveries(p: Principal = Depends(require("manage_integrations"))):
    return json.loads(Webhooks(repo_for(p)).deliveries().to_json(orient="records"))


@app.post("/v1/webhooks/deliveries/{delivery_id}/replay", tags=["integrations"])
def replay(delivery_id: str, p: Principal = Depends(require("manage_integrations"))):
    try:
        Webhooks(repo_for(p)).replay(delivery_id, p.actor)
    except KeyError:
        raise HTTPException(404, "Dead delivery not found") from None
    return {"status": "pending"}


@app.post("/v1/integrations/sandbox/receive", tags=["integrations"], include_in_schema=False)
async def sandbox_receiver(request: Request):
    """Reference receiver for testing a real HTTP webhook against this API: verifies the
    signature with the sending webhook's secret and acknowledges idempotently."""
    body = await request.body()
    delivery = request.headers.get("X-RetentionIQ-Delivery", "")
    from sqlalchemy import text
    with get_engine().connect() as c:
        row = c.execute(text("SELECT w.secret FROM webhook_deliveries d JOIN webhooks w ON w.webhook_id = d.webhook_id "
                             "WHERE d.delivery_id = :d"), dict(d=delivery)).first()
    if row is None or not verify_signature(row[0], body, request.headers.get("X-RetentionIQ-Signature", "")):
        raise HTTPException(401, "Invalid signature")
    return {"received": True, "idempotency_key": request.headers.get("Idempotency-Key"), "at": now_iso()}


# -------------------------------------------------------- experiments
@app.get("/v1/experiments", tags=["experiments"])
def list_experiments(p: Principal = Depends(require("read"))):
    return json.loads(Experiments(repo_for(p)).list().to_json(orient="records"))


@app.post("/v1/experiments", status_code=201, tags=["experiments"])
def create_experiment(req: schemas.ExperimentCreate, p: Principal = Depends(require("manage_experiments"))):
    try:
        exp_id = Experiments(repo_for(p)).create(actor=p.actor, **req.model_dump())
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None
    return {"experiment_id": exp_id}


@app.get("/v1/experiments/{experiment_id}/analysis", tags=["experiments"])
def analyze_experiment(experiment_id: str, p: Principal = Depends(require("read"))):
    try:
        result = Experiments(repo_for(p)).analyze(experiment_id)
    except KeyError:
        raise HTTPException(404, "Experiment not found") from None
    from retentioniq.monitoring.summary import json_safe
    return json_safe(result)


# ---------------------------------------------------------- assistant
@app.post("/v1/assistant/query", tags=["assistant"])
def assistant_query(req: schemas.AssistantRequest, p: Principal = Depends(require("use_assistant"))):
    from retentioniq.assistant.semantic import ask
    from retentioniq.monitoring.summary import json_safe
    out = ask(repo_for(p), req.question, p.actor)
    if out["status"] != "ok":
        raise HTTPException(422, out["error"])
    res = out["result"]
    return json_safe({"interpretation": out["interpretation"], "parser": out["parser"], "value": res["value"],
                      "n": res["n"], "suppressed_groups": res["suppressed"],
                      "rows": res["table"].replace({np.nan: None}).to_dict(orient="records")})


# ------------------------------------------------------------- models
@app.get("/v1/models", tags=["model"])
def list_models(_: Principal = Depends(require("read"))):
    return registry.read_registry()


@app.post("/v1/models/retrain", response_model=schemas.JobOut, status_code=202, tags=["model"])
def retrain(req: schemas.RetrainRequest, p: Principal = Depends(require("manage_models"))):
    job_id = JobQueue().enqueue("retrain_model", {**req.model_dump(), "actor": p.actor}, org_id=p.org_id,
                                created_by=p.actor, max_attempts=1)
    repo_for(p).audit(p.actor, "model.retrain_requested", "job", job_id, req.model_dump())
    return {"job_id": job_id, "job_type": "retrain_model", "status": "queued"}


@app.post("/v1/models/{version}/status", tags=["model"])
def set_model_status(version: str, req: schemas.ModelStatusRequest, p: Principal = Depends(require("manage_models"))):
    try:
        registry.set_status(version, req.status, p.actor, req.reason)
    except (PromotionError, ModelNotAvailable) as exc:
        raise HTTPException(409, str(exc)) from None
    repo = repo_for(p)
    repo.audit(p.actor, f"model.{req.status}", "model", version, {"reason": req.reason})
    if req.status == "production":
        repo.emit("model.promoted", {"version": version, "by": p.actor, "reason": req.reason})
    return {"version": version, "status": req.status}


# --------------------------------------------------------- monitoring
@app.get("/v1/monitoring/summary", tags=["ops"])
def monitoring_summary(window_days: int | None = None, p: Principal = Depends(require("read"))):
    from retentioniq.monitoring.summary import monitoring_summary as summarize
    service = {"p50_ms": round(float(np.percentile(_latencies, 50)), 1) if _latencies else None,
               "p95_ms": round(float(np.percentile(_latencies, 95)), 1) if _latencies else None,
               "queue": JobQueue().depth()}
    return {"service": service, **summarize(repo_for(p), get_service(), window_days=window_days)}


@app.get("/", include_in_schema=False)
def root():
    return PlainTextResponse(f"RetentionIQ API {__version__} - see /docs")
