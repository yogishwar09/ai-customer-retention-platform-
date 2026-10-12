"""Relational schema. Every business table carries ``org_id``: tenant isolation is
enforced in the repository layer, never by UI filters."""

from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import Table as _Table

metadata = MetaData()
SCHEMA_VERSION = 2


def Table(name, *cols, **kw):  # noqa: N802 - mirrors sqlalchemy.Table
    return _Table(name, metadata, *cols, **kw)


def ts(name, nullable=True):
    return Column(name, String(32), nullable=nullable)  # ISO-8601 UTC strings, portable + sortable


def org():
    return Column("org_id", String(40), ForeignKey("organizations.org_id"), nullable=False, index=True)


organizations = Table(
    "organizations",
    Column("org_id", String(40), primary_key=True),
    Column("name", String(120), nullable=False),
    Column("slug", String(60), nullable=False, unique=True),
    Column("settings", Text),
    ts("created_at", False),
)

users = Table(
    "users",
    Column("user_id", String(40), primary_key=True),
    org(),
    Column("email", String(254), nullable=False, unique=True),
    Column("display_name", String(120)),
    Column("role", String(20), nullable=False),
    Column("password_hash", String(255), nullable=False),
    Column("active", Integer, nullable=False, default=1),
    Column("failed_attempts", Integer, nullable=False, default=0),
    ts("locked_until"), ts("last_login_at"), ts("created_at", False),
)

api_keys = Table(
    "api_keys",
    Column("key_id", String(40), primary_key=True),
    org(),
    Column("name", String(120), nullable=False),
    Column("role", String(20), nullable=False),
    Column("prefix", String(16), nullable=False, index=True),
    Column("key_hash", String(64), nullable=False),
    Column("created_by", String(254)),
    ts("created_at", False), ts("last_used_at"), ts("revoked_at"),
)

predictions = Table(
    "predictions",
    Column("prediction_id", String(40), primary_key=True),
    org(),
    Column("customer_id", String(64)),
    Column("churn_probability", Float, nullable=False),
    Column("risk_tier", String(10), nullable=False),
    Column("model_version", String(80), nullable=False),
    ts("scored_at", False),
    Column("data_quality_status", String(10), nullable=False),
    Column("data_quality_warnings", Text),
    Column("borderline", Integer, nullable=False, default=0),
    Column("source", String(30), nullable=False),
    Column("batch_job_id", String(40)),
    Index("ix_pred_org_customer", "org_id", "customer_id", "scored_at"),
    Index("ix_pred_org_scored", "org_id", "scored_at"),
)

customer_features_snapshot = Table(
    "customer_features_snapshot",
    Column("prediction_id", String(40), ForeignKey("predictions.prediction_id"), primary_key=True),
    org(),
    Column("customer_id", String(64)),
    ts("snapshot_at", False),
    Column("schema_version", String(16), nullable=False),
    Column("features", Text, nullable=False),
)

prediction_explanations = Table(
    "prediction_explanations",
    Column("prediction_id", String(40), ForeignKey("predictions.prediction_id"), primary_key=True),
    Column("rank", Integer, primary_key=True),
    org(),
    Column("feature", String(40), nullable=False),
    Column("contribution", Float, nullable=False),
    Column("direction", String(20), nullable=False),
    Column("reason", String(200), nullable=False),
    Column("method", String(80), nullable=False),
)

shadow_predictions = Table(
    "shadow_predictions",
    Column("prediction_id", String(40), ForeignKey("predictions.prediction_id"), primary_key=True),
    Column("model_version", String(80), primary_key=True),
    org(),
    Column("churn_probability", Float, nullable=False),
    ts("scored_at", False),
)

retention_actions = Table(
    "retention_actions",
    Column("action_id", String(40), primary_key=True),
    org(),
    Column("prediction_id", String(40), ForeignKey("predictions.prediction_id")),
    Column("customer_id", String(64), index=True),
    Column("recommendation_code", String(40), nullable=False),
    Column("recommendation_name", String(120), nullable=False),
    Column("policy_version", String(40), nullable=False),
    Column("requires_approval", Integer, nullable=False),
    Column("status", String(20), nullable=False),
    Column("experiment_id", String(40)),
    Column("experiment_arm", String(12), nullable=False),
    Column("expected_value", Float),
    Column("owner", String(254)),
    Column("note", Text),
    ts("created_at", False), ts("updated_at", False),
)

action_outcomes = Table(
    "action_outcomes",
    Column("outcome_id", String(40), primary_key=True),
    org(),
    Column("action_id", String(40), ForeignKey("retention_actions.action_id"), nullable=False, unique=True),
    Column("outcome_type", String(20), nullable=False),
    ts("observed_at", False),
    Column("outcome_window_days", Integer),
    Column("label_source", String(60), nullable=False),
    Column("recorded_by", String(254)),
)

observed_labels = Table(
    "observed_labels",
    org(),
    Column("customer_id", String(64), nullable=False),
    Column("churned", Integer, nullable=False),
    ts("observed_at", False),
    Column("label_source", String(80), nullable=False),
    UniqueConstraint("org_id", "customer_id", name="uq_labels_org_customer"),
)

audit_events = Table(
    "audit_events",
    Column("event_id", String(40), primary_key=True),
    org(),
    ts("ts", False),
    Column("actor", String(254), nullable=False),
    Column("event_type", String(60), nullable=False),
    Column("target_type", String(40)),
    Column("target_id", String(80)),
    Column("metadata", Text),
    Index("ix_audit_org_ts", "org_id", "ts"),
)

batch_jobs = Table(
    "batch_jobs",
    Column("job_id", String(40), primary_key=True),
    org(),
    Column("filename_safe", String(160)),
    Column("status", String(20), nullable=False),
    Column("rows_total", Integer), Column("rows_accepted", Integer),
    Column("rows_rejected", Integer), Column("rows_warning", Integer),
    Column("model_version", String(80)),
    Column("actor", String(254)),
    Column("source", String(30)),
    ts("started_at", False), ts("completed_at"),
    Column("error", Text),
    Column("error_report_uri", String(400)),
    Column("result_uri", String(400)),
)

jobs = Table(
    "jobs",
    Column("job_id", String(40), primary_key=True),
    Column("org_id", String(40), ForeignKey("organizations.org_id"), nullable=True, index=True),
    Column("job_type", String(40), nullable=False),
    Column("payload", Text),
    Column("status", String(12), nullable=False),  # queued | running | succeeded | failed | dead
    Column("attempts", Integer, nullable=False, default=0),
    Column("max_attempts", Integer, nullable=False, default=3),
    ts("run_after", False),
    Column("locked_by", String(80)), ts("locked_at"),
    Column("progress", Float),
    Column("result", Text), Column("error", Text),
    Column("created_by", String(254)),
    ts("created_at", False), ts("finished_at"),
    Index("ix_jobs_status_run_after", "status", "run_after"),
)

schedules = Table(
    "schedules",
    Column("name", String(60), primary_key=True),
    Column("interval_seconds", Integer, nullable=False),
    Column("enabled", Integer, nullable=False, default=1),
    ts("last_enqueued_at"),
)

webhooks = Table(
    "webhooks",
    Column("webhook_id", String(40), primary_key=True),
    org(),
    Column("name", String(120), nullable=False),
    Column("url", String(500), nullable=False),
    Column("secret", String(120), nullable=False),
    Column("event_types", Text, nullable=False),
    Column("active", Integer, nullable=False, default=1),
    Column("created_by", String(254)),
    ts("created_at", False),
)

outbox_events = Table(
    "outbox_events",
    Column("event_id", String(40), primary_key=True),
    org(),
    Column("event_type", String(60), nullable=False),
    Column("payload", Text, nullable=False),
    ts("created_at", False),
)

webhook_deliveries = Table(
    "webhook_deliveries",
    Column("delivery_id", String(40), primary_key=True),
    org(),
    Column("event_id", String(40), ForeignKey("outbox_events.event_id"), nullable=False),
    Column("webhook_id", String(40), ForeignKey("webhooks.webhook_id"), nullable=False),
    Column("status", String(12), nullable=False),  # pending | succeeded | dead
    Column("attempts", Integer, nullable=False, default=0),
    ts("next_attempt_at", False),
    Column("last_status_code", Integer),
    Column("last_error", Text),
    ts("delivered_at"),
    UniqueConstraint("event_id", "webhook_id", name="uq_delivery_event_hook"),
    Index("ix_deliveries_due", "status", "next_attempt_at"),
)

sandbox_inbox = Table(
    "sandbox_inbox",
    Column("idempotency_key", String(80), primary_key=True),
    org(),
    Column("event_type", String(60)),
    Column("signature_valid", Integer, nullable=False),
    Column("payload", Text),
    Column("delivery_count", Integer, nullable=False, default=1),
    ts("received_at", False),
)

experiments = Table(
    "experiments",
    Column("experiment_id", String(40), primary_key=True),
    org(),
    Column("name", String(160), nullable=False),
    Column("hypothesis", Text),
    Column("action_code", String(40)),  # NULL = applies to every recommended action
    Column("control_share", Float, nullable=False),
    Column("status", String(12), nullable=False),  # draft | running | stopped | completed
    Column("primary_metric", String(40), nullable=False),
    Column("min_detectable_effect", Float, nullable=False),
    Column("alpha", Float, nullable=False),
    Column("power", Float, nullable=False),
    Column("created_by", String(254)),
    ts("created_at", False), ts("started_at"), ts("ended_at"),
)

assistant_queries = Table(
    "assistant_queries",
    Column("query_id", String(40), primary_key=True),
    org(),
    Column("actor", String(254), nullable=False),
    Column("question", Text, nullable=False),
    Column("interpreted", Text),
    Column("parser", String(20)),
    Column("status", String(12), nullable=False),
    Column("error", Text),
    ts("asked_at", False),
)
