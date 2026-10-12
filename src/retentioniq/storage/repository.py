"""Tenant-scoped data access. Every query is filtered by the repository's
``org_id``; there is no method that reads across tenants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import Connection, Engine, text

from retentioniq.storage.engine import get_engine
from retentioniq.storage.util import new_id, now_iso

ACTION_STATUSES = ("recommended", "approved", "rejected", "in_progress", "held_out", "completed", "cancelled")
OUTCOME_TYPES = ("retained", "churned", "no_response", "declined", "opted_out")
EVENT_TYPES = ("customer.high_risk", "customer.risk_rising", "action.created", "action.approved",
               "action.outcome_recorded", "batch.completed", "alert.raised", "model.promoted")


class Repository:
    def __init__(self, org_id: str, engine: Engine | None = None):
        if not org_id:
            raise ValueError("org_id is required")
        self.org_id = org_id
        self.engine = engine or get_engine()

    # ---------------------------------------------------------- helpers
    def query_df(self, sql: str, params: dict | None = None) -> pd.DataFrame:
        params = {"org": self.org_id, **(params or {})}
        with self.engine.connect() as c:
            return pd.read_sql_query(text(sql), c, params=params)

    def execute(self, sql: str, params: dict | None = None, conn: Connection | None = None) -> int:
        params = {"org": self.org_id, **(params or {})}
        if conn is not None:
            return conn.execute(text(sql), params).rowcount
        with self.engine.begin() as c:
            return c.execute(text(sql), params).rowcount

    def scalar(self, sql: str, params: dict | None = None) -> Any:
        with self.engine.connect() as c:
            return c.execute(text(sql), {"org": self.org_id, **(params or {})}).scalar()

    # ------------------------------------------------------------ audit
    def audit(self, actor: str, event_type: str, target_type: str | None = None,
              target_id: str | None = None, metadata: dict | None = None,
              conn: Connection | None = None) -> None:
        self.execute(
            "INSERT INTO audit_events (event_id, org_id, ts, actor, event_type, target_type, target_id, metadata) "
            "VALUES (:i, :org, :t, :a, :e, :tt, :ti, :m)",
            dict(i=new_id("evt"), t=now_iso(), a=actor, e=event_type, tt=target_type, ti=target_id,
                 m=json.dumps(metadata or {}, default=str)), conn=conn)

    def audit_log(self, limit: int = 500) -> pd.DataFrame:
        return self.query_df("SELECT * FROM audit_events WHERE org_id = :org ORDER BY ts DESC LIMIT :n",
                             dict(n=limit))

    # ----------------------------------------------- integration events
    def emit(self, event_type: str, payload: dict[str, Any], conn: Connection | None = None) -> str | None:
        """Transactional outbox: the event and its pending deliveries are written in the
        same transaction as the business change; a worker delivers them later."""
        if event_type not in EVENT_TYPES:
            raise ValueError(f"Unknown event type {event_type}")

        def _do(c: Connection) -> str | None:
            hooks = c.execute(text("SELECT webhook_id, event_types FROM webhooks WHERE org_id = :org AND active = 1"),
                              {"org": self.org_id}).mappings().all()
            targets = [h["webhook_id"] for h in hooks if event_type in json.loads(h["event_types"])]
            if not targets:
                return None
            event_id = new_id("whe")
            ts = now_iso()
            c.execute(text("INSERT INTO outbox_events (event_id, org_id, event_type, payload, created_at) "
                           "VALUES (:i, :org, :e, :p, :t)"),
                      dict(i=event_id, org=self.org_id, e=event_type,
                           p=json.dumps({"event_id": event_id, "type": event_type, "created_at": ts,
                                         "org_id": self.org_id, "data": payload}, default=str), t=ts))
            for wid in targets:
                c.execute(text("INSERT INTO webhook_deliveries (delivery_id, org_id, event_id, webhook_id, status, "
                               "attempts, next_attempt_at) VALUES (:d, :org, :e, :w, 'pending', 0, :t)"),
                          dict(d=new_id("dlv"), org=self.org_id, e=event_id, w=wid, t=ts))
            return event_id

        if conn is not None:
            return _do(conn)
        with self.engine.begin() as c:
            return _do(c)

    # ------------------------------------------------------ predictions
    def save_predictions(self, records: list[dict[str, Any]], shadow: list[dict[str, Any]] | None = None) -> None:
        if not records:
            return
        org = self.org_id
        with self.engine.begin() as c:
            c.execute(text(
                "INSERT INTO predictions (prediction_id, org_id, customer_id, churn_probability, risk_tier, "
                "model_version, scored_at, data_quality_status, data_quality_warnings, borderline, source, "
                "batch_job_id) VALUES (:prediction_id, :org, :customer_id, :p, :tier, :mv, :ts, :dq, :dqw, :b, "
                ":src, :job)"),
                [dict(prediction_id=r["prediction_id"], org=org, customer_id=r.get("customer_id"),
                      p=r["churn_probability"], tier=r["risk_tier"], mv=r["model_version"], ts=r["scored_at"],
                      dq=r["data_quality"]["status"], dqw=json.dumps(r["data_quality"]["warnings"]),
                      b=int(r.get("borderline", False)), src=r["source"], job=r.get("batch_job_id"))
                 for r in records])
            c.execute(text(
                "INSERT INTO customer_features_snapshot (prediction_id, org_id, customer_id, snapshot_at, "
                "schema_version, features) VALUES (:i, :org, :c, :t, :s, :f)"),
                [dict(i=r["prediction_id"], org=org, c=r.get("customer_id"), t=r["scored_at"],
                      s=r["schema_version"], f=json.dumps(r["features"], default=str)) for r in records])
            expl = [dict(i=r["prediction_id"], rank=k, org=org, f=d["feature"], c=d["contribution"],
                         d=d["direction"], r=d["reason"], m=r["explanation_method"])
                    for r in records for k, d in enumerate(r["explanations"])]
            if expl:
                c.execute(text(
                    "INSERT INTO prediction_explanations (prediction_id, rank, org_id, feature, contribution, "
                    "direction, reason, method) VALUES (:i, :rank, :org, :f, :c, :d, :r, :m)"), expl)
            if shadow:
                c.execute(text(
                    "INSERT INTO shadow_predictions (prediction_id, model_version, org_id, churn_probability, "
                    "scored_at) VALUES (:i, :mv, :org, :p, :t)"),
                    [dict(s, org=org) for s in shadow])

    def get_prediction(self, prediction_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as c:
            p = dict(org=self.org_id, i=prediction_id)
            row = c.execute(text("SELECT * FROM predictions WHERE prediction_id = :i AND org_id = :org"),
                            p).mappings().first()
            if row is None:
                return None
            snap = c.execute(text("SELECT features FROM customer_features_snapshot "
                                  "WHERE prediction_id = :i AND org_id = :org"), p).mappings().first()
            expl = c.execute(text(
                "SELECT feature, contribution, direction, reason, method FROM prediction_explanations "
                "WHERE prediction_id = :i AND org_id = :org ORDER BY rank"), p).mappings().all()
        out = dict(row)
        out["data_quality_warnings"] = json.loads(out["data_quality_warnings"] or "[]")
        out["features"] = json.loads(snap["features"]) if snap else None
        out["explanations"] = [dict(e) for e in expl]
        return out

    def _flatten(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        feats = pd.json_normalize(df["features"].map(json.loads))
        return pd.concat([df.drop(columns=["features"]).reset_index(drop=True), feats], axis=1)

    def latest_scores(self) -> pd.DataFrame:
        """Most recent prediction per customer with features, top reasons and the
        previous score (for risk-trend detection)."""
        df = self.query_df(
            """
            WITH ranked AS (
                SELECT p.prediction_id, p.customer_id, p.churn_probability, p.scored_at,
                       ROW_NUMBER() OVER (PARTITION BY p.customer_id ORDER BY p.scored_at DESC,
                                          p.prediction_id DESC) AS rn
                FROM predictions p WHERE p.org_id = :org AND p.customer_id IS NOT NULL
            )
            SELECT p.*, s.features, prev.churn_probability AS previous_probability,
                   prev.scored_at AS previous_scored_at
            FROM ranked r
            JOIN predictions p ON p.prediction_id = r.prediction_id
            JOIN customer_features_snapshot s ON s.prediction_id = p.prediction_id
            LEFT JOIN ranked prev ON prev.customer_id = r.customer_id AND prev.rn = 2
            WHERE r.rn = 1
            """)
        if df.empty:
            return df
        df = self._flatten(df)
        df["risk_change"] = df["churn_probability"] - df["previous_probability"]
        reasons = self.query_df(
            "SELECT e.prediction_id, e.rank, e.reason FROM prediction_explanations e "
            "WHERE e.org_id = :org AND e.direction = 'increases_risk' ORDER BY e.prediction_id, e.rank")
        top = (reasons.groupby("prediction_id")["reason"]
               .apply(lambda s: " · ".join(s.head(3))).rename("top_reasons"))
        return df.merge(top, how="left", left_on="prediction_id", right_index=True)

    def scores_window(self, since: str | None = None, batch_job_id: str | None = None) -> pd.DataFrame:
        sql = ("SELECT p.prediction_id, p.churn_probability, p.data_quality_status, p.scored_at, s.features "
               "FROM predictions p JOIN customer_features_snapshot s ON s.prediction_id = p.prediction_id "
               "WHERE p.org_id = :org")
        params: dict[str, Any] = {}
        if since:
            sql += " AND p.scored_at >= :since"
            params["since"] = since
        if batch_job_id:
            sql += " AND p.batch_job_id = :job"
            params["job"] = batch_job_id
        return self._flatten(self.query_df(sql, params))

    def customer_ids(self) -> list[str]:
        return self.query_df("SELECT DISTINCT customer_id FROM predictions WHERE org_id = :org "
                             "AND customer_id IS NOT NULL ORDER BY customer_id")["customer_id"].tolist()

    def customer_history(self, customer_id: str) -> pd.DataFrame:
        return self.query_df(
            "SELECT prediction_id, scored_at, churn_probability, risk_tier, model_version, data_quality_status, "
            "source FROM predictions WHERE org_id = :org AND customer_id = :c ORDER BY scored_at",
            dict(c=customer_id))

    def shadow_comparison(self) -> pd.DataFrame:
        return self.query_df(
            "SELECT p.prediction_id, p.customer_id, p.churn_probability AS champion, p.model_version AS champion_version, "
            "s.churn_probability AS challenger, s.model_version AS challenger_version, p.scored_at "
            "FROM shadow_predictions s JOIN predictions p ON p.prediction_id = s.prediction_id "
            "WHERE s.org_id = :org")

    # ---------------------------------------------------------- actions
    def create_action(self, *, prediction_id: str | None, customer_id: str | None,
                      recommendation: dict[str, Any], actor: str, note: str = "",
                      experiment_id: str | None = None, arm: str | None = None) -> str:
        action_id = new_id("act")
        ts = now_iso()
        arm = arm or recommendation.get("experiment_arm", "treatment")
        status = "held_out" if arm == "control" else "recommended"
        with self.engine.begin() as c:
            self.execute(
                "INSERT INTO retention_actions (action_id, org_id, prediction_id, customer_id, recommendation_code, "
                "recommendation_name, policy_version, requires_approval, status, experiment_id, experiment_arm, "
                "expected_value, owner, note, created_at, updated_at) VALUES (:a, :org, :p, :c, :code, :name, :pv, "
                ":ra, :st, :ex, :arm, :ev, :o, :n, :t, :t)",
                dict(a=action_id, p=prediction_id, c=customer_id, code=recommendation["code"],
                     name=recommendation["name"], pv=recommendation["policy_version"],
                     ra=int(recommendation["requires_approval"]), st=status, ex=experiment_id, arm=arm,
                     ev=recommendation.get("expected_value"), o=actor, n=note, t=ts), conn=c)
            self.audit(actor, "action.created", "retention_action", action_id,
                       {"code": recommendation["code"], "customer_id": customer_id, "arm": arm}, conn=c)
            if arm != "control":
                self.emit("action.created", {"action_id": action_id, "customer_id": customer_id,
                                             "recommendation": recommendation["code"],
                                             "requires_approval": bool(recommendation["requires_approval"])}, conn=c)
        return action_id

    def update_action_status(self, action_id: str, status: str, actor: str, note: str = "") -> None:
        if status not in ACTION_STATUSES:
            raise ValueError(f"Unknown status {status}")
        with self.engine.begin() as c:
            n = self.execute(
                "UPDATE retention_actions SET status = :s, note = COALESCE(NULLIF(CAST(:n AS VARCHAR), ''), note), "
                "updated_at = :t "
                "WHERE action_id = :a AND org_id = :org", dict(s=status, n=note, t=now_iso(), a=action_id), conn=c)
            if n == 0:
                raise KeyError(action_id)
            self.audit(actor, f"action.{status}", "retention_action", action_id,
                       {"note": note} if note else {}, conn=c)
            if status == "approved":
                row = c.execute(text("SELECT customer_id, recommendation_code FROM retention_actions "
                                     "WHERE action_id = :a"), dict(a=action_id)).mappings().first()
                self.emit("action.approved", {"action_id": action_id, "customer_id": row["customer_id"],
                                              "recommendation": row["recommendation_code"],
                                              "approved_by": actor}, conn=c)

    def get_action(self, action_id: str) -> dict[str, Any] | None:
        df = self.query_df("SELECT * FROM retention_actions WHERE action_id = :a AND org_id = :org", dict(a=action_id))
        return None if df.empty else df.iloc[0].to_dict()

    def record_outcome(self, action_id: str, outcome_type: str, actor: str,
                       window_days: int | None = None, label_source: str = "manual") -> str:
        if outcome_type not in OUTCOME_TYPES:
            raise ValueError(f"Unknown outcome {outcome_type}")
        outcome_id = new_id("out")
        with self.engine.begin() as c:
            row = c.execute(text("SELECT customer_id, experiment_arm FROM retention_actions "
                                 "WHERE action_id = :a AND org_id = :org"),
                            dict(a=action_id, org=self.org_id)).mappings().first()
            if row is None:
                raise KeyError(action_id)
            self.execute(
                "INSERT INTO action_outcomes (outcome_id, org_id, action_id, outcome_type, observed_at, "
                "outcome_window_days, label_source, recorded_by) VALUES (:i, :org, :a, :o, :t, :w, :l, :r) "
                "ON CONFLICT (action_id) DO UPDATE SET outcome_type = excluded.outcome_type, "
                "observed_at = excluded.observed_at, recorded_by = excluded.recorded_by",
                dict(i=outcome_id, a=action_id, o=outcome_type, t=now_iso(), w=window_days, l=label_source,
                     r=actor), conn=c)
            self.execute("UPDATE retention_actions SET status = 'completed', updated_at = :t "
                         "WHERE action_id = :a AND org_id = :org AND status NOT IN ('rejected', 'cancelled')",
                         dict(t=now_iso(), a=action_id), conn=c)
            if outcome_type in ("retained", "churned") and row["customer_id"]:
                self.execute(
                    "INSERT INTO observed_labels (org_id, customer_id, churned, observed_at, label_source) "
                    "VALUES (:org, :c, :y, :t, 'action_outcome') ON CONFLICT (org_id, customer_id) DO UPDATE SET "
                    "churned = excluded.churned, observed_at = excluded.observed_at, label_source = excluded.label_source",
                    dict(c=row["customer_id"], y=int(outcome_type == "churned"), t=now_iso()), conn=c)
            self.audit(actor, "action.outcome_recorded", "retention_action", action_id,
                       {"outcome": outcome_type}, conn=c)
            self.emit("action.outcome_recorded", {"action_id": action_id, "customer_id": row["customer_id"],
                                                  "outcome": outcome_type, "arm": row["experiment_arm"]}, conn=c)
        return outcome_id

    def actions(self, customer_id: str | None = None) -> pd.DataFrame:
        sql = ("SELECT a.*, o.outcome_type, o.observed_at AS outcome_at FROM retention_actions a "
               "LEFT JOIN action_outcomes o ON o.action_id = a.action_id WHERE a.org_id = :org")
        if customer_id:
            return self.query_df(sql + " AND a.customer_id = :c ORDER BY a.created_at DESC", dict(c=customer_id))
        return self.query_df(sql + " ORDER BY a.created_at DESC")

    # ----------------------------------------------------------- labels
    def upsert_labels(self, labels: list[tuple[str, int]], source: str, actor: str) -> int:
        ts = now_iso()
        with self.engine.begin() as c:
            if labels:
                c.execute(text(
                    "INSERT INTO observed_labels (org_id, customer_id, churned, observed_at, label_source) "
                    "VALUES (:org, :c, :y, :t, :s) ON CONFLICT (org_id, customer_id) DO UPDATE SET "
                    "churned = excluded.churned, observed_at = excluded.observed_at, label_source = excluded.label_source"),
                    [dict(org=self.org_id, c=cid, y=int(v), t=ts, s=source) for cid, v in labels])
            self.audit(actor, "labels.imported", "observed_labels", None, {"count": len(labels), "source": source},
                       conn=c)
        return len(labels)

    def labels(self) -> pd.DataFrame:
        return self.query_df("SELECT customer_id, churned, observed_at, label_source FROM observed_labels "
                             "WHERE org_id = :org")

    # ------------------------------------------------------- batch jobs
    def start_batch_job(self, filename: str, actor: str, model_version: str, source: str = "upload",
                        job_id: str | None = None) -> str:
        job_id = job_id or new_id("job")
        safe = "".join(ch for ch in Path(filename).name if ch.isalnum() or ch in "._- ")[:120]
        with self.engine.begin() as c:
            self.execute("INSERT INTO batch_jobs (job_id, org_id, filename_safe, status, model_version, actor, source, "
                         "started_at) VALUES (:j, :org, :f, 'running', :m, :a, :s, :t)",
                         dict(j=job_id, f=safe, m=model_version, a=actor, s=source, t=now_iso()), conn=c)
            self.audit(actor, "batch.started", "batch_job", job_id, {"file": safe, "source": source}, conn=c)
        return job_id

    def finish_batch_job(self, job_id: str, *, status: str, summary: dict[str, Any] | None = None,
                         error: str | None = None, actor: str = "system", error_report_uri: str | None = None,
                         result_uri: str | None = None, high_risk: int | None = None) -> None:
        s = summary or {}
        with self.engine.begin() as c:
            self.execute(
                "UPDATE batch_jobs SET status = :st, rows_total = :rt, rows_accepted = :ra, rows_rejected = :rr, "
                "rows_warning = :rw, completed_at = :t, error = :e, error_report_uri = :eu, result_uri = :ru "
                "WHERE job_id = :j AND org_id = :org",
                dict(st=status, rt=s.get("rows_total"), ra=s.get("rows_accepted"), rr=s.get("rows_rejected"),
                     rw=s.get("rows_with_warnings"), t=now_iso(), e=error, eu=error_report_uri, ru=result_uri,
                     j=job_id), conn=c)
            self.audit(actor, f"batch.{status}", "batch_job", job_id, s, conn=c)
            if status == "completed":
                self.emit("batch.completed", {"batch_job_id": job_id, **{k: s.get(k) for k in (
                    "rows_total", "rows_accepted", "rows_rejected")}, "high_risk": high_risk}, conn=c)

    def batch_jobs(self) -> pd.DataFrame:
        return self.query_df("SELECT * FROM batch_jobs WHERE org_id = :org ORDER BY started_at DESC")

    def get_batch_job(self, job_id: str) -> dict | None:
        df = self.query_df("SELECT * FROM batch_jobs WHERE org_id = :org AND job_id = :j", dict(j=job_id))
        return None if df.empty else df.iloc[0].to_dict()

    def count(self, table: str) -> int:
        allowed = {"predictions", "retention_actions", "action_outcomes", "observed_labels", "audit_events",
                   "batch_jobs", "webhooks", "experiments"}
        if table not in allowed:
            raise ValueError(table)
        return int(self.scalar(f"SELECT COUNT(*) FROM {table} WHERE org_id = :org"))  # noqa: S608

    def reset(self) -> None:
        """Delete all tenant data (admin demo reset). Child tables first."""
        tables = ["sandbox_inbox", "webhook_deliveries", "outbox_events", "assistant_queries", "action_outcomes",
                  "retention_actions", "shadow_predictions", "prediction_explanations",
                  "customer_features_snapshot", "predictions", "observed_labels", "batch_jobs", "experiments",
                  "webhooks", "audit_events"]
        with self.engine.begin() as c:
            for t in tables:
                c.execute(text(f"DELETE FROM {t} WHERE org_id = :org"), {"org": self.org_id})  # noqa: S608
            c.execute(text("DELETE FROM jobs WHERE org_id = :org"), {"org": self.org_id})
