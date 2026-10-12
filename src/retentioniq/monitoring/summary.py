"""Operational summary shared by the API and the Monitoring page."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from retentioniq import config
from retentioniq.data.schema import FEATURE_COLUMNS
from retentioniq.models.evaluate import threshold_metrics

RUNBOOK = "docs/runbooks.md"


def json_safe(obj: Any) -> Any:
    """Recursively convert numpy scalars and NaN/inf into JSON-safe values."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def delayed_label_performance(scores: pd.DataFrame, labels: pd.DataFrame,
                              threshold: float) -> dict[str, Any]:
    min_n = config.settings()["monitoring"]["min_labels_for_performance"]
    if scores.empty or labels.empty:
        return {"status": "no_labels", "n": 0}
    joined = scores.merge(labels, on="customer_id", how="inner")
    n = len(joined)
    if n < min_n or joined["churned"].nunique() < 2:
        return {"status": "insufficient", "n": n, "required": min_n}
    y = joined["churned"].to_numpy(int)
    p = joined["churn_probability"].to_numpy(float)
    tm = threshold_metrics(y, p, threshold)
    return {"status": "ok", "n": n, "observed_churn_rate": float(y.mean()),
            "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "precision": tm["precision"], "recall": tm["recall"],
            "mean_predicted": float(p.mean())}


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def impact_by_arm(actions: pd.DataFrame, min_per_arm: int = 30) -> dict[str, Any]:
    """Retention rate by experiment arm among actions with a final outcome."""
    if actions.empty or "outcome_type" not in actions:
        return {"status": "no_outcomes", "arms": []}
    final = actions[actions["outcome_type"].isin(["retained", "churned"])]
    arms = []
    for arm in ("treatment", "control"):
        sub = final[final["experiment_arm"] == arm]
        n, k = len(sub), int((sub["outcome_type"] == "retained").sum())
        lo, hi = _wilson(k, n)
        arms.append({"arm": arm, "n": n, "retained": k,
                     "retention_rate": k / n if n else None, "ci_low": lo, "ci_high": hi})
    enough = all(a["n"] >= min_per_arm for a in arms)
    out: dict[str, Any] = {"status": "ok" if enough else "insufficient", "arms": arms,
                           "min_per_arm": min_per_arm}
    if enough:
        out["difference"] = arms[0]["retention_rate"] - arms[1]["retention_rate"]
    return out


def shadow_summary(db, labels: pd.DataFrame) -> dict[str, Any]:
    """Champion vs challenger agreement (and accuracy once labels exist)."""
    df = db.shadow_comparison()
    if df.empty:
        return {"status": "no_challenger"}
    out: dict[str, Any] = {
        "status": "ok", "n": int(len(df)), "challenger_version": df["challenger_version"].iloc[-1],
        "mean_abs_difference": float((df["champion"] - df["challenger"]).abs().mean()),
        "correlation": float(df["champion"].corr(df["challenger"])) if len(df) > 2 else None,
    }
    joined = df.merge(labels, on="customer_id", how="inner").drop_duplicates("customer_id", keep="last")
    if len(joined) >= 50 and joined["churned"].nunique() == 2:
        y = joined["churned"].to_numpy(int)
        out["labelled"] = int(len(joined))
        out["champion_pr_auc"] = float(average_precision_score(y, joined["champion"]))
        out["challenger_pr_auc"] = float(average_precision_score(y, joined["challenger"]))
    return out


def monitoring_summary(db, service, *, window_days: int | None = None,
                       batch_job_id: str | None = None) -> dict[str, Any]:
    """``window_days``/``batch_job_id`` choose the population compared against the
    training reference for drift (defaults to settings.monitoring.drift_window_days)."""
    m = config.settings()["monitoring"]
    latest = db.latest_scores()
    if batch_job_id:
        window_label = f"batch job {batch_job_id}"
        window = db.scores_window(batch_job_id=batch_job_id)
    else:
        window_days = window_days or m["drift_window_days"]
        since = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat(timespec="seconds")
        window_label = f"last {window_days} day(s)"
        window = db.scores_window(since=since)
    labels = db.labels()[["customer_id", "churned"]]
    jobs = db.batch_jobs()
    alerts: list[dict[str, Any]] = []

    drift = pd.DataFrame()
    dq_rate = None
    score_mean = None
    drift_status = "no_data"
    if not window.empty:
        if len(window) >= m["min_rows_for_drift"]:
            drift = service.drift(window[FEATURE_COLUMNS], window["churn_probability"].to_numpy())
            drift_status = "ok"
        else:
            drift_status = "insufficient"
        dq_rate = float((window["data_quality_status"] != "passed").mean())
        score_mean = float(window["churn_probability"].mean())
        for _, row in (drift[drift["status"] != "stable"] if not drift.empty else drift).iterrows():
            alerts.append({
                "severity": "high" if row["status"] == "alert" else "medium",
                "signal": "data_drift", "owner": "ML engineer",
                "message": f"PSI {row['psi']:.2f} on {row['feature']} ({window_label}) vs training reference",
                "runbook": f"{RUNBOOK}#data-drift"})
    if not latest.empty:
        if dq_rate is not None and dq_rate > m["dq_warning_rate_alert"]:
            alerts.append({"severity": "medium", "signal": "data_quality", "owner": "Data analyst",
                           "message": f"{dq_rate:.0%} of predictions ({window_label}) have data-quality warnings",
                           "runbook": f"{RUNBOOK}#data-quality"})
    if len(window) >= 20 and window["churn_probability"].round(6).nunique() <= 1:
        alerts.append({"severity": "high", "signal": "prediction_behaviour", "owner": "ML engineer",
                       "message": f"All scores in {window_label} are identical", "runbook": f"{RUNBOOK}#scores"})

    perf = delayed_label_performance(latest, labels, service.model.threshold)
    if perf["status"] == "ok":
        ref = service.model.evaluation.get("test", {})
        if ref and perf["pr_auc"] < ref.get("pr_auc", 0) - 0.10:
            alerts.append({"severity": "high", "signal": "model_performance", "owner": "ML engineer",
                           "message": f"Delayed-label PR-AUC {perf['pr_auc']:.2f} is >0.10 below "
                                      f"the test baseline {ref['pr_auc']:.2f}",
                           "runbook": f"{RUNBOOK}#performance"})

    batch = {"jobs": int(len(jobs))}
    if not jobs.empty:
        done = jobs[jobs["status"] == "completed"]
        total = done["rows_total"].fillna(0).sum()
        batch.update({
            "failed_jobs": int((jobs["status"] == "failed").sum()),
            "rejected_row_rate": float(done["rows_rejected"].fillna(0).sum() / total) if total else None,
        })
        if batch["failed_jobs"]:
            alerts.append({"severity": "medium", "signal": "batch_operations", "owner": "Platform admin",
                           "message": f"{batch['failed_jobs']} batch job(s) failed",
                           "runbook": f"{RUNBOOK}#batch-jobs"})

    impact = impact_by_arm(db.actions())
    from retentioniq.integrations.webhooks import summary as webhook_summary
    deliveries = webhook_summary(db)
    if deliveries.get("dead"):
        alerts.append({"severity": "medium", "signal": "integrations", "owner": "Platform admin",
                       "message": f"{deliveries['dead']} webhook deliveries are dead-lettered",
                       "runbook": f"{RUNBOOK}#integrations"})
    shadow = shadow_summary(db, labels)
    return json_safe({
        "model_version": service.model.version,
        "customers_scored": int(len(latest)),
        "last_scored_at": latest["scored_at"].max() if not latest.empty else None,
        "mean_score": score_mean,
        "data_quality_warning_rate": dq_rate,
        "window": window_label,
        "window_rows": int(len(window)),
        "drift_status": drift_status,
        "drift_min_rows": m["min_rows_for_drift"],
        "drift": [] if drift.empty else drift.replace({np.nan: None}).to_dict(orient="records"),
        "delayed_label_performance": perf,
        "batch": batch,
        "impact": impact,
        "webhook_deliveries": deliveries,
        "shadow": shadow,
        "alerts": alerts,
    })
