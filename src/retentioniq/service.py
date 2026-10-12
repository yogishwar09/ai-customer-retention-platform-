"""The one prediction service used by the Streamlit UI, the REST API and the
batch scorer. Validation, preprocessing, scoring, explanation and decisioning
all flow through ``PredictionService.score`` so every channel behaves the same.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from retentioniq import config
from retentioniq.data.schema import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    SCHEMA_VERSION,
    ValidationResult,
    validate_frame,
)
from retentioniq.decisioning import policy
from retentioniq.decisioning.risk import revenue_at_risk, risk_tier, tier_cutoffs
from retentioniq.models import registry
from retentioniq.models.explain import Explainer
from retentioniq.storage.repository import Repository
from retentioniq.storage.util import new_id, now_iso


@dataclass
class ScoreBatch:
    records: list[dict[str, Any]]
    validation: ValidationResult
    latency_ms: float

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for r in self.records:
            rows.append({
                "prediction_id": r["prediction_id"],
                ID_COLUMN: r.get("customer_id"),
                "churn_probability": round(r["churn_probability"], 4),
                "risk_tier": r["risk_tier"],
                "revenue_at_risk_estimate": round(r["revenue_at_risk"], 2),
                "reason_codes": " · ".join(d["reason"] for d in r["explanations"]
                                           if d["direction"] == "increases_risk"),
                "recommended_action": r["recommendation"]["code"],
                "requires_approval": r["recommendation"]["requires_approval"],
                "data_quality_status": r["data_quality"]["status"],
                "data_quality_warnings": "; ".join(r["data_quality"]["warnings"]),
                "borderline": r["borderline"],
                "model_version": r["model_version"],
                "scored_at": r["scored_at"],
                **r["features"],
            })
        return pd.DataFrame(rows)


class PredictionService:
    """Stateless with respect to tenants: pass the tenant ``Repository`` per call."""

    def __init__(self, model: registry.LoadedModel | None = None,
                 challenger: registry.LoadedModel | None = None, *, load_challenger: bool = True):
        self.model = model or registry.load()
        if challenger is None and load_challenger:
            staging = registry.staging_version()
            if staging and staging != self.model.version:
                try:
                    challenger = registry.load(staging)
                except Exception:  # a broken challenger must never break production scoring
                    challenger = None
        self.challenger = challenger
        self.explainer = Explainer(self.model.pipeline, self.model.algorithm,
                                   background=self._background())
        self.reference_profile = self._load_reference()

    def _background(self) -> pd.DataFrame | None:
        if self.model.algorithm != "logistic_regression":
            return None
        from retentioniq.data.loader import load_training_data
        df, _ = load_training_data()
        return df[FEATURE_COLUMNS]

    def _load_reference(self) -> dict[str, Any] | None:
        path = config.MODELS_DIR / self.model.version / "reference_profile.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    # ------------------------------------------------------------------ info
    def model_info(self) -> dict[str, Any]:
        m = self.model.metadata
        t = self.model.evaluation.get("test", {})
        return {
            "model_version": self.model.version,
            "algorithm": m["algorithm"],
            "trained_at": m["trained_at"],
            "schema_version": m["schema_version"],
            "operating_threshold": m["operating_threshold"],
            "risk_tiers": tier_cutoffs(m["operating_threshold"]),
            "explanation_method": m["explanation_method"],
            "training_data_fingerprint": m["training_data_fingerprint"],
            "test_metrics": {k: t.get(k) for k in ("roc_auc", "pr_auc", "brier", "prevalence")},
            "release_gate_passed": m["release_gate"]["passed"],
            "policy_version": config.playbooks().get("policy_version"),
            "challenger_version": self.challenger.version if self.challenger else None,
        }

    # --------------------------------------------------------------- scoring
    def score(self, data: pd.DataFrame | list[dict[str, Any]] | dict[str, Any], *,
              source: str = "api", repo: Repository | None = None, persist: bool = False,
              batch_job_id: str | None = None, explain_top_n: int = 5,
              progress: Callable[[float], None] | None = None, chunk_size: int = 1000,
              scored_at: str | None = None) -> ScoreBatch:
        started = time.perf_counter()
        if isinstance(data, dict):
            data = [data]
        frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        validation = validate_frame(frame)
        valid = validation.valid
        records: list[dict[str, Any]] = []
        threshold = self.model.threshold
        margin = config.settings()["decision"]["borderline_margin"]
        scored_at = scored_at or now_iso()

        for start in range(0, len(valid), chunk_size):
            chunk = valid.iloc[start:start + chunk_size]
            X = chunk[FEATURE_COLUMNS]
            probs = self.model.pipeline.predict_proba(X)[:, 1]
            drivers = self.explainer.drivers(X, top_n=explain_top_n)
            for pos, (idx, p) in enumerate(zip(chunk.index, probs, strict=True)):
                p = float(p)
                features = {c: _py(chunk.at[idx, c]) for c in FEATURE_COLUMNS}
                cid = str(chunk.at[idx, ID_COLUMN]) if validation.has_ids else None
                warnings = validation.row_warnings.get(idx, [])
                status = "warning" if warnings else "passed"
                tier = risk_tier(p, threshold)
                borderline = abs(p - threshold) < margin
                rec = policy.recommend(features, p, tier, quality_status=status,
                                       borderline=borderline, customer_id=cid)
                records.append({
                    "prediction_id": new_id("pred"),
                    "customer_id": cid,
                    "model_version": self.model.version,
                    "schema_version": SCHEMA_VERSION,
                    "scored_at": scored_at,
                    "churn_probability": p,
                    "risk_tier": tier,
                    "above_operating_threshold": p >= threshold,
                    "borderline": borderline,
                    "revenue_at_risk": revenue_at_risk(p, features.get("MonthlyCharges")),
                    "explanations": [d.as_dict() for d in drivers[pos]],
                    "explanation_method": self.explainer.method,
                    "recommendation": rec.as_dict(),
                    "data_quality": {"status": status, "warnings": warnings},
                    "features": features,
                    "source": source,
                    "batch_job_id": batch_job_id,
                })
            if progress:
                progress(min(1.0, (start + len(chunk)) / max(len(valid), 1)))

        if persist:
            if repo is None:
                raise ValueError("persist=True requires a tenant repository")
            self.persist(records, repo, valid)
        return ScoreBatch(records, validation, (time.perf_counter() - started) * 1000)

    def persist(self, records: list[dict[str, Any]], repo: Repository, valid: pd.DataFrame | None = None) -> None:
        """Store predictions (+ shadow scores) and emit risk events for tenants' webhooks."""
        previous = self._previous_scores(repo, [r["customer_id"] for r in records if r["customer_id"]])
        shadow = None
        if self.challenger is not None and records:
            X = pd.DataFrame([r["features"] for r in records])[FEATURE_COLUMNS]
            probs = self.challenger.pipeline.predict_proba(X)[:, 1]
            shadow = [{"i": r["prediction_id"], "mv": self.challenger.version, "p": float(p), "t": r["scored_at"]}
                      for r, p in zip(records, probs, strict=True)]
        repo.save_predictions(records, shadow)
        self._emit_risk_events(repo, records, previous)

    @staticmethod
    def _previous_scores(repo: Repository, customer_ids: list[str]) -> dict[str, tuple[float, str]]:
        if not customer_ids:
            return {}
        out: dict[str, tuple[float, str]] = {}
        for start in range(0, len(customer_ids), 500):
            chunk = customer_ids[start:start + 500]
            names = {f"c{i}": c for i, c in enumerate(chunk)}
            df = repo.query_df(
                "SELECT customer_id, churn_probability, risk_tier FROM ("
                "SELECT customer_id, churn_probability, risk_tier, ROW_NUMBER() OVER (PARTITION BY customer_id "
                "ORDER BY scored_at DESC, prediction_id DESC) AS rn FROM predictions WHERE org_id = :org AND "
                f"customer_id IN ({', '.join(':' + k for k in names)})) t WHERE rn = 1", names)
            out.update({r.customer_id: (float(r.churn_probability), r.risk_tier) for r in df.itertuples()})
        return out

    @staticmethod
    def _emit_risk_events(repo: Repository, records: list[dict[str, Any]],
                          previous: dict[str, tuple[float, str]]) -> None:
        if not repo.scalar("SELECT COUNT(*) FROM webhooks WHERE org_id = :org AND active = 1"):
            return
        delta = config.settings()["decision"]["risk_rising_delta"]
        for r in records:
            cid = r["customer_id"]
            if not cid:
                continue
            prev_p, prev_tier = previous.get(cid, (None, None))
            base = {"customer_id": cid, "prediction_id": r["prediction_id"],
                    "churn_probability": round(r["churn_probability"], 4), "risk_tier": r["risk_tier"],
                    "model_version": r["model_version"],
                    "reasons": [d["reason"] for d in r["explanations"] if d["direction"] == "increases_risk"][:3],
                    "recommended_action": r["recommendation"]["code"]}
            if r["risk_tier"] == "high" and prev_tier != "high":
                repo.emit("customer.high_risk", {**base, "previous_tier": prev_tier})
            if prev_p is not None and r["churn_probability"] - prev_p >= delta:
                repo.emit("customer.risk_rising", {**base, "previous_probability": round(prev_p, 4),
                                                   "change": round(r["churn_probability"] - prev_p, 4)})

    def what_if(self, base: dict[str, Any], feature: str, values: list[Any]) -> pd.DataFrame:
        """Model sensitivity: probability as one input varies (not a causal effect)."""
        rows = []
        base = {k: v for k, v in base.items() if k != ID_COLUMN}  # hypotheticals share no identity
        for v in values:
            candidate = dict(base)
            candidate[feature] = v
            rows.append(candidate)
        result = validate_frame(pd.DataFrame(rows))
        if result.valid.empty:
            return pd.DataFrame(columns=[feature, "churn_probability"])
        probs = self.model.pipeline.predict_proba(result.valid[FEATURE_COLUMNS])[:, 1]
        return pd.DataFrame({feature: result.valid[feature].tolist(), "churn_probability": probs})

    def drift(self, frame: pd.DataFrame, scores: np.ndarray | None = None) -> pd.DataFrame:
        from retentioniq.monitoring.drift import drift_report
        if self.reference_profile is None:
            return pd.DataFrame()
        return drift_report(self.reference_profile, frame, scores)


def _py(v: Any) -> Any:
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return None if np.isnan(v) else float(v)
    return v
