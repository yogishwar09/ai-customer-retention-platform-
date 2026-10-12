"""Natural-language analytics over a governed semantic layer.

The question is translated into a *restricted query spec* (one approved metric,
an optional approved dimension, equality filters on approved fields). The spec
is validated and then executed by this module against the caller's tenant data
with pandas - no model-generated SQL is ever run. Small groups are suppressed,
and every question is written to an audit table.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from retentioniq import config
from retentioniq.storage.repository import Repository
from retentioniq.storage.util import new_id, now_iso

METRICS = {
    "customers": "Number of scored customers",
    "high_risk_customers": "Customers in the high risk tier",
    "avg_risk": "Average estimated churn probability",
    "revenue_at_risk": "Estimated annualised revenue at risk (p x monthly charges x 12)",
    "rising_risk_customers": "Customers whose score rose by at least the configured delta",
    "observed_churn_rate": "Churn rate among customers with an outcome label",
    "actions": "Number of retention actions",
    "retention_rate": "Retained / (retained + churned) among actions with a final outcome",
}
CUSTOMER_DIMENSIONS = ["risk_tier", "Contract", "InternetService", "PaymentMethod", "tenure_band", "SeniorCitizen",
                       "PaperlessBilling", "TechSupport", "OnlineSecurity"]
ACTION_DIMENSIONS = ["recommendation", "action_status", "experiment_arm"]
DIMENSIONS = CUSTOMER_DIMENSIONS + ACTION_DIMENSIONS
ACTION_METRICS = {"actions", "retention_rate"}


class QueryError(ValueError):
    pass


@dataclass
class QuerySpec:
    metric: str
    group_by: str = "none"
    filters: list[dict[str, str]] = field(default_factory=list)
    top_n: int = 0

    def validate(self) -> QuerySpec:
        if self.metric not in METRICS:
            raise QueryError(f"Unknown metric '{self.metric}'")
        if self.group_by != "none" and self.group_by not in DIMENSIONS:
            raise QueryError(f"Unknown dimension '{self.group_by}'")
        is_action = self.metric in ACTION_METRICS
        allowed = (ACTION_DIMENSIONS + CUSTOMER_DIMENSIONS) if is_action else CUSTOMER_DIMENSIONS
        if self.group_by != "none" and self.group_by not in allowed:
            raise QueryError(f"'{self.group_by}' cannot be used with metric '{self.metric}'")
        for f in self.filters:
            if f.get("field") not in allowed:
                raise QueryError(f"Filter field '{f.get('field')}' is not allowed")
            if not isinstance(f.get("value"), str) or len(f["value"]) > 60:
                raise QueryError("Filter values must be short strings")
        if not 0 <= int(self.top_n) <= 50:
            raise QueryError("top_n must be between 0 and 50")
        return self

    def describe(self) -> str:
        text = METRICS[self.metric]
        if self.group_by != "none":
            text += f" by {self.group_by}"
        if self.filters:
            text += " where " + " and ".join(f"{f['field']} = {f['value']}" for f in self.filters)
        if self.top_n:
            text += f" (top {self.top_n})"
        return text


# ------------------------------------------------------------------ parsers
_SYNONYMS = {
    "revenue_at_risk": ["revenue", "money", "dollars", "exposure", "at risk revenue"],
    "rising_risk_customers": ["rising", "increasing", "trending up", "getting worse"],
    "high_risk_customers": ["high risk", "high-risk", "most at risk", "risky"],
    "observed_churn_rate": ["churn rate", "actually churned", "observed churn"],
    "retention_rate": ["retention rate", "retained", "saved"],
    "actions": ["actions", "outreach", "campaign", "tasks"],
    "avg_risk": ["average risk", "avg risk", "mean risk", "average score", "average churn"],
    "customers": ["how many customers", "number of customers", "customers"],
}
_DIM_SYNONYMS = {
    "Contract": ["contract"], "InternetService": ["internet", "fiber", "dsl"],
    "PaymentMethod": ["payment", "pay"], "tenure_band": ["tenure", "new customers", "how long"],
    "risk_tier": ["tier"], "SeniorCitizen": ["senior", "age"], "PaperlessBilling": ["paperless", "billing"],
    "TechSupport": ["tech support", "support"], "OnlineSecurity": ["security"],
    "recommendation": ["recommendation", "playbook", "action type", "per action"],
    "action_status": ["status"], "experiment_arm": ["arm", "control", "treatment"],
}
_VALUE_FILTERS = [
    ("Contract", "Month-to-month", ["month-to-month", "month to month", "monthly contract"]),
    ("Contract", "One year", ["one year", "1 year", "annual"]),
    ("Contract", "Two year", ["two year", "2 year"]),
    ("InternetService", "Fiber optic", ["fiber"]),
    ("InternetService", "DSL", ["dsl"]),
    ("PaymentMethod", "Electronic check", ["electronic check", "e-check"]),
    ("risk_tier", "high", ["high risk", "high-risk"]),
    ("SeniorCitizen", "1", ["senior"]),
]


def keyword_parse(question: str) -> QuerySpec:
    q = question.lower()
    metric = next((m for m, words in _SYNONYMS.items() if any(w in q for w in words)), "customers")
    group = "none"
    for m in re.finditer(r"\b(?:by|per|across|for each|split by|broken down by)\s+([a-z\- ]+)", q):
        tail = m.group(1)
        group = next((d for d, words in _DIM_SYNONYMS.items() if any(tail.startswith(w) for w in words)), "none")
        if group != "none":
            break
    if group == "none" and re.search(r"\b(top|which|what)\b", q):
        # "top 3 payment methods ..." / "which contract ..." name the dimension without "by"
        group = next((d for d, words in _DIM_SYNONYMS.items()
                      if any(re.search(rf"\b{re.escape(w)}", q) for w in words)
                      and d not in ("risk_tier",)), "none")
    filters = []
    for fld, value, words in _VALUE_FILTERS:
        if fld == group or (fld == "risk_tier" and metric == "high_risk_customers"):
            continue
        if any(w in q for w in words) and not any(f["field"] == fld for f in filters):
            filters.append({"field": fld, "value": value})
    top = re.search(r"\btop\s+(\d{1,2})\b", q)
    spec = QuerySpec(metric, group, filters, int(top.group(1)) if top else 0)
    if metric not in ACTION_METRICS and group in ACTION_DIMENSIONS:
        spec.group_by = "none"
    return spec.validate()


# ---------------------------------------------------------------- execution
def _tenure_band(t: pd.Series) -> pd.Series:
    return pd.cut(t, [-1, 6, 12, 24, 48, 1000],
                  labels=["0-6 mo", "7-12 mo", "13-24 mo", "25-48 mo", "49+ mo"]).astype(str)


def _customer_frame(repo: Repository) -> pd.DataFrame:
    df = repo.latest_scores()
    if df.empty:
        return df
    df["tenure_band"] = _tenure_band(df["tenure"])
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int).astype(str)
    df["revenue_at_risk"] = df["churn_probability"] * df["MonthlyCharges"] * config.settings()["value"]["horizon_months"]
    labels = repo.labels()[["customer_id", "churned"]]
    return df.merge(labels, on="customer_id", how="left")


def _action_frame(repo: Repository, customers: pd.DataFrame) -> pd.DataFrame:
    a = repo.actions()
    if a.empty:
        return a
    a = a.rename(columns={"recommendation_name": "recommendation", "status": "action_status"})
    cols = ["customer_id", *[c for c in CUSTOMER_DIMENSIONS if c in customers]]
    return a.merge(customers[cols], on="customer_id", how="left") if not customers.empty else a


def execute(repo: Repository, spec: QuerySpec) -> dict[str, Any]:
    min_n = config.settings()["segments"]["min_cohort_size"]
    customers = _customer_frame(repo)
    frame = _action_frame(repo, customers) if spec.metric in ACTION_METRICS else customers
    if frame.empty:
        return {"value": None, "table": pd.DataFrame(), "suppressed": 0, "n": 0}
    for f in spec.filters:
        frame = frame[frame[f["field"]].astype(str).str.lower() == f["value"].lower()]
    delta = config.settings()["decision"]["risk_rising_delta"]

    def agg(g: pd.DataFrame) -> tuple[float | None, int]:
        n = len(g)
        m = spec.metric
        if m == "customers":
            return float(n), n
        if m == "high_risk_customers":
            return float((g["risk_tier"] == "high").sum()), n
        if m == "avg_risk":
            return float(g["churn_probability"].mean()) if n else None, n
        if m == "revenue_at_risk":
            return float(g["revenue_at_risk"].sum()), n
        if m == "rising_risk_customers":
            return float((g["risk_change"] >= delta).sum()), n
        if m == "observed_churn_rate":
            lab = g["churned"].dropna()
            return (float(lab.mean()) if len(lab) else None), int(len(lab))
        if m == "actions":
            return float(n), n
        if m == "retention_rate":
            final = g[g["outcome_type"].isin(["retained", "churned"])]
            return (float((final["outcome_type"] == "retained").mean()) if len(final) else None), int(len(final))
        raise QueryError(m)

    value, n = agg(frame)
    out: dict[str, Any] = {"value": value, "n": n, "suppressed": 0, "table": pd.DataFrame()}
    if spec.group_by != "none" and spec.group_by in frame:
        rows = []
        for key, g in frame.groupby(spec.group_by, dropna=False):
            v, gn = agg(g)
            rows.append({spec.group_by: str(key), "value": v, "n": gn})
        table = pd.DataFrame(rows)
        small = table["n"] < min_n
        out["suppressed"] = int(small.sum())
        table.loc[small, "value"] = np.nan
        table = table.sort_values("value", ascending=False, na_position="last")
        if spec.top_n:
            table = table.head(spec.top_n)
        out["table"] = table.reset_index(drop=True)
    return out


def ask(repo: Repository, question: str, actor: str) -> dict[str, Any]:
    question = (question or "").strip()[:500]
    parser, error, spec = "keywords", None, None
    try:
        if not question:
            raise QueryError("Ask a question first.")
        spec = keyword_parse(question)
        result = execute(repo, spec)
        status = "ok"
    except QueryError as exc:
        result, status, error = None, "rejected", str(exc)
    repo.execute(
        "INSERT INTO assistant_queries (query_id, org_id, actor, question, interpreted, parser, status, error, "
        "asked_at) VALUES (:i, :org, :a, :q, :s, :p, :st, :e, :t)",
        dict(i=new_id("aq"), a=actor, q=question, s=json.dumps(asdict(spec)) if spec else None, p=parser,
             st=status, e=error, t=now_iso()))
    return {"spec": spec, "parser": parser, "status": status, "error": error, "result": result,
            "interpretation": spec.describe() if spec else None}
