"""CRM-shaped exports of the retention action queue (column mappings are config)."""

from __future__ import annotations

import pandas as pd

TEMPLATES = {
    "generic": {
        "customer_id": "customer_id", "risk": "churn_probability", "tier": "risk_tier",
        "action": "action_name", "action_code": "action_code", "reasons": "top_reasons",
        "requires_approval": "requires_approval", "model_version": "model_version",
    },
    "salesforce_task": {
        "WhatId": "customer_id", "Subject": "action_name", "Priority": "priority_label",
        "Description": "description", "Status": "task_status", "Type": "channel",
    },
    "hubspot_ticket": {
        "customer_id": "customer_id", "subject": "action_name", "hs_ticket_priority": "priority_label",
        "content": "description", "hs_pipeline_stage": "task_status",
    },
}


def export(queue: pd.DataFrame, template: str) -> pd.DataFrame:
    mapping = TEMPLATES[template]
    q = queue.copy()
    q["priority_label"] = q["risk_tier"].map({"high": "HIGH", "medium": "MEDIUM", "low": "LOW"})
    q["description"] = ("Estimated churn risk " + (q["churn_probability"] * 100).round(0).astype(int).astype(str)
                        + "%. Drivers: " + q["top_reasons"].fillna("n/a")
                        + ". Advisory recommendation - verify before contact.")
    q["task_status"] = q["requires_approval"].map({True: "Waiting approval", False: "Not Started"})
    q["channel"] = q.get("channel", "Other")
    return pd.DataFrame({out: q[src] for out, src in mapping.items() if src in q})
