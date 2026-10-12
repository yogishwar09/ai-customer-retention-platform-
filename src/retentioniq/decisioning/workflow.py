"""Action workflow: turn a stored prediction into a governed retention action,
applying experiment assignment (control customers are held out, not contacted)."""

from __future__ import annotations

from typing import Any

from retentioniq.decisioning import policy
from retentioniq.experiments.registry import Experiments
from retentioniq.storage.repository import Repository


def recommendation_for(prediction: dict[str, Any]) -> policy.Recommendation:
    return policy.recommend(prediction["features"], prediction["churn_probability"], prediction["risk_tier"],
                            quality_status=prediction.get("data_quality_status", "passed"),
                            borderline=bool(prediction.get("borderline")),
                            customer_id=prediction.get("customer_id"))


def create_action(repo: Repository, prediction: dict[str, Any], actor: str, note: str = "",
                  recommendation: dict[str, Any] | None = None) -> dict[str, Any]:
    rec = recommendation or recommendation_for(prediction).as_dict()
    if rec["code"] == "MONITOR":
        raise ValueError("No action to log: the policy recommends monitoring only")
    experiment_id, arm = Experiments(repo).assignment(rec["code"], prediction.get("customer_id"))
    action_id = repo.create_action(prediction_id=prediction["prediction_id"],
                                   customer_id=prediction.get("customer_id"), recommendation=rec, actor=actor,
                                   note=note, experiment_id=experiment_id, arm=arm)
    return {"action_id": action_id, "experiment_id": experiment_id, "arm": arm, "recommendation": rec}
