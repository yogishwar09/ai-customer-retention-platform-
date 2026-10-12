"""Policy-driven next-best-action recommendations.

expected value ≈ p(churn) × annual value × assumed effectiveness − action cost

The effectiveness term is an *assumption* from configs/playbooks.yaml until it
is estimated from controlled experiments. Recommendations are advisory; actions
flagged ``requires_approval`` need a manager before any customer contact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retentioniq import config
from retentioniq.data.schema import FEATURE_COLUMNS
from retentioniq.decisioning.risk import TIER_ORDER, annual_value

_OPS = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "in": lambda a, b: a in b,
    "lt": lambda a, b: a is not None and a < b,
    "lte": lambda a, b: a is not None and a <= b,
    "gt": lambda a, b: a is not None and a > b,
    "gte": lambda a, b: a is not None and a >= b,
}


class PolicyError(ValueError):
    pass


@dataclass
class Recommendation:
    code: str
    name: str
    description: str
    channel: str
    requires_approval: bool
    expected_value: float
    assumptions: dict[str, Any]
    eligible_alternatives: list[dict[str, Any]] = field(default_factory=list)
    policy_version: str = ""
    rationale: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code, "name": self.name, "description": self.description,
            "channel": self.channel, "requires_approval": self.requires_approval,
            "expected_value": round(self.expected_value, 2), "assumptions": self.assumptions,
            "eligible_alternatives": self.eligible_alternatives,
            "policy_version": self.policy_version, "rationale": self.rationale,
        }


def validate_playbooks(pb: dict[str, Any]) -> None:
    """Rules may only reference features that exist in the data contract."""
    codes = set()
    for action in pb.get("actions", []):
        if action["code"] in codes:
            raise PolicyError(f"Duplicate action code {action['code']}")
        codes.add(action["code"])
        for cond in action.get("all_of", []) + action.get("any_of", []):
            if cond["feature"] not in FEATURE_COLUMNS:
                raise PolicyError(f"{action['code']} references unknown feature {cond['feature']}")
            if cond["op"] not in _OPS:
                raise PolicyError(f"{action['code']} uses unknown operator {cond['op']}")
        if action.get("min_risk_tier") and action["min_risk_tier"] not in TIER_ORDER:
            raise PolicyError(f"{action['code']} has invalid min_risk_tier")
    if not any(a.get("special") == "fallback" for a in pb.get("actions", [])):
        raise PolicyError("Playbooks need a fallback action")


def _matches(cond: dict[str, Any], features: dict[str, Any]) -> bool:
    return _OPS[cond["op"]](features.get(cond["feature"]), cond["value"])


def recommend(features: dict[str, Any], probability: float, tier: str, *,
              quality_status: str = "passed", borderline: bool = False,
              customer_id: str | None = None) -> Recommendation:
    pb = config.playbooks()
    actions = pb["actions"]
    by_special = {a.get("special"): a for a in actions if a.get("special")}
    value = annual_value(features.get(config.settings()["value"]["field"]))
    version = pb.get("policy_version", "unversioned")

    def build(action: dict[str, Any], ev: float, rationale: str,
              alternatives: list[dict[str, Any]] | None = None) -> Recommendation:
        return Recommendation(
            code=action["code"], name=action["name"], description=action["description"].strip(),
            channel=action["channel"], requires_approval=bool(action.get("requires_approval")),
            expected_value=ev,
            assumptions={"churn_probability": round(probability, 4), "annual_value": round(value, 2),
                         "assumed_effectiveness": action.get("assumed_effectiveness", 0.0),
                         "action_cost": action.get("cost", 0)},
            eligible_alternatives=alternatives or [], policy_version=version, rationale=rationale)

    if quality_status != "passed" and tier != "low" and "data_quality" in by_special:
        return build(by_special["data_quality"], -by_special["data_quality"]["cost"],
                     "Input has data-quality warnings; verify before contacting the customer.")

    eligible = []
    for action in actions:
        if action.get("special"):
            continue
        min_tier = action.get("min_risk_tier", "low")
        if TIER_ORDER[tier] < TIER_ORDER[min_tier]:
            continue
        if not all(_matches(c, features) for c in action.get("all_of", [])):
            continue
        if action.get("any_of") and not any(_matches(c, features) for c in action["any_of"]):
            continue
        ev = probability * value * action.get("assumed_effectiveness", 0.0) - action.get("cost", 0)
        eligible.append((ev, action))

    eligible.sort(key=lambda t: t[0], reverse=True)
    alternatives = [{"code": a["code"], "name": a["name"], "expected_value": round(ev, 2)}
                    for ev, a in eligible]

    if borderline and tier != "low" and "data_quality" in by_special:
        rec = build(by_special["data_quality"], -by_special["data_quality"]["cost"],
                    "Score is within the borderline margin of the operating threshold; "
                    "a human should review before outreach.", alternatives)
        return rec

    if eligible and eligible[0][0] > 0:
        ev, action = eligible[0]
        return build(action, ev, f"Highest expected value among {len(eligible)} eligible action(s).",
                     alternatives[1:])
    fallback = by_special["fallback"]
    reason = ("Risk tier is low." if tier == "low"
              else "No eligible action has positive expected value under current assumptions.")
    return build(fallback, 0.0, reason, alternatives)
