"""Risk tiers and value estimates. Thresholds come from settings.yaml."""

from __future__ import annotations

import math

from retentioniq import config

TIER_ORDER = {"low": 0, "medium": 1, "high": 2}


def tier_cutoffs(operating_threshold: float | None = None) -> dict[str, float]:
    t = config.settings()["risk_tiers"]
    medium = t["medium"]
    if medium == "auto":
        if operating_threshold is None:
            raise ValueError("risk_tiers.medium is 'auto' but no operating threshold was given")
        medium = operating_threshold
    return {"high": float(t["high"]), "medium": min(float(medium), float(t["high"]))}


def risk_tier(probability: float, operating_threshold: float | None = None) -> str:
    cut = tier_cutoffs(operating_threshold)
    if probability >= cut["high"]:
        return "high"
    if probability >= cut["medium"]:
        return "medium"
    return "low"


def annual_value(monthly_charges: float | None) -> float:
    """Customer value proxy used for prioritisation (documented in settings.yaml)."""
    if monthly_charges is None or (isinstance(monthly_charges, float) and math.isnan(monthly_charges)):
        return 0.0
    return float(monthly_charges) * config.settings()["value"]["horizon_months"]


def revenue_at_risk(probability: float, monthly_charges: float | None) -> float:
    return probability * annual_value(monthly_charges)
