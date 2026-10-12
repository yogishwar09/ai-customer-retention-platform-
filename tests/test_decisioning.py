import copy

import pytest

from retentioniq import config
from retentioniq.decisioning import policy
from retentioniq.decisioning.risk import risk_tier, tier_cutoffs


def test_tier_boundaries_follow_operating_threshold():
    cut = tier_cutoffs(0.3)
    assert cut == {"high": 0.6, "medium": 0.3}
    assert risk_tier(0.29, 0.3) == "low"
    assert risk_tier(0.30, 0.3) == "medium"
    assert risk_tier(0.60, 0.3) == "high"


def test_auto_medium_requires_threshold():
    with pytest.raises(ValueError):
        tier_cutoffs(None)


def test_low_risk_gets_monitor(good_row):
    rec = policy.recommend(good_row, 0.1, "low", customer_id="x")
    assert rec.code == "MONITOR"


def test_data_quality_warning_routes_to_review(good_row):
    rec = policy.recommend(good_row, 0.8, "high", quality_status="warning")
    assert rec.code == "DATA_REVIEW"


def test_borderline_routes_to_review(good_row):
    rec = policy.recommend(good_row, 0.31, "medium", borderline=True)
    assert rec.code == "DATA_REVIEW"
    assert rec.eligible_alternatives  # still shows what would be eligible


def test_highest_expected_value_wins_and_ev_formula(good_row):
    rec = policy.recommend(good_row, 0.7, "high")
    a = rec.assumptions
    expected = a["churn_probability"] * a["annual_value"] * a["assumed_effectiveness"] - a["action_cost"]
    assert rec.expected_value == pytest.approx(expected, abs=0.01)
    assert all(rec.expected_value >= alt["expected_value"] for alt in rec.eligible_alternatives)


def test_renewal_requires_approval(good_row):
    row = dict(good_row, tenure=8, PaymentMethod="Mailed check", OnlineSecurity="Yes", OnlineBackup="Yes",
               TechSupport="Yes", MonthlyCharges=105.0)
    rec = policy.recommend(row, 0.75, "high")
    assert rec.code == "RENEWAL_CONVERSATION"
    assert rec.requires_approval


def test_two_year_customer_not_offered_renewal(good_row):
    row = dict(good_row, Contract="Two year", tenure=40)
    rec = policy.recommend(row, 0.7, "high")
    assert rec.code != "RENEWAL_CONVERSATION"


def test_playbooks_validate():
    policy.validate_playbooks(config.playbooks())


def test_playbook_with_unknown_feature_is_rejected():
    pb = copy.deepcopy(config.playbooks())
    pb["actions"][1]["all_of"].append({"feature": "CreditScore", "op": "lt", "value": 500})
    with pytest.raises(policy.PolicyError, match="unknown feature"):
        policy.validate_playbooks(pb)
