import numpy as np
import pytest

from retentioniq.data.schema import FEATURE_COLUMNS, validate_frame
from retentioniq.decisioning.workflow import create_action
from retentioniq.experiments import stats, uplift
from retentioniq.experiments.registry import Experiments, assign_arm


def test_sample_size_matches_reference():
    # Classic example: 50% -> 55% at alpha 0.05, power 0.8 needs ~1,565 per arm
    assert 1500 <= stats.required_sample_size(0.5, 0.05) <= 1600
    assert stats.unequal_allocation_total(1000, 0.5) == 2000
    assert stats.unequal_allocation_total(1000, 0.1) > 5000


def test_two_proportion_test():
    r = stats.two_proportion_test(600, 1000, 500, 1000)
    assert r["significant"] and r["difference"] == pytest.approx(0.1) and r["ci_low"] > 0
    r = stats.two_proportion_test(51, 100, 50, 100)
    assert not r["significant"]


def test_assignment_deterministic_and_balanced():
    arms = [assign_arm("exp1", f"C{i}", 0.2) for i in range(10000)]
    assert arms == [assign_arm("exp1", f"C{i}", 0.2) for i in range(10000)]
    assert 0.18 < arms.count("control") / len(arms) < 0.22
    other = [assign_arm("exp2", f"C{i}", 0.2) for i in range(10000)]
    assert arms != other  # salted per experiment


def test_scope_rules_and_specific_wins(org):
    ex = Experiments(org)
    g = ex.create(name="global", actor="m", start=True)
    s = ex.create(name="specific", actor="m", start=True, action_code="SERVICE_RECOVERY", control_share=0.5)
    with pytest.raises(ValueError):
        ex.create(name="dup", actor="m", start=True)
    assert ex.running_for("SERVICE_RECOVERY")["experiment_id"] == s
    assert ex.running_for("BILLING_SUPPORT")["experiment_id"] == g


def test_control_customers_are_held_out_and_analysed(service, org, raw_df):
    ex = Experiments(org)
    exp_id = ex.create(name="all", actor="m", start=True, control_share=0.5, min_detectable_effect=0.3)
    sample = raw_df.drop(columns=["Churn"]).sample(200, random_state=1)
    batch = service.score(sample, repo=org, persist=True)
    arms = []
    for r in batch.records:
        if r["recommendation"]["code"] in ("MONITOR",):
            continue
        act = create_action(org, {**r, "data_quality_status": r["data_quality"]["status"]}, "agent")
        arms.append(act["arm"])
        org.record_outcome(act["action_id"], "retained", "agent")
    assert "control" in arms and "treatment" in arms
    held = org.actions()
    assert set(held.loc[held["experiment_arm"] == "control", "status"]) <= {"completed"}
    a = ex.analyze(exp_id)
    assert a["arms"]["control"]["with_outcome"] > 0
    assert a["test"]["difference"] == pytest.approx(0.0)
    assert a["verdict"] in ("no_detectable_effect", "underpowered")


def test_uplift_simulation_recovers_known_effect(raw_df):
    X = validate_frame(raw_df.drop(columns=["Churn"])).valid[FEATURE_COLUMNS]
    base = np.full(len(X), 0.3)
    r = uplift.run_simulation(X, base, population=20000)
    assert r["correlation_with_truth"] > 0.5
    assert abs(r["observed_ate"] - r["true_ate"]) < 0.03
    assert r["auuc_gain"] > 0
    assert set(r["segments"]["segment"]) >= {"Persuadables", "Sleeping dogs"}
