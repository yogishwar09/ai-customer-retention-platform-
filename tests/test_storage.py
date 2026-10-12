import json

import pytest

from retentioniq.decisioning.workflow import create_action
from retentioniq.monitoring.summary import impact_by_arm, monitoring_summary
from retentioniq.storage.accounts import Accounts
from retentioniq.storage.repository import Repository


def _persist(service, repo, row):
    batch = service.score(row, repo=repo, persist=True)
    return {**batch.records[0], "data_quality_status": batch.records[0]["data_quality"]["status"]}


def test_prediction_roundtrip(service, org, good_row):
    rec = _persist(service, org, good_row)
    stored = org.get_prediction(rec["prediction_id"])
    assert stored["customer_id"] == "T-0001"
    assert stored["features"]["Contract"] == "Month-to-month"
    assert len(stored["explanations"]) == len(rec["explanations"])
    latest = org.latest_scores()
    assert len(latest) == 1 and "top_reasons" in latest and latest["previous_probability"].isna().all()


def test_tenant_isolation(service, trained, good_row):
    acc = Accounts()
    a = Repository(acc.create_org("Tenant A", "iso-a"))
    b = Repository(acc.create_org("Tenant B", "iso-b"))
    rec = _persist(service, a, good_row)
    a.upsert_labels([("T-0001", 1)], "test", "t")
    assert b.get_prediction(rec["prediction_id"]) is None
    assert b.latest_scores().empty and b.labels().empty and b.audit_log().empty is not None
    assert b.customer_history("T-0001").empty
    with pytest.raises(KeyError):
        b.update_action_status("act_x", "approved", "x")
    act = create_action(a, rec, "agent@a")
    assert b.get_action(act["action_id"]) is None
    with pytest.raises(KeyError):
        b.record_outcome(act["action_id"], "retained", "x")


def test_latest_scores_tracks_trend(service, org, good_row):
    _persist(service, org, dict(good_row, Contract="Two year", tenure=60, TotalCharges=5000))
    _persist(service, org, good_row)
    latest = org.latest_scores()
    assert len(latest) == 1
    assert latest.iloc[0]["Contract"] == "Month-to-month"
    assert latest.iloc[0]["risk_change"] > 0.15


def test_action_lifecycle_audit_and_feedback_label(service, org, good_row):
    rec = _persist(service, org, good_row)
    act = create_action(org, rec, "agent@test")
    org.update_action_status(act["action_id"], "approved", "manager@test")
    org.record_outcome(act["action_id"], "churned", "agent@test", window_days=30)
    assert org.get_action(act["action_id"])["status"] == "completed"
    events = set(org.audit_log()["event_type"])
    assert {"action.created", "action.approved", "action.outcome_recorded"} <= events
    labels = org.labels()
    assert labels.set_index("customer_id").loc["T-0001", "churned"] == 1  # outcome feeds the label store


def test_outcome_is_idempotent_per_action(service, org, good_row):
    rec = _persist(service, org, good_row)
    act = create_action(org, rec, "a")
    org.record_outcome(act["action_id"], "no_response", "a")
    org.record_outcome(act["action_id"], "retained", "a")
    assert org.actions().iloc[0]["outcome_type"] == "retained"


def test_invalid_status_and_unknown_action(org):
    with pytest.raises(ValueError):
        org.update_action_status("x", "teleported", "a")
    with pytest.raises(KeyError):
        org.update_action_status("missing", "approved", "a")
    with pytest.raises(KeyError):
        org.record_outcome("missing", "retained", "a")


def test_outbox_only_for_subscribed_events(service, org, good_row):
    from retentioniq.integrations.webhooks import Webhooks
    assert org.emit("action.approved", {"x": 1}) is None  # no webhooks -> no event
    Webhooks(org).create("crm", "sandbox://crm", ["customer.high_risk"], "admin")
    assert org.emit("action.approved", {"x": 1}) is None
    service.score(good_row, repo=org, persist=True)  # high-risk profile -> event
    df = org.query_df("SELECT e.event_type, e.payload FROM outbox_events e WHERE e.org_id = :org")
    assert list(df["event_type"]) == ["customer.high_risk"]
    payload = json.loads(df["payload"].iloc[0])
    assert payload["data"]["customer_id"] == "T-0001" and payload["org_id"] == org.org_id


def test_batch_job_filename_is_sanitised(org):
    job = org.start_batch_job("../../etc/passwd;rm -rf.csv", "a", "v1")
    name = org.batch_jobs().set_index("job_id").loc[job, "filename_safe"]
    assert "/" not in name and ";" not in name


def test_impact_needs_enough_outcomes():
    import pandas as pd
    small = pd.DataFrame({"experiment_arm": ["treatment", "control"], "outcome_type": ["retained", "churned"]})
    assert impact_by_arm(small)["status"] == "insufficient"


def test_monitoring_summary_with_labels(service, org, raw_df):
    sample = raw_df.sample(300, random_state=5)
    service.score(sample.drop(columns=["Churn"]), repo=org, persist=True)
    org.upsert_labels([(r.customerID, int(r.Churn == "Yes")) for r in sample.itertuples()], "test", "t")
    s = monitoring_summary(org, service)
    assert s["delayed_label_performance"]["status"] == "ok"
    assert s["delayed_label_performance"]["roc_auc"] > 0.75
    assert s["drift_status"] == "ok"
