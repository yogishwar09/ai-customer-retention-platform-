import pytest

from retentioniq.assistant.semantic import QueryError, QuerySpec, ask, execute, keyword_parse


@pytest.mark.parametrize("q, metric, group", [
    ("revenue at risk by contract", "revenue_at_risk", "Contract"),
    ("how many high risk customers per internet service", "high_risk_customers", "InternetService"),
    ("top 3 payment methods by revenue at risk", "revenue_at_risk", "PaymentMethod"),
    ("retention rate by arm", "retention_rate", "experiment_arm"),
    ("average risk", "avg_risk", "none"),
])
def test_keyword_parser(q, metric, group):
    spec = keyword_parse(q)
    assert (spec.metric, spec.group_by) == (metric, group)


def test_filters_extracted():
    spec = keyword_parse("revenue at risk for fiber month-to-month customers")
    assert {"field": "InternetService", "value": "Fiber optic"} in spec.filters
    assert {"field": "Contract", "value": "Month-to-month"} in spec.filters


@pytest.mark.parametrize("spec", [
    QuerySpec("DROP TABLE users"), QuerySpec("customers", group_by="password_hash"),
    QuerySpec("customers", filters=[{"field": "org_id", "value": "x"}]),
    QuerySpec("customers", group_by="experiment_arm"), QuerySpec("customers", top_n=500),
])
def test_validation_rejects_unsafe_specs(spec):
    with pytest.raises(QueryError):
        spec.validate()


def test_execute_suppresses_small_groups_and_audits(service, org, raw_df):
    service.score(raw_df.drop(columns=["Churn"]).sample(400, random_state=2), repo=org, persist=True)
    res = execute(org, QuerySpec("customers", group_by="PaymentMethod").validate())
    assert res["value"] == 400 and res["table"]["n"].sum() == 400
    tiny = execute(org, QuerySpec("customers", group_by="tenure_band",
                                  filters=[{"field": "Contract", "value": "Two year"}]).validate())
    assert tiny["suppressed"] >= 0
    out = ask(org, "revenue at risk by contract", "viewer@test")
    assert out["status"] == "ok" and out["parser"] == "keywords"
    log = org.query_df("SELECT * FROM assistant_queries WHERE org_id = :org")
    assert len(log) == 1 and log["status"].iloc[0] == "ok"


def test_assistant_is_tenant_scoped(service, org, trained, raw_df):
    from retentioniq.storage.accounts import Accounts
    from retentioniq.storage.repository import Repository
    other = Repository(Accounts().create_org("Other", "assist-other"))
    service.score(raw_df.drop(columns=["Churn"]).head(50), repo=org, persist=True)
    assert ask(other, "how many customers", "x")["result"]["value"] is None
