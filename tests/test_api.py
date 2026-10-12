import pytest
from fastapi.testclient import TestClient

from api.main import app
from retentioniq.storage.accounts import Accounts


@pytest.fixture(scope="module")
def client(trained):
    return TestClient(app)


@pytest.fixture
def keys(org):
    acc = Accounts()
    return {role: {"Authorization": f"Bearer {acc.create_api_key(org.org_id, role, role, 'test')[1]}"}
            for role in ("viewer", "agent", "manager", "admin")} | {"org": org}


def test_health_and_metrics_open(client):
    assert client.get("/health").json()["model_loaded"] is True
    m = client.get("/metrics")
    assert m.status_code == 200 and "retentioniq_http_requests_total" in m.text


def test_auth_required(client, good_row):
    assert client.get("/v1/model-info").status_code == 401
    assert client.post("/v1/predictions", json=good_row).status_code == 401
    assert client.get("/v1/model-info", headers={"Authorization": "Bearer riq_bad_key"}).status_code == 401


def test_rbac_enforced(client, keys, good_row):
    assert client.get("/v1/model-info", headers=keys["viewer"]).status_code == 200
    assert client.post("/v1/predictions", json=good_row, headers=keys["viewer"]).status_code == 403
    assert client.post("/v1/predictions", json=good_row, headers=keys["agent"]).status_code == 200
    assert client.get("/v1/webhooks", headers=keys["manager"]).status_code == 403
    assert client.get("/v1/webhooks", headers=keys["admin"]).status_code == 200
    assert client.post("/v1/models/retrain", json={}, headers=keys["manager"]).status_code == 403


def test_prediction_contract_and_headers(client, keys, good_row):
    r = client.post("/v1/predictions", json=good_row, headers=keys["agent"])
    body = r.json()
    assert body["model_version"] and body["scored_at"] and body["data_quality"]["status"] == "passed"
    assert all(h in r.headers for h in ("X-Request-ID", "X-Content-Type-Options", "X-Frame-Options"))


@pytest.mark.parametrize("mutate", [
    lambda r: r.pop("Contract"), lambda r: r.update(tenure="ten"), lambda r: r.update(Contract="Weekly"),
    lambda r: r.update(extra_field=1), lambda r: r.update(MonthlyCharges=-5),
])
def test_invalid_payloads_return_422(client, keys, good_row, mutate):
    mutate(good_row)
    assert client.post("/v1/predictions", json=good_row, headers=keys["agent"]).status_code == 422


def test_cross_field_rule_and_empty(client, keys, good_row):
    good_row["InternetService"] = "No"
    r = client.post("/v1/predictions", json=good_row, headers=keys["agent"])
    assert r.status_code == 422 and "No internet service" in r.json()["detail"]
    assert client.post("/v1/predictions/batch", json={"customers": []}, headers=keys["agent"]).status_code == 422


def test_tenant_isolation_over_api(client, keys, good_row, trained):
    pid = client.post("/v1/predictions?persist=true", json=good_row, headers=keys["agent"]).json()["prediction_id"]
    acc = Accounts()
    other = acc.create_org("API other", "api-other")
    other_key = {"Authorization": f"Bearer {acc.create_api_key(other, 'x', 'admin', 't')[1]}"}
    assert client.get(f"/v1/predictions/{pid}", headers=keys["viewer"]).status_code == 200
    assert client.get(f"/v1/predictions/{pid}", headers=other_key).status_code == 404
    assert client.get("/v1/customers/T-0001/retention-summary", headers=other_key).status_code == 404


def test_action_flow_with_approval_rules(client, keys, good_row):
    good_row.update(customerID="FLOW-1", tenure=8, Partner="No", PaymentMethod="Mailed check", OnlineSecurity="Yes",
                    OnlineBackup="Yes", TechSupport="Yes", StreamingMovies="Yes", MonthlyCharges=105.0,
                    TotalCharges=840.0)
    pid = client.post("/v1/predictions?persist=true", json=good_row, headers=keys["agent"]).json()["prediction_id"]
    act = client.post("/v1/actions/recommend", json={"prediction_id": pid}, headers=keys["agent"]).json()
    assert act["recommendation"]["requires_approval"]
    aid = act["action_id"]
    assert client.patch(f"/v1/actions/{aid}", json={"status": "approved"}, headers=keys["agent"]).status_code == 403
    assert client.patch(f"/v1/actions/{aid}", json={"status": "approved"}, headers=keys["manager"]).status_code == 200
    r = client.post(f"/v1/actions/{aid}/outcome", json={"outcome_type": "retained"}, headers=keys["agent"])
    assert r.status_code == 201
    summary = client.get("/v1/customers/FLOW-1/retention-summary", headers=keys["viewer"]).json()
    assert summary["actions"][0]["outcome_type"] == "retained"


def test_async_job_endpoints(client, keys):
    from retentioniq import config
    from retentioniq.jobs.queue import JobQueue
    from retentioniq.jobs.worker import run_job
    data = (config.SAMPLES_DIR / "template.csv").read_bytes()
    j = client.post("/v1/jobs/batch-scoring", files={"file": ("t.csv", data, "text/csv")}, headers=keys["agent"])
    assert j.status_code == 202
    jid = j.json()["job_id"]
    q = JobQueue()
    while (job := q.claim(job_types=["batch_score"])) and job["job_id"] != jid:
        q.complete(job["job_id"])
    run_job(q, job)
    status = client.get(f"/v1/jobs/{jid}", headers=keys["viewer"]).json()
    assert status["status"] == "succeeded" and status["result"]["rows_accepted"] == 3
    assert client.get(f"/v1/jobs/{jid}/results", headers=keys["manager"]).status_code == 200
    bad = client.post("/v1/jobs/batch-scoring", files={"file": ("t.exe", b"x", "application/x")}, headers=keys["agent"])
    assert bad.status_code == 415


def test_webhooks_experiments_assistant(client, keys):
    w = client.post("/v1/webhooks", json={"name": "crm", "url": "sandbox://crm", "event_types": ["action.approved"]},
                    headers=keys["admin"])
    assert w.status_code == 201 and w.json()["signing_secret"].startswith("whsec_")
    bad = client.post("/v1/webhooks", json={"name": "x", "url": "https://127.0.0.1/x", "event_types": ["action.approved"]},
                      headers=keys["admin"])
    assert bad.status_code == 422
    e = client.post("/v1/experiments", json={"name": "API test", "start": True}, headers=keys["manager"])
    assert e.status_code == 201
    a = client.get(f"/v1/experiments/{e.json()['experiment_id']}/analysis", headers=keys["viewer"]).json()
    assert a["verdict"] == "collecting"
    q = client.post("/v1/assistant/query", json={"question": "how many customers by contract"}, headers=keys["viewer"])
    assert q.status_code == 200 and "interpretation" in q.json()


def test_rate_limit(client, keys, monkeypatch):
    from retentioniq import config
    real = config.settings()
    monkeypatch.setattr(config, "settings", lambda: {**real, "api": {**real["api"], "rate_limit_per_minute": 3}})
    codes = [client.get("/v1/model-info", headers=keys["viewer"]).status_code for _ in range(6)]
    assert 429 in codes and codes[0] == 200
