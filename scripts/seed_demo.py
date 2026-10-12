"""Seed a multi-tenant demo.

Creates two organisations so tenant isolation is visible:

* **Demo Telco** - the 1,409 *held-out test* customers of the IBM Telco sample, each
  with two score snapshots (one ~5 weeks ago from features rolled back a month, one in
  the last two weeks) so trends and rising-risk alerts have history; real churn labels
  for ~60% as delayed labels; two running experiments; advisory actions for high-risk
  customers; and outcomes for half of those actions taken from the customers' *real
  historical* churn labels. Nobody was actually contacted, so experiments correctly
  show no detectable treatment effect - the platform never manufactures impact.
* **Northwind Mobile** - 300 synthetic customers, to demonstrate isolation.

Simple local demo logins (shown on the sign-in page): admin / manager / agent / viewer
(@demo.com) and northwind (@northwind.com), all with password ``demo123``. Set
RETENTIONIQ_DEMO_PASSWORD to change it. Refuses to run when RETENTIONIQ_ENV=production -
these accounts are for local demos only.

    python scripts/seed_demo.py [--reset]
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from retentioniq import config
from retentioniq.data.loader import load_raw
from retentioniq.decisioning.workflow import create_action
from retentioniq.experiments.registry import Experiments
from retentioniq.integrations.webhooks import Webhooks, dispatch_due
from retentioniq.models import registry
from retentioniq.models.train import ensure_model
from retentioniq.service import PredictionService
from retentioniq.storage.accounts import Accounts
from retentioniq.storage.engine import get_engine
from retentioniq.storage.repository import Repository

SEED = "demo-seed"
CREDENTIALS_FILE = config.ARTIFACTS_DIR / "demo_credentials.txt"
ORGS = [
    {"name": "Demo Telco", "slug": "demo-telco", "domain": "demo.com",
     "roles": ["admin", "manager", "agent", "viewer"]},
    {"name": "Northwind Mobile", "slug": "northwind", "domain": "northwind.com", "roles": ["northwind"],
     "role_of": {"northwind": "admin"}},
]


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _rolled_back(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Features as they plausibly looked ~1 month earlier (demo history only)."""
    prev = df.copy()
    prev["tenure"] = (prev["tenure"].astype(int) - 1).clip(lower=0)
    monthly = prev["MonthlyCharges"].astype(float)
    # A subset recently had a price increase or a contract downgrade -> their risk rose.
    price_up = rng.random(len(prev)) < 0.10
    prev.loc[price_up, "MonthlyCharges"] = (monthly[price_up] - rng.uniform(10, 25, price_up.sum())).clip(18.25)
    downgrade = (rng.random(len(prev)) < 0.08) & (prev["Contract"] == "Month-to-month") & (prev["tenure"] > 6)
    prev.loc[downgrade, "Contract"] = "One year"
    total = pd.to_numeric(prev["TotalCharges"], errors="coerce")
    prev["TotalCharges"] = (total - monthly).clip(lower=0).round(2)
    prev.loc[prev["tenure"] == 0, "TotalCharges"] = np.nan
    return prev


def _score_with_dates(svc: PredictionService, repo: Repository, frame: pd.DataFrame, source: str,
                      start_days_ago: float, span_days: float, rng: np.random.Generator) -> list[dict]:
    batch = svc.score(frame, source=source)
    now = datetime.now(timezone.utc)
    for rec in batch.records:
        rec["scored_at"] = _iso(now - timedelta(days=start_days_ago - rng.uniform(0, span_days)))
    svc.persist(batch.records, repo)
    return batch.records


DEMO_PASSWORD = "demo123"


def _credentials(password: str | None) -> str:
    return password or os.environ.get("RETENTIONIQ_DEMO_PASSWORD") or DEMO_PASSWORD


def main(reset: bool = False) -> dict:
    if os.environ.get("RETENTIONIQ_ENV") == "production":
        raise SystemExit("Refusing to seed demo users in production (RETENTIONIQ_ENV=production).")
    ensure_model()
    engine = get_engine()
    accounts = Accounts(engine)
    existing = {o["slug"]: o for o in accounts.list_orgs()}
    if existing and not reset:
        print("Demo already seeded; use --reset to reseed.")
        return {"orgs": existing}

    rng = np.random.default_rng(42)
    password = _credentials(None)
    lines = ["RetentionIQ demo credentials (local demo only - do not reuse elsewhere)", ""]
    org_ids = {}
    for spec in ORGS:
        org = existing.get(spec["slug"])
        if org:
            Repository(org["org_id"], engine).reset()
            org_id = org["org_id"]
        else:
            org_id = accounts.create_org(spec["name"], spec["slug"])
        org_ids[spec["slug"]] = org_id
        users = {u["email"]: u for u in accounts.list_users(org_id)}
        for name in spec["roles"]:
            role = spec.get("role_of", {}).get(name, name)
            email = f"{name}@{spec['domain']}"
            if email in users:
                accounts.update_user(org_id, users[email]["user_id"], role=role, active=True, password=password,
                                     enforce_strength=False)
            else:
                accounts.create_user(org_id, email, password, role, display_name=f"{spec['name']} {role.title()}",
                                     enforce_strength=False)
            lines.append(f"{spec['name']:<18} {role:<8} {email}")
        for key in accounts.list_api_keys(org_id):
            if not key["revoked_at"]:
                accounts.revoke_api_key(org_id, key["key_id"])
    lines += ["", f"Password for every demo user: {password}", ""]

    model = registry.load()
    svc = PredictionService(model=model, load_challenger=False)

    # ------------------------------------------------------------ Demo Telco
    telco = Repository(org_ids["demo-telco"], engine)
    hooks = Webhooks(telco)
    hooks.create("Sandbox CRM", "sandbox://crm", ["customer.high_risk", "action.approved",
                                                  "action.outcome_recorded", "alert.raised", "batch.completed"],
                 SEED)
    exps = Experiments(telco)
    exps.create(name="Global retention holdout", actor=SEED, start=True, control_share=0.10,
                min_detectable_effect=0.05,
                hypothesis="Advisory retention outreach improves 90-day retention versus no contact.")
    exps.create(name="Proactive service check pilot", actor=SEED, start=True, action_code="SERVICE_RECOVERY",
                control_share=0.5, min_detectable_effect=0.08,
                hypothesis="Fiber customers without tech support retain better after a proactive service check.")

    test_ids = set(model.evaluation["test_predictions"]["customer_id"])
    raw = load_raw()
    portfolio = raw[raw["customerID"].isin(test_ids)].reset_index(drop=True)
    labels = portfolio[["customerID", "Churn"]].copy()
    current = portfolio.drop(columns=["Churn"])
    _score_with_dates(svc, telco, _rolled_back(current, rng), "demo_seed_history", 40, 7, rng)
    records = _score_with_dates(svc, telco, current, "demo_seed", 14, 14, rng)

    labelled = labels.sample(frac=0.6, random_state=42)
    telco.upsert_labels([(r.customerID, int(r.Churn == "Yes")) for r in labelled.itertuples()],
                        source="demo_seed_holdout_labels", actor=SEED)

    truth = dict(zip(labels["customerID"], labels["Churn"], strict=True))
    high = sorted((r for r in records if r["risk_tier"] == "high"),
                  key=lambda r: -r["recommendation"]["expected_value"])
    created = 0
    for i, rec in enumerate(high[:160]):
        pred = {**rec, "data_quality_status": rec["data_quality"]["status"]}
        if rec["recommendation"]["code"] == "MONITOR":
            continue
        act = create_action(telco, pred, SEED, note="Auto-created from demo seed (advisory)")
        created += 1
        if i % 2 == 0:  # outcome from the customer's real historical label, never invented
            telco.record_outcome(act["action_id"], "churned" if truth[rec["customer_id"]] == "Yes" else "retained",
                                 SEED, window_days=90, label_source="demo_seed_historical_label")
        elif act["arm"] != "control" and i % 5 == 1:
            telco.update_action_status(act["action_id"], "approved" if rec["recommendation"]["requires_approval"]
                                       else "in_progress", "manager@demo.com")
    telco.audit(SEED, "demo.seeded", "organization", org_ids["demo-telco"],
                {"customers": len(records), "actions": created, "model_version": model.version})

    _, token = accounts.create_api_key(org_ids["demo-telco"], "demo-integration", "manager", SEED)
    lines.append(f"Demo Telco API key (manager role): {token}")

    # -------------------------------------------------------------- Northwind
    north = Repository(org_ids["northwind"], engine)
    sample = config.SAMPLES_DIR / "batch_clean.csv"
    if sample.exists():
        nw = pd.read_csv(sample)
        nw["customerID"] = nw["customerID"].str.replace("SYN-", "NW-", regex=False)
        _score_with_dates(svc, north, nw, "demo_seed", 10, 10, rng)
    north.audit(SEED, "demo.seeded", "organization", org_ids["northwind"], {"customers": 300})

    dispatch_due(engine)
    with engine.begin() as c:  # restart schedules so the seeded history isn't rescored immediately
        from sqlalchemy import text
        c.execute(text("DELETE FROM schedules"))
        c.execute(text("DELETE FROM jobs WHERE status IN ('queued', 'succeeded') AND created_by = 'scheduler'"))
    inbox = config.PROJECT_ROOT / config.settings()["jobs"]["inbox_dir"]
    for spec in ORGS:
        (inbox / spec["slug"]).mkdir(parents=True, exist_ok=True)

    CREDENTIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CREDENTIALS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        CREDENTIALS_FILE.chmod(0o600)
    except OSError:
        pass
    print(f"Seeded Demo Telco ({len(records):,} customers, {created} actions) and Northwind Mobile. "
          f"Sign-in details: {CREDENTIALS_FILE}")
    return {"orgs": org_ids, "password": password, "api_key": token}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    main(parser.parse_args().reset)
