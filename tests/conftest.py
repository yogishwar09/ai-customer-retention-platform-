"""Test isolation: artifacts, database, reports and configs go to a temp dir; one
model is trained per session (reduced bootstrap) and the demo is seeded once."""

import os
import shutil
import tempfile
import time
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="retentioniq-tests-"))
ROOT = Path(__file__).resolve().parents[1]
os.environ["RETENTIONIQ_ARTIFACTS_DIR"] = str(_TMP / "artifacts")
os.environ["RETENTIONIQ_DB_PATH"] = str(_TMP / "artifacts" / "test.db")
os.environ["RETENTIONIQ_REPORTS_DIR"] = str(_TMP / "reports")
os.environ["RETENTIONIQ_CONFIG_DIR"] = str(_TMP / "configs")
os.environ["RETENTIONIQ_INPROCESS_WORKER"] = "0"
shutil.copytree(ROOT / "configs", _TMP / "configs")
cfg = (_TMP / "configs" / "settings.yaml").read_text()
(_TMP / "configs" / "settings.yaml").write_text(cfg.replace("bootstrap_resamples: 500", "bootstrap_resamples: 50"))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

DEMO_PASSWORD = "demo123"


@pytest.fixture(scope="session")
def trained():
    from retentioniq.models import registry
    from retentioniq.models.train import train
    out = train(log=lambda *_: None)
    assert registry.production_version() == out["version"]
    return out


@pytest.fixture(scope="session")
def service(trained):
    from retentioniq.service import PredictionService
    return PredictionService(load_challenger=False)


@pytest.fixture(scope="session")
def seeded(trained):
    from scripts.seed_demo import main
    return main(reset=True)


@pytest.fixture(scope="session")
def raw_df():
    from retentioniq.data.loader import load_raw
    return load_raw()


@pytest.fixture
def org(trained):
    """A fresh, empty tenant per test."""
    from retentioniq.storage.accounts import Accounts
    from retentioniq.storage.repository import Repository
    slug = f"t-{time.time_ns()}"
    org_id = Accounts().create_org(f"Test {slug}", slug)
    return Repository(org_id)


@pytest.fixture
def good_row():
    return {
        "customerID": "T-0001", "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 85.0, "TotalCharges": 420.0,
    }


@pytest.fixture
def frame(good_row):
    def make(*overrides):
        rows = []
        for i, o in enumerate(overrides or [{}]):
            r = dict(good_row, customerID=f"T-{i:04d}")
            r.update(o)
            rows.append(r)
        return pd.DataFrame(rows)
    return make


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TMP, ignore_errors=True)
