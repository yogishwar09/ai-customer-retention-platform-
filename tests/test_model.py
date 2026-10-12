import json

import numpy as np
import pytest

from retentioniq import config
from retentioniq.data.schema import FEATURE_COLUMNS
from retentioniq.models import registry


def test_training_is_registered_and_passes_gate(trained):
    reg = registry.read_registry()
    assert any(m["version"] == trained["version"] for m in reg["models"])
    assert trained["metadata"]["release_gate"]["passed"]


def test_model_regression_floor(trained):
    """Flag material, unexplained performance drops on the fixed held-out split."""
    t = trained["evaluation"]["test"]
    assert t["roc_auc"] >= 0.82
    assert t["pr_auc"] >= 0.60
    assert t["brier"] <= 0.15


def test_metadata_is_complete(trained):
    m = trained["metadata"]
    for key in ("model_version", "schema_version", "training_data_fingerprint", "artifact_sha256",
                "package_versions", "split", "label_definition", "operating_threshold"):
        assert key in m
    assert m["feature_columns"] == FEATURE_COLUMNS


def test_reports_written(trained):
    assert (config.MODELS_DIR / trained["version"] / "model_card.md").exists()
    assert (config.REPORTS_DIR / "model_card.md").exists()
    assert "Held-out test performance" in (config.REPORTS_DIR / "evaluation_report.md").read_text()


def test_tampered_artifact_is_refused(trained, tmp_path):
    folder = config.MODELS_DIR / trained["version"]
    meta_path = folder / "metadata.json"
    original = meta_path.read_text()
    meta = json.loads(original)
    meta["artifact_sha256"] = "0" * 64
    meta_path.write_text(json.dumps(meta))
    try:
        with pytest.raises(registry.ArtifactIntegrityError):
            registry.load(trained["version"])
    finally:
        meta_path.write_text(original)


def test_service_matches_raw_pipeline(service, raw_df):
    """Train/inference consistency: the service and the persisted pipeline agree."""
    sample = raw_df.drop(columns=["Churn"]).sample(50, random_state=3)
    batch = service.score(sample)
    from retentioniq.data.schema import validate_frame
    X = validate_frame(sample).valid[FEATURE_COLUMNS]
    direct = service.model.pipeline.predict_proba(X)[:, 1]
    np.testing.assert_allclose([r["churn_probability"] for r in batch.records], direct, rtol=1e-6)


def test_scoring_is_deterministic(service, good_row):
    a = service.score(good_row).records[0]
    b = service.score(good_row).records[0]
    assert a["churn_probability"] == b["churn_probability"]
    assert a["explanations"] == b["explanations"]


def test_prediction_contract(service, good_row):
    r = service.score(good_row).records[0]
    assert 0 <= r["churn_probability"] <= 1
    assert r["risk_tier"] in {"low", "medium", "high"}
    assert r["model_version"] == service.model.version
    assert r["data_quality"]["status"] == "passed"
    assert r["explanations"] and {"feature", "reason", "direction"} <= set(r["explanations"][0])


def test_xgboost_contributions_are_additive(service, raw_df):
    if service.model.algorithm != "xgboost":
        pytest.skip("additivity check is for TreeSHAP log-odds")
    X = raw_df.drop(columns=["Churn"]).head(20)
    from retentioniq.data.schema import validate_frame
    X = validate_frame(X).valid[FEATURE_COLUMNS]
    contrib = service.explainer.contributions(X).sum(axis=1).to_numpy()
    p = service.model.pipeline.predict_proba(X)[:, 1]
    logit = np.log(p / (1 - p))
    # logit = bias + sum(contributions); the bias is constant across rows
    np.testing.assert_allclose(logit - contrib, (logit - contrib)[0], atol=1e-4)


def test_what_if_returns_one_row_per_value(service, good_row):
    curve = service.what_if(good_row, "Contract", ["Month-to-month", "One year", "Two year"])
    assert len(curve) == 3
    assert curve["churn_probability"].iloc[0] > curve["churn_probability"].iloc[2]


def test_governed_lifecycle_and_shadow_scoring(trained, org, good_row):
    from retentioniq.models.train import train
    from retentioniq.service import PredictionService
    second = train(log=lambda *_: None)
    assert second["status"] == "candidate"  # never auto-promoted once production exists
    v2 = second["version"]
    with pytest.raises(registry.PromotionError, match="approval reason"):
        registry.set_status(v2, "production", "admin", "")
    registry.set_status(v2, "staging", "admin", "shadow test")
    assert registry.staging_version() == v2
    svc = PredictionService()
    assert svc.challenger is not None and svc.challenger.version == v2
    svc.score(good_row, repo=org, persist=True)
    shadow = org.shadow_comparison()
    assert len(shadow) == 1 and shadow["challenger_version"].iloc[0] == v2
    with pytest.raises(registry.PromotionError):
        registry.set_status(trained["version"], "archived", "admin")  # can't demote production directly
    registry.set_status(v2, "production", "admin", "matched champion in shadow")
    assert registry.production_version() == v2
    registry.set_status(trained["version"], "production", "admin", "rollback drill")  # rollback path
    assert registry.production_version() == trained["version"]
    history = next(m for m in registry.read_registry()["models"] if m["version"] == v2)["history"]
    assert [h["status"] for h in history][-3:] == ["staging", "production", "archived"]
    with pytest.raises(registry.ModelNotAvailable):
        registry.set_status("does-not-exist", "staging", "admin")
