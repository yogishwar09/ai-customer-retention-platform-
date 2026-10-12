"""Reproducible training + evaluation.

    python -m retentioniq.models.train

1. Load and validate the dataset through the data contract.
2. Stratified split: train / validation / test (test is never used for choices).
3. Compare a transparent baseline (logistic regression) with tree ensembles
   using stratified CV on the training split.
4. Pick the model with the best validation PR-AUC; choose the operating
   threshold on validation by minimising the configured misclassification cost.
5. Refit the chosen configuration on train+validation, then report final
   metrics once on the held-out test set (with bootstrap CIs).
6. Persist pipeline + metadata + evaluation + drift reference profile,
   register the version, and write the evaluation report and model card.
"""

from __future__ import annotations

import json
import platform
import time
import warnings
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from retentioniq import __version__, config
from retentioniq.data.loader import file_fingerprint, load_training_data
from retentioniq.data.schema import FEATURE_COLUMNS, SCHEMA_VERSION, TARGET_COLUMN
from retentioniq.features.pipeline import build_preprocessor
from retentioniq.models import evaluate as ev
from retentioniq.models import registry
from retentioniq.models.explain import Explainer
from retentioniq.monitoring.drift import build_reference_profile

Log = Callable[[str], None]

# numpy 2.0 + Apple Accelerate emits spurious matmul RuntimeWarnings on macOS.
warnings.filterwarnings("ignore", message=r".*encountered in (matmul|dot)", category=RuntimeWarning)


def candidate_models(seed: int) -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, C=1.0),
        "random_forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, max_features="sqrt",
            n_jobs=-1, random_state=seed),
        "xgboost": XGBClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=2, reg_lambda=1.0,
            eval_metric="logloss", n_jobs=4, random_state=seed),
    }


def make_pipeline(estimator: Any) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("model", clone(estimator))])


def feedback_rows(org_id: str, exclude_ids: set[str]) -> tuple[pd.DataFrame, int]:
    """Labelled production snapshots (latest score before the label) from one tenant.
    Customers that overlap the reference held-out test split are excluded so test
    metrics stay comparable across versions."""
    import json

    from retentioniq.storage.repository import Repository
    repo = Repository(org_id)
    df = repo.query_df(
        """
        SELECT l.customer_id, l.churned, s.features FROM observed_labels l
        JOIN predictions p ON p.org_id = l.org_id AND p.customer_id = l.customer_id
        JOIN customer_features_snapshot s ON s.prediction_id = p.prediction_id
        WHERE l.org_id = :org AND p.scored_at = (
            SELECT MAX(p2.scored_at) FROM predictions p2 WHERE p2.org_id = l.org_id
            AND p2.customer_id = l.customer_id AND p2.scored_at <= l.observed_at)
        """)
    if df.empty:
        return pd.DataFrame(columns=[*FEATURE_COLUMNS, TARGET_COLUMN]), 0
    df = df.drop_duplicates("customer_id", keep="last")
    overlap = df["customer_id"].isin(exclude_ids)
    df = df[~overlap]
    if df.empty:
        return pd.DataFrame(columns=[*FEATURE_COLUMNS, TARGET_COLUMN]), int(overlap.sum())
    feats = pd.json_normalize(df["features"].map(json.loads))[FEATURE_COLUMNS]
    feats[TARGET_COLUMN] = df["churned"].astype(int).to_numpy()
    return feats, int(overlap.sum())


def tuned_xgboost(X: pd.DataFrame, y: np.ndarray, seed: int, cv: StratifiedKFold, log: Log) -> Any:
    search = RandomizedSearchCV(
        make_pipeline(candidate_models(seed)["xgboost"]),
        {"model__n_estimators": [200, 300, 400, 600], "model__max_depth": [2, 3, 4, 5],
         "model__learning_rate": [0.02, 0.03, 0.05, 0.08], "model__min_child_weight": [1, 2, 4, 8],
         "model__subsample": [0.7, 0.8, 0.9], "model__colsample_bytree": [0.6, 0.8, 1.0],
         "model__reg_lambda": [0.5, 1.0, 2.0, 5.0]},
        n_iter=16, scoring="average_precision", cv=cv, random_state=seed, n_jobs=1)
    search.fit(X, y)
    params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    log(f"  tuned xgboost best CV PR-AUC {search.best_score_:.3f} with {params}")
    return XGBClassifier(**{**candidate_models(seed)["xgboost"].get_params(), **params})


def train(log: Log = print, *, write_reports: bool = True, tune: bool = False,
          feedback_org: str | None = None, actor: str = "training-pipeline") -> dict[str, Any]:
    started = time.time()
    s = config.settings()
    mcfg = s["model"]
    seed = int(mcfg["random_seed"])

    df, validation = load_training_data()
    if validation.summary["rows_rejected"]:
        log(f"Data contract rejected {validation.summary['rows_rejected']} training rows")
    X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN].to_numpy()
    ids = df["customerID"].to_numpy()
    log(f"Loaded {len(df):,} labelled rows; churn prevalence {y.mean():.1%}")

    X_rest, X_test, y_rest, y_test, id_rest, id_test = train_test_split(
        X, y, ids, test_size=mcfg["test_size"], stratify=y, random_state=seed)
    val_share = mcfg["validation_size"] / (1 - mcfg["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_share, stratify=y_rest, random_state=seed)

    feedback_info = {"org_id": feedback_org, "rows_used": 0, "rows_excluded_test_overlap": 0}
    if feedback_org:
        fb, excluded = feedback_rows(feedback_org, set(map(str, id_test)))
        feedback_info.update(rows_used=int(len(fb)), rows_excluded_test_overlap=excluded)
        if len(fb):
            X_train = pd.concat([X_train, fb[FEATURE_COLUMNS]], ignore_index=True)
            y_train = np.concatenate([y_train, fb[TARGET_COLUMN].to_numpy()])
            X_rest = pd.concat([X_rest, fb[FEATURE_COLUMNS]], ignore_index=True)
            y_rest = np.concatenate([y_rest, fb[TARGET_COLUMN].to_numpy()])
        log(f"Feedback labels: {len(fb):,} rows added to training "
            f"({excluded:,} excluded because they overlap the held-out test set)")
    log(f"Split train={len(X_train):,} validation={len(X_val):,} test={len(X_test):,}")

    cv = StratifiedKFold(n_splits=mcfg["cv_folds"], shuffle=True, random_state=seed)
    comparison = []
    fitted: dict[str, Pipeline] = {}
    candidates = candidate_models(seed)
    if tune:
        candidates["xgboost_tuned"] = tuned_xgboost(X_train, y_train, seed, cv, log)
    for name, est in candidates.items():
        pipe = make_pipeline(est)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="average_precision")
        pipe.fit(X_train, y_train)
        p_val = pipe.predict_proba(X_val)[:, 1]
        rm = ev.ranking_metrics(y_val, p_val)
        comparison.append({
            "algorithm": name,
            "cv_pr_auc_mean": float(cv_scores.mean()),
            "cv_pr_auc_std": float(cv_scores.std()),
            "val_pr_auc": rm["pr_auc"], "val_roc_auc": rm["roc_auc"], "val_brier": rm["brier"],
        })
        fitted[name] = pipe
        log(f"  {name:<20} CV PR-AUC {cv_scores.mean():.3f}±{cv_scores.std():.3f} | "
            f"val PR-AUC {rm['pr_auc']:.3f} ROC-AUC {rm['roc_auc']:.3f} Brier {rm['brier']:.3f}")

    best = max(comparison, key=lambda r: (round(r["val_pr_auc"], 3), -r["val_brier"]))
    algorithm = best["algorithm"]
    for row in comparison:
        row["selected"] = row["algorithm"] == algorithm

    p_val_best = fitted[algorithm].predict_proba(X_val)[:, 1]
    threshold = ev.choose_threshold(y_val, p_val_best, mcfg["cost_false_positive"],
                                    mcfg["cost_false_negative"])
    log(f"Selected {algorithm}; operating threshold {threshold:.2f} "
        f"(cost FN:FP = {mcfg['cost_false_negative']}:{mcfg['cost_false_positive']})")

    final = make_pipeline(candidates[algorithm])
    final.fit(X_rest, y_rest)
    p_test = final.predict_proba(X_test)[:, 1]

    test_metrics = {
        **ev.ranking_metrics(y_test, p_test),
        "at_threshold": ev.threshold_metrics(y_test, p_test, threshold),
        "at_0_5": ev.threshold_metrics(y_test, p_test, 0.5),
        "ci_95": ev.bootstrap_ci(y_test, p_test, mcfg["bootstrap_resamples"], seed),
        "precision_at_k": ev.precision_at_k(y_test, p_test, [50, 100, 200, 300]),
        "expected_cost_per_customer": ev.expected_cost(
            y_test, p_test, threshold, mcfg["cost_false_positive"], mcfg["cost_false_negative"]),
    }
    log(f"Test ROC-AUC {test_metrics['roc_auc']:.3f} PR-AUC {test_metrics['pr_auc']:.3f} "
        f"Brier {test_metrics['brier']:.3f} | at threshold: precision "
        f"{test_metrics['at_threshold']['precision']:.3f} recall {test_metrics['at_threshold']['recall']:.3f}")

    explain_algo = "xgboost" if algorithm.startswith("xgboost") else algorithm
    explainer = Explainer(final, explain_algo, background=X_rest)
    sample = X_test.sample(n=min(500, len(X_test)), random_state=seed)
    global_importance = explainer.global_importance(sample)

    gate = s["release_gate"]
    gate_checks = {
        "test_roc_auc": [test_metrics["roc_auc"], gate["min_test_roc_auc"],
                         test_metrics["roc_auc"] >= gate["min_test_roc_auc"]],
        "test_pr_auc": [test_metrics["pr_auc"], gate["min_test_pr_auc"],
                        test_metrics["pr_auc"] >= gate["min_test_pr_auc"]],
        "brier": [test_metrics["brier"], gate["max_brier"], test_metrics["brier"] <= gate["max_brier"]],
    }
    passed = all(v[2] for v in gate_checks.values())
    # First gate-passing model bootstraps production; later ones need explicit approval.
    if not passed:
        status = "rejected"
    elif registry.production_version() is None:
        status = "production"
    else:
        status = "candidate"

    trained_at = datetime.now(timezone.utc)
    version = f"churn-{algorithm.replace('_', '-')}-{trained_at:%Y%m%d%H%M%S%f}"[:-3]
    folder = config.MODELS_DIR / version
    folder.mkdir(parents=True, exist_ok=True)
    model_path = folder / "model.joblib"
    joblib.dump(final, model_path)

    metadata = {
        "model_version": version,
        "algorithm": explain_algo,
        "candidate_name": algorithm,
        "operating_threshold": threshold,
        "tuned": tune,
        "feedback": feedback_info,
        "schema_version": SCHEMA_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "label_definition": ("Churn = 'Yes' in the IBM Telco sample: customer left within the "
                             "last month of the snapshot. No timestamps are available, so the "
                             "split is stratified random rather than time-based."),
        "split": {"method": "stratified random", "seed": seed,
                  "train": int(len(X_train)), "validation": int(len(X_val)),
                  "test": int(len(X_test)), "final_fit_rows": int(len(X_rest))},
        "class_prevalence": float(y.mean()),
        "trained_at": trained_at.isoformat(timespec="seconds"),
        "training_data_fingerprint": file_fingerprint(config.DATA_PATH),
        "training_rows": int(len(df)),
        "artifact_sha256": registry.sha256(model_path),
        "explanation_method": explainer.method,
        "package_versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                             "xgboost": xgboost.__version__, "pandas": pd.__version__,
                             "numpy": np.__version__, "retentioniq": __version__},
        "cost_assumptions": {"false_positive": mcfg["cost_false_positive"],
                             "false_negative": mcfg["cost_false_negative"]},
        "release_gate": {"passed": passed, "checks": gate_checks},
        "status_at_registration": status,
    }
    groups = {"gender": X_test["gender"].to_numpy(),
              "SeniorCitizen": X_test["SeniorCitizen"].astype(str).to_numpy()}
    evaluation = {
        "model_version": version,
        "comparison": comparison,
        "test": test_metrics,
        "curves": ev.curves(y_test, p_test),
        "calibration": ev.calibration_points(y_test, p_test),
        "lift": ev.lift_table(y_test, p_test),
        "subgroups": ev.subgroup_metrics(y_test, p_test, groups, threshold),
        "global_importance": global_importance,
        "test_predictions": {"customer_id": [str(i) for i in id_test],
                             "y": [int(v) for v in y_test],
                             "p": [round(float(v), 5) for v in p_test]},
    }
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (folder / "evaluation.json").write_text(json.dumps(evaluation), encoding="utf-8")
    profile = build_reference_profile(X_rest, final.predict_proba(X_rest)[:, 1])
    (folder / "reference_profile.json").write_text(json.dumps(profile), encoding="utf-8")

    registry.register(version, status, actor=actor, summary={
        "algorithm": algorithm,
        "feedback_rows": feedback_info["rows_used"],
        "test_roc_auc": round(test_metrics["roc_auc"], 4),
        "test_pr_auc": round(test_metrics["pr_auc"], 4),
        "brier": round(test_metrics["brier"], 4),
        "threshold": threshold,
        "gate_passed": passed,
    })
    log(f"Registered {version} as {status} "
        f"({'gate passed' if passed else 'release gate FAILED'}) in {time.time() - started:.1f}s")

    if write_reports:
        from retentioniq.reporting import publish
        from retentioniq.reporting import write_reports as _write
        _write(metadata, evaluation, folder)
        if status == "production":
            publish(version)
            log(f"Published reports to {config.REPORTS_DIR}")
    return {"version": version, "status": status, "metadata": metadata, "evaluation": evaluation}


def ensure_model(log: Log = print) -> str:
    """Train once if no production model exists (first start on a fresh host)."""
    version = registry.production_version()
    if version and (config.MODELS_DIR / version / "model.joblib").exists():
        return version
    return train(log)["version"]


if __name__ == "__main__":
    train()
