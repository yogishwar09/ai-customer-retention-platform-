"""Uplift (heterogeneous treatment effect) modelling.

Churn models answer "who may leave?". Uplift models answer "for whom does the
intervention change the outcome?". That is only identifiable from randomised
treatment/control data, so this module:

* fits a **T-learner** (separate outcome models for treated and control) on
  real experiment outcomes once enough exist, and
* provides a **simulation lab** that generates a synthetic randomised campaign
  with a *known* effect function, to validate that the pipeline recovers it.
  Simulated results are never presented as real impact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from retentioniq.data.schema import FEATURE_COLUMNS
from retentioniq.features.pipeline import build_preprocessor

MIN_PER_ARM = 200


def _learner(seed: int) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()),
                     ("model", XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                                             colsample_bytree=0.8, eval_metric="logloss", n_jobs=4,
                                             random_state=seed))])


@dataclass
class TLearner:
    treated: Pipeline
    control: Pipeline

    @classmethod
    def fit(cls, X: pd.DataFrame, treatment: np.ndarray, retained: np.ndarray, seed: int = 0) -> TLearner:
        t = treatment.astype(bool)
        if t.sum() < 20 or (~t).sum() < 20:
            raise ValueError("Each arm needs at least 20 observations to fit")
        m_t = clone(_learner(seed)).fit(X[t], retained[t])
        m_c = clone(_learner(seed + 1)).fit(X[~t], retained[~t])
        return cls(m_t, m_c)

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """Estimated increase in retention probability if treated."""
        return self.treated.predict_proba(X)[:, 1] - self.control.predict_proba(X)[:, 1]


def qini_curve(uplift: np.ndarray, treatment: np.ndarray, retained: np.ndarray,
               points: int = 50) -> pd.DataFrame:
    """Incremental retained customers when targeting the top-x% by predicted uplift."""
    order = np.argsort(-uplift)
    t, y = treatment[order].astype(bool), retained[order].astype(float)
    n = len(uplift)
    rows = []
    for frac in np.linspace(0, 1, points + 1):
        k = int(round(frac * n))
        tt, yy = t[:k], y[:k]
        n_t, n_c = tt.sum(), (~tt).sum()
        y_t, y_c = yy[tt].sum(), yy[~tt].sum()
        inc = y_t - (y_c * n_t / n_c if n_c else 0.0)
        rows.append({"fraction_targeted": float(frac), "incremental_retained": float(inc)})
    out = pd.DataFrame(rows)
    total = out["incremental_retained"].iloc[-1]
    out["random"] = out["fraction_targeted"] * total
    return out


def auuc_gain(curve: pd.DataFrame) -> float:
    """Area between the Qini curve and the random-targeting line (normalised by n points)."""
    return float((curve["incremental_retained"] - curve["random"]).mean())


def segment(uplift: np.ndarray, base_risk: np.ndarray, uplift_threshold: float = 0.02,
            risk_threshold: float = 0.35) -> np.ndarray:
    seg = np.where(uplift < -uplift_threshold, "Sleeping dogs",
                   np.where(uplift >= uplift_threshold, "Persuadables",
                            np.where(base_risk >= risk_threshold, "Lost causes", "Sure things")))
    return seg


# ------------------------------------------------------------- simulation
def true_effect(X: pd.DataFrame) -> np.ndarray:
    """Ground-truth effect used only by the simulator (reduction in churn probability)."""
    tau = np.full(len(X), 0.02)
    mtm_new = (X["Contract"] == "Month-to-month") & (X["tenure"] < 12)
    fiber_nosupport = (X["InternetService"] == "Fiber optic") & (X["TechSupport"] == "No")
    loyal = (X["Contract"] == "Two year") & (X["tenure"] > 48)
    tau = tau + 0.12 * mtm_new + 0.07 * fiber_nosupport - 0.06 * loyal
    return tau.to_numpy() if hasattr(tau, "to_numpy") else tau


def simulate_campaign(X: pd.DataFrame, base_churn: np.ndarray, *, treat_share: float = 0.5,
                      seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tau = true_effect(X)
    treated = rng.random(len(X)) < treat_share
    p_churn = np.clip(base_churn - tau * treated, 0.001, 0.999)
    churned = rng.random(len(X)) < p_churn
    df = X[FEATURE_COLUMNS].copy().reset_index(drop=True)
    df["treatment"] = treated.astype(int)
    df["retained"] = (~churned).astype(int)
    df["true_uplift"] = tau
    df["base_churn"] = base_churn
    return df


def run_simulation(X: pd.DataFrame, base_churn: np.ndarray, seed: int = 7, population: int = 20_000) -> dict[str, Any]:
    """Bootstrap the portfolio to ``population`` synthetic customers (uplift effects of a few
    points need tens of thousands of observations to estimate), then simulate."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(X), size=max(population, len(X)))
    Xb = X.iloc[idx].reset_index(drop=True).copy()
    Xb["tenure"] = np.clip(Xb["tenure"].to_numpy() + rng.integers(-2, 3, len(Xb)), 0, 120)
    Xb["MonthlyCharges"] = Xb["MonthlyCharges"].to_numpy() * rng.uniform(0.95, 1.05, len(Xb))
    pb = np.clip(np.asarray(base_churn)[idx] * rng.uniform(0.9, 1.1, len(Xb)), 0.001, 0.999)
    sim = simulate_campaign(Xb, pb, seed=seed)
    train, test = train_test_split(sim, test_size=0.4, random_state=seed, stratify=sim["treatment"])
    model = TLearner.fit(train[FEATURE_COLUMNS], train["treatment"].to_numpy(), train["retained"].to_numpy(), seed)
    pred = model.predict_uplift(test[FEATURE_COLUMNS])
    curve = qini_curve(pred, test["treatment"].to_numpy(), test["retained"].to_numpy())
    oracle = qini_curve(test["true_uplift"].to_numpy(), test["treatment"].to_numpy(), test["retained"].to_numpy())
    risk_curve = qini_curve(test["base_churn"].to_numpy(), test["treatment"].to_numpy(), test["retained"].to_numpy())
    corr = float(np.corrcoef(pred, test["true_uplift"])[0, 1])
    segs = segment(pred, test["base_churn"].to_numpy())
    seg_table = (pd.DataFrame({"segment": segs, "predicted_uplift": pred, "true_uplift": test["true_uplift"],
                               "base_churn": test["base_churn"]})
                 .groupby("segment").agg(customers=("segment", "size"),
                                         avg_predicted_uplift=("predicted_uplift", "mean"),
                                         avg_true_uplift=("true_uplift", "mean"),
                                         avg_base_churn=("base_churn", "mean")).reset_index())
    return {"curve": curve, "oracle": oracle, "risk_curve": risk_curve, "auuc_gain": auuc_gain(curve),
            "auuc_gain_oracle": auuc_gain(oracle), "auuc_gain_risk": auuc_gain(risk_curve),
            "correlation_with_truth": corr, "segments": seg_table, "n_train": len(train), "n_test": len(test),
            "population": len(sim),
            "observed_ate": float(sim.loc[sim.treatment == 1, "retained"].mean()
                                  - sim.loc[sim.treatment == 0, "retained"].mean()),
            "true_ate": float(sim["true_uplift"].mean())}


def real_outcome_data(repo) -> pd.DataFrame:
    """Actions with a final outcome and a randomised arm, joined to the scored features."""
    import json
    df = repo.query_df(
        "SELECT a.experiment_arm AS arm, o.outcome_type, s.features FROM retention_actions a "
        "JOIN action_outcomes o ON o.action_id = a.action_id "
        "JOIN customer_features_snapshot s ON s.prediction_id = a.prediction_id "
        "WHERE a.org_id = :org AND a.experiment_id IS NOT NULL AND o.outcome_type IN ('retained', 'churned')")
    if df.empty:
        return df
    feats = pd.json_normalize(df["features"].map(json.loads))
    out = feats[FEATURE_COLUMNS].copy()
    out["treatment"] = (df["arm"] == "treatment").astype(int).to_numpy()
    out["retained"] = (df["outcome_type"] == "retained").astype(int).to_numpy()
    return out
