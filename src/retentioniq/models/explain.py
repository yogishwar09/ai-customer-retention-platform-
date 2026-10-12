"""Per-customer explanations aggregated back to contract-level features.

Methods (chosen by model type):
* XGBoost  - native TreeSHAP contributions (log-odds space)
* Random forest - shap.TreeExplainer (probability space)
* Logistic regression - exact linear contributions relative to the training
  mean (log-odds space)

Contributions describe how the *model output* moves under the method's
assumptions. They are not causal effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from retentioniq.data.schema import FIELD_BY_NAME
from retentioniq.features.pipeline import transformed_to_source_index

METHOD_LABELS = {
    "xgboost": "TreeSHAP (XGBoost native), log-odds",
    "random_forest": "TreeSHAP (shap.TreeExplainer), probability",
    "logistic_regression": "Linear contribution vs. training mean, log-odds",
}


@dataclass
class Driver:
    feature: str
    label: str
    value: Any
    contribution: float
    direction: str  # increases_risk | decreases_risk
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"feature": self.feature, "label": self.label, "value": _jsonable(self.value),
                "contribution": round(float(self.contribution), 5),
                "direction": self.direction, "reason": self.reason}


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    return v


def reason_text(feature: str, value: Any, direction: str) -> str:
    """Plain-language reason code for a driver."""
    up = direction == "increases_risk"
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return f"{FIELD_BY_NAME[feature].label} not recorded"
    if feature == "tenure":
        months = int(value)
        return f"{'Short' if up else 'Long'} tenure ({months} month{'s' if months != 1 else ''})"
    if feature == "MonthlyCharges":
        return f"{'High' if up else 'Moderate'} monthly charges (${value:,.2f})"
    if feature == "TotalCharges":
        return f"Lifetime charges of ${value:,.0f}"
    if feature == "Contract":
        return f"{value} contract"
    if feature == "PaymentMethod":
        return f"Pays by {str(value).lower()}"
    if feature == "InternetService":
        return "No internet service" if value == "No" else f"{value} internet"
    if feature == "SeniorCitizen":
        return "Senior citizen" if int(value) == 1 else "Not a senior citizen"
    if feature == "PaperlessBilling":
        return "Paperless billing" if value == "Yes" else "Paper billing"
    label = FIELD_BY_NAME[feature].label
    if value == "Yes":
        return f"Has {label.lower()}"
    if value == "No":
        return f"No {label.lower()}"
    return f"{label}: {value}"


class Explainer:
    def __init__(self, pipeline: Any, algorithm: str, background: pd.DataFrame | None = None):
        self.pipeline = pipeline
        self.algorithm = algorithm
        self.pre = pipeline.named_steps["preprocess"]
        self.clf = pipeline.named_steps["model"]
        self.source_index = np.array(transformed_to_source_index())
        self.sources = list(dict.fromkeys(self.source_index))
        self._shap = None
        self._mean = None
        if algorithm == "logistic_regression" and background is not None:
            self._mean = self.pre.transform(background).mean(axis=0)

    @property
    def method(self) -> str:
        return METHOD_LABELS.get(self.algorithm, self.algorithm)

    def _raw_contributions(self, Xt: np.ndarray) -> np.ndarray:
        if self.algorithm == "xgboost":
            import xgboost as xgb
            contrib = self.clf.get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)
            return contrib[:, :-1]  # drop bias column
        if self.algorithm == "random_forest":
            if self._shap is None:
                import shap
                self._shap = shap.TreeExplainer(self.clf)
            values = self._shap.shap_values(Xt, check_additivity=False)
            if isinstance(values, list):
                return np.asarray(values[1])
            values = np.asarray(values)
            return values[:, :, 1] if values.ndim == 3 else values
        if self.algorithm == "logistic_regression":
            mean = self._mean if self._mean is not None else np.zeros(Xt.shape[1])
            return (Xt - mean) * self.clf.coef_[0]
        raise ValueError(f"No explainer for algorithm {self.algorithm}")

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Rows = customers, columns = contract features, values = contribution."""
        Xt = self.pre.transform(X)
        raw = self._raw_contributions(Xt)
        out = pd.DataFrame(0.0, index=X.index, columns=self.sources)
        for j, src in enumerate(self.source_index):
            out[src] += raw[:, j]
        return out

    def drivers(self, X: pd.DataFrame, top_n: int = 5) -> list[list[Driver]]:
        contrib = self.contributions(X)
        result = []
        for idx in X.index:
            row = contrib.loc[idx].sort_values(key=np.abs, ascending=False)
            drivers = []
            for feature, value in row.head(top_n).items():
                if abs(value) < 1e-9:
                    continue
                direction = "increases_risk" if value > 0 else "decreases_risk"
                raw_value = X.at[idx, feature]
                drivers.append(Driver(feature, FIELD_BY_NAME[feature].label, raw_value,
                                      float(value), direction,
                                      reason_text(feature, raw_value, direction)))
            result.append(drivers)
        return result

    def stability(self, X: pd.DataFrame, top_k: int = 3, copies: int = 6,
                  seed: int = 0) -> list[dict[str, Any]]:
        """Do the top drivers survive small, plausible perturbations of numeric inputs?"""
        rng = np.random.default_rng(seed)
        base = self.contributions(X)
        results = []
        for idx in X.index:
            top = set(base.loc[idx].abs().nlargest(top_k).index)
            perturbed = pd.concat([X.loc[[idx]]] * copies, ignore_index=True)
            perturbed["MonthlyCharges"] = perturbed["MonthlyCharges"] * rng.uniform(0.95, 1.05, copies)
            perturbed["tenure"] = np.clip(perturbed["tenure"] + rng.integers(-1, 2, copies), 0, 120)
            if perturbed["TotalCharges"].notna().all():
                perturbed["TotalCharges"] = perturbed["TotalCharges"] * rng.uniform(0.97, 1.03, copies)
            pc = self.contributions(perturbed)
            jaccards = []
            for i in range(copies):
                other = set(pc.iloc[i].abs().nlargest(top_k).index)
                jaccards.append(len(top & other) / len(top | other))
            score = float(np.mean(jaccards))
            results.append({"score": round(score, 3), "stable": score >= 0.67})
        return results

    def global_importance(self, X: pd.DataFrame) -> list[dict[str, Any]]:
        contrib = self.contributions(X)
        mean_abs = contrib.abs().mean().sort_values(ascending=False)
        signed = contrib.mean()
        return [{"feature": f, "label": FIELD_BY_NAME[f].label,
                 "mean_abs_contribution": float(v), "mean_contribution": float(signed[f])}
                for f, v in mean_abs.items()]
