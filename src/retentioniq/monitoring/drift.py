"""Population Stability Index (PSI) drift checks against the training reference.

PSI < 0.10 stable · 0.10-0.25 moderate shift · > 0.25 significant shift
(thresholds configurable in settings.yaml). Drift is a signal to investigate,
not an automatic retraining trigger.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from retentioniq import config
from retentioniq.data.schema import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

_EPS = 1e-4


def _numeric_edges(values: np.ndarray, bins: int = 10) -> list[float]:
    values = values[~np.isnan(values)]
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    return [float(e) for e in edges]


def _hist(values: np.ndarray, edges: list[float]) -> list[float]:
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return [0.0] * (len(edges) - 1)
    inner = np.clip(values, edges[0], edges[-1])
    counts, _ = np.histogram(inner, bins=edges)
    return [float(c) for c in counts / counts.sum()]


def build_reference_profile(X: pd.DataFrame, scores: np.ndarray) -> dict[str, Any]:
    profile: dict[str, Any] = {"rows": int(len(X)), "numeric": {}, "categorical": {}}
    for col in NUMERIC_COLUMNS:
        vals = X[col].to_numpy(dtype=float)
        edges = _numeric_edges(vals)
        profile["numeric"][col] = {"edges": edges, "dist": _hist(vals, edges),
                                   "mean": float(np.nanmean(vals)),
                                   "missing_rate": float(np.isnan(vals).mean())}
    for col in CATEGORICAL_COLUMNS:
        dist = X[col].astype(str).value_counts(normalize=True)
        profile["categorical"][col] = {str(k): float(v) for k, v in dist.items()}
    edges = [i / 10 for i in range(11)]
    profile["score"] = {"edges": edges, "dist": _hist(np.asarray(scores, float), edges),
                        "mean": float(np.mean(scores))}
    return profile


def psi(expected: list[float], actual: list[float]) -> float:
    e = np.clip(np.asarray(expected, float), _EPS, None)
    a = np.clip(np.asarray(actual, float), _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status_for(value: float) -> str:
    m = config.settings()["monitoring"]
    if value >= m["psi_alert"]:
        return "alert"
    if value >= m["psi_warning"]:
        return "warning"
    return "stable"


def drift_report(reference: dict[str, Any], current: pd.DataFrame,
                 scores: np.ndarray | None = None) -> pd.DataFrame:
    rows = []
    for col, ref in reference["numeric"].items():
        if col not in current:
            continue
        vals = current[col].to_numpy(dtype=float)
        value = psi(ref["dist"], _hist(vals, ref["edges"]))
        rows.append({"feature": col, "type": "numeric", "psi": value,
                     "reference": f"mean {ref['mean']:.1f}",
                     "current": f"mean {np.nanmean(vals):.1f}" if len(vals) else "n/a"})
    for col, ref in reference["categorical"].items():
        if col not in current:
            continue
        cur = current[col].astype(str).value_counts(normalize=True)
        cats = sorted(set(ref) | set(cur.index))
        value = psi([ref.get(c, 0.0) for c in cats], [float(cur.get(c, 0.0)) for c in cats])
        top_ref = max(ref, key=ref.get)
        rows.append({"feature": col, "type": "categorical", "psi": value,
                     "reference": f"{top_ref} {ref[top_ref]:.0%}",
                     "current": f"{top_ref} {float(cur.get(top_ref, 0.0)):.0%}"})
    if scores is not None and len(scores):
        ref = reference["score"]
        value = psi(ref["dist"], _hist(np.asarray(scores, float), ref["edges"]))
        rows.append({"feature": "churn score", "type": "prediction", "psi": value,
                     "reference": f"mean {ref['mean']:.3f}",
                     "current": f"mean {float(np.mean(scores)):.3f}"})
    out = pd.DataFrame(rows)
    if len(out):
        out["status"] = out["psi"].map(status_for)
        out = out.sort_values("psi", ascending=False, ignore_index=True)
    return out
