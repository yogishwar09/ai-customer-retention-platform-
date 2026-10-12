"""Evaluation metrics beyond accuracy: ranking, calibration, lift, capacity and
subgroup checks. All functions are pure and work on numpy arrays."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / len(y)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "flagged_share": float(pred.mean()),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def expected_cost(y: np.ndarray, p: np.ndarray, threshold: float,
                  cost_fp: float, cost_fn: float) -> float:
    pred = p >= threshold
    fp = np.sum(pred & (y == 0))
    fn = np.sum(~pred & (y == 1))
    return float((fp * cost_fp + fn * cost_fn) / len(y))


def choose_threshold(y: np.ndarray, p: np.ndarray, cost_fp: float, cost_fn: float) -> float:
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    costs = [expected_cost(y, p, t, cost_fp, cost_fn) for t in grid]
    return float(grid[int(np.argmin(costs))])


def ranking_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "prevalence": float(np.mean(y)),
    }


def bootstrap_ci(y: np.ndarray, p: np.ndarray, n: int, seed: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    roc, pr = [], []
    idx = np.arange(len(y))
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if y[s].min() == y[s].max():
            continue
        roc.append(roc_auc_score(y[s], p[s]))
        pr.append(average_precision_score(y[s], p[s]))
    pct = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    return {"roc_auc": pct(roc), "pr_auc": pct(pr)}


def precision_at_k(y: np.ndarray, p: np.ndarray, ks: list[int]) -> dict[str, float]:
    order = np.argsort(-p)
    return {str(k): float(y[order[:k]].mean()) for k in ks if k <= len(y)}


def lift_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict[str, Any]]:
    order = np.argsort(-p)
    chunks = np.array_split(order, bins)
    base = y.mean()
    total_pos = y.sum()
    rows, cum_pos = [], 0
    for i, chunk in enumerate(chunks, start=1):
        rate = y[chunk].mean()
        cum_pos += y[chunk].sum()
        rows.append({
            "decile": i,
            "customers": int(len(chunk)),
            "min_score": float(p[chunk].min()),
            "churn_rate": float(rate),
            "lift": float(rate / base) if base else 0.0,
            "cumulative_capture": float(cum_pos / total_pos) if total_pos else 0.0,
        })
    return rows


def calibration_points(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    order = np.argsort(p)
    out = []
    for chunk in np.array_split(order, bins):
        out.append({"mean_predicted": float(p[chunk].mean()),
                    "observed_rate": float(y[chunk].mean()),
                    "count": int(len(chunk))})
    return out


def curves(y: np.ndarray, p: np.ndarray, max_points: int = 200) -> dict[str, Any]:
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)

    def thin(a: np.ndarray, b: np.ndarray) -> tuple[list[float], list[float]]:
        if len(a) > max_points:
            keep = np.linspace(0, len(a) - 1, max_points).astype(int)
            a, b = a[keep], b[keep]
        return [round(float(v), 4) for v in a], [round(float(v), 4) for v in b]

    fpr_l, tpr_l = thin(fpr, tpr)
    rec_l, prec_l = thin(rec, prec)
    return {"roc": {"fpr": fpr_l, "tpr": tpr_l}, "pr": {"recall": rec_l, "precision": prec_l}}


def subgroup_metrics(y: np.ndarray, p: np.ndarray, groups: dict[str, np.ndarray],
                     threshold: float, min_size: int = 30) -> list[dict[str, Any]]:
    rows = []
    for attribute, values in groups.items():
        for level in sorted(np.unique(values), key=str):
            mask = values == level
            n = int(mask.sum())
            row: dict[str, Any] = {"attribute": attribute, "group": str(level), "n": n,
                                   "prevalence": float(y[mask].mean()) if n else None}
            if n >= min_size and 0 < y[mask].sum() < n:
                tm = threshold_metrics(y[mask], p[mask], threshold)
                row.update({
                    "roc_auc": float(roc_auc_score(y[mask], p[mask])),
                    "recall": tm["recall"], "precision": tm["precision"],
                    "flagged_share": tm["flagged_share"],
                })
            rows.append(row)
    return rows
