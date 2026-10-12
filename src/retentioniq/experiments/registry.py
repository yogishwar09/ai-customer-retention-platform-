"""Experiment registry, deterministic assignment and analysis."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from retentioniq.experiments import stats
from retentioniq.storage.repository import Repository
from retentioniq.storage.util import new_id, now_iso

STATUSES = ("draft", "running", "stopped", "completed")
GUARDRAIL_OUTCOMES = ("opted_out", "declined")


def assign_arm(experiment_id: str, customer_id: str, control_share: float) -> str:
    """Stable hash assignment: the same customer always lands in the same arm of an
    experiment, and assignment is independent across experiments (salted by id)."""
    h = hashlib.sha256(f"{experiment_id}:{customer_id}".encode()).hexdigest()
    return "control" if int(h[:8], 16) / 0xFFFFFFFF < control_share else "treatment"


class Experiments:
    def __init__(self, repo: Repository):
        self.repo = repo

    def create(self, *, name: str, actor: str, hypothesis: str = "", action_code: str | None = None,
               control_share: float = 0.1, min_detectable_effect: float = 0.05, alpha: float = 0.05,
               power: float = 0.8, start: bool = False) -> str:
        if not 0.01 <= control_share <= 0.5:
            raise ValueError("control_share must be between 0.01 and 0.5")
        if not 0 < min_detectable_effect < 0.5:
            raise ValueError("min_detectable_effect must be between 0 and 0.5")
        if start and self._same_scope_running(action_code):
            raise ValueError(f"A running experiment already has this scope ({action_code or 'all actions'})")
        exp_id = new_id("exp")
        ts = now_iso()
        self.repo.execute(
            "INSERT INTO experiments (experiment_id, org_id, name, hypothesis, action_code, control_share, status, "
            "primary_metric, min_detectable_effect, alpha, power, created_by, created_at, started_at) VALUES "
            "(:i, :org, :n, :h, :a, :c, :s, 'retention_rate', :m, :al, :p, :by, :t, :st)",
            dict(i=exp_id, n=name, h=hypothesis, a=action_code, c=control_share, s="running" if start else "draft",
                 m=min_detectable_effect, al=alpha, p=power, by=actor, t=ts, st=ts if start else None))
        self.repo.audit(actor, "experiment.created", "experiment", exp_id, {"name": name, "action_code": action_code})
        return exp_id

    def set_status(self, experiment_id: str, status: str, actor: str) -> None:
        if status not in STATUSES:
            raise ValueError(status)
        exp = self.get(experiment_id)
        if exp is None:
            raise KeyError(experiment_id)
        if status == "running" and exp["status"] != "running":
            clash = self._same_scope_running(exp["action_code"])
            if clash and clash["experiment_id"] != experiment_id:
                raise ValueError(f"'{clash['name']}' is already running with the same scope")
        ts = now_iso()
        self.repo.execute(
            "UPDATE experiments SET status = CAST(:s AS VARCHAR), started_at = COALESCE(started_at, "
            "CASE WHEN CAST(:s AS VARCHAR) = 'running' THEN CAST(:t AS VARCHAR) END), ended_at = CASE WHEN "
            "CAST(:s AS VARCHAR) IN ('stopped', 'completed') THEN CAST(:t AS VARCHAR) ELSE ended_at END "
            "WHERE experiment_id = :i AND org_id = :org", dict(s=status, t=ts, i=experiment_id))
        self.repo.audit(actor, f"experiment.{status}", "experiment", experiment_id)

    def get(self, experiment_id: str) -> dict | None:
        df = self.repo.query_df("SELECT * FROM experiments WHERE org_id = :org AND experiment_id = :i",
                                dict(i=experiment_id))
        return None if df.empty else df.iloc[0].to_dict()

    def list(self) -> pd.DataFrame:
        return self.repo.query_df("SELECT * FROM experiments WHERE org_id = :org ORDER BY created_at DESC")

    def _same_scope_running(self, action_code: str | None) -> dict | None:
        """Two experiments may not randomise the same population at once."""
        sql = "SELECT * FROM experiments WHERE org_id = :org AND status = 'running' AND "
        sql += "action_code IS NULL" if action_code is None else "action_code = :a"
        df = self.repo.query_df(sql, dict(a=action_code))
        return None if df.empty else df.iloc[0].to_dict()

    def running_for(self, action_code: str | None) -> dict | None:
        """The running experiment covering an action: a specific one wins over a global one."""
        df = self.repo.query_df(
            "SELECT * FROM experiments WHERE org_id = :org AND status = 'running' "
            "AND (action_code = :a OR action_code IS NULL) ORDER BY CASE WHEN action_code IS NULL THEN 1 ELSE 0 END, "
            "started_at", dict(a=action_code))
        return None if df.empty else df.iloc[0].to_dict()

    def assignment(self, action_code: str, customer_id: str | None) -> tuple[str | None, str]:
        exp = self.running_for(action_code) if customer_id else None
        if exp is None:
            return None, "treatment"
        return exp["experiment_id"], assign_arm(exp["experiment_id"], customer_id, float(exp["control_share"]))

    def analyze(self, experiment_id: str, baseline_retention: float | None = None) -> dict[str, Any]:
        exp = self.get(experiment_id)
        if exp is None:
            raise KeyError(experiment_id)
        df = self.repo.query_df(
            "SELECT a.experiment_arm AS arm, o.outcome_type FROM retention_actions a "
            "LEFT JOIN action_outcomes o ON o.action_id = a.action_id "
            "WHERE a.org_id = :org AND a.experiment_id = :i", dict(i=experiment_id))
        assigned = df.groupby("arm").size().to_dict() if not df.empty else {}
        final = df[df["outcome_type"].isin(["retained", "churned"])] if not df.empty else df
        arms = {}
        for arm in ("treatment", "control"):
            sub = final[final["arm"] == arm] if not final.empty else final
            all_arm = df[df["arm"] == arm] if not df.empty else df
            arms[arm] = {"assigned": int(assigned.get(arm, 0)), "with_outcome": int(len(sub)),
                         "retained": int((sub["outcome_type"] == "retained").sum()) if len(sub) else 0,
                         "guardrail_events": int(all_arm["outcome_type"].isin(GUARDRAIL_OUTCOMES).sum())
                         if len(all_arm) else 0}
        t, c = arms["treatment"], arms["control"]
        base = baseline_retention if baseline_retention is not None else (
            c["retained"] / c["with_outcome"] if c["with_outcome"] else 0.75)
        n_equal = stats.required_sample_size(base, float(exp["min_detectable_effect"]), float(exp["alpha"]),
                                             float(exp["power"]))
        total_needed = stats.unequal_allocation_total(n_equal, float(exp["control_share"]))
        needed_control = int(total_needed * float(exp["control_share"]))
        needed_treatment = total_needed - needed_control
        test = stats.two_proportion_test(t["retained"], t["with_outcome"], c["retained"], c["with_outcome"],
                                         float(exp["alpha"]))
        enough = t["with_outcome"] >= needed_treatment and c["with_outcome"] >= needed_control
        if test["status"] != "ok":
            verdict = "collecting"
        elif not enough:
            verdict = "underpowered"
        elif test["significant"]:
            verdict = "positive" if test["difference"] > 0 else "negative"
        else:
            verdict = "no_detectable_effect"
        return {"experiment": exp, "arms": arms, "test": test, "baseline_retention": base,
                "required": {"treatment": needed_treatment, "control": needed_control, "total": total_needed},
                "achieved_power": stats.achieved_power(base, float(exp["min_detectable_effect"]),
                                                       t["with_outcome"], c["with_outcome"], float(exp["alpha"])),
                "verdict": verdict}
