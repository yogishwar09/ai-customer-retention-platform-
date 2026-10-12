import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from retentioniq import config
from retentioniq.monitoring.summary import monitoring_summary
from ui import common
from ui.theme import SERIES, layout

svc = common.service()
db = common.db()
common.header("Monitoring", "Data quality, drift, prediction behaviour, delayed-label performance and impact.")

m = config.settings()["monitoring"]
jobs_df = db.batch_jobs()
options = {f"Last {d} day{'s' if d > 1 else ''}": ("days", d) for d in (1, 7, 30)}
options.update({f"Batch job · {r.filename_safe} ({r.started_at[:16].replace('T', ' ')})": ("job", r.job_id)
                for r in jobs_df.itertuples()} if not jobs_df.empty else {})
labels = list(options)
default_label = f"Last {m['drift_window_days']} days"
choice = st.selectbox("Monitoring window (population compared with the training reference)", labels,
                      index=labels.index(default_label) if default_label in labels else 0)
kind, value = options[choice]
s = (monitoring_summary(db, svc, batch_job_id=value) if kind == "job"
     else monitoring_summary(db, svc, window_days=value))
perf = s["delayed_label_performance"]
ref = svc.model.evaluation["test"]

common.kpi_row([
    common.kpi("Open alerts", f"{len(s['alerts'])}",
               ", ".join(sorted({a['signal'].replace('_', ' ') for a in s["alerts"]})) or "all clear"),
    common.kpi("Predictions in window", f"{s['window_rows']:,}",
               f"{s['customers_scored']:,} customers scored overall"),
    common.kpi("Mean estimated risk", f"{s['mean_score']:.3f}" if s["mean_score"] is not None else "—",
               f"training reference {svc.reference_profile['score']['mean']:.3f}" if svc.reference_profile else ""),
    common.kpi("Delayed-label PR-AUC", f"{perf['pr_auc']:.3f}" if perf["status"] == "ok" else "—",
               f"n={perf['n']:,} · test baseline {ref['pr_auc']:.3f}" if perf["status"] == "ok"
               else f"needs ≥ {m['min_labels_for_performance']} labels"),
    common.kpi("DQ warning rate", f"{s['data_quality_warning_rate']:.1%}" if s["data_quality_warning_rate"] is not None else "—",
               f"alert above {m['dq_warning_rate_alert']:.0%}"),
])

st.subheader("Alerts")
if not s["alerts"]:
    common.note("● No active alerts.", "ok")
else:
    alerts = pd.DataFrame(s["alerts"])
    alerts["severity"] = alerts["severity"].map({"high": "▲ High", "medium": "◆ Medium"})
    st.dataframe(alerts[["severity", "signal", "message", "owner", "runbook"]], hide_index=True,
                 use_container_width=True)

tab1, tab2, tab3, tab4 = st.tabs(["Data drift", "Volume & quality", "Delayed-label performance",
                                  "Impact (experiment)"])
with tab1:
    if s["drift_status"] != "ok":
        common.empty_state("Not enough data for drift",
                           f"{s['window_rows']} predictions in {s['window']}; PSI needs at least "
                           f"{s['drift_min_rows']}.")
    else:
        drift = pd.DataFrame(s["drift"])
        drift["status_label"] = drift["status"].map({"stable": "● Stable", "warning": "◆ Warning",
                                                     "alert": "▲ Alert"})
        d = drift.iloc[::-1]
        fig = go.Figure(go.Bar(y=d["feature"], x=d["psi"], orientation="h",
                               marker=dict(color=[{"stable": "#9EC5F4", "warning": "#FAB219", "alert": "#D03B3B"}[v]
                                                  for v in d["status"]], cornerradius=4),
                               customdata=d[["status_label", "reference", "current"]].values,
                               hovertemplate="%{y}<br>PSI %{x:.3f} · %{customdata[0]}<br>reference %{customdata[1]}"
                                             "<br>current %{customdata[2]}<extra></extra>"))
        for x in (m["psi_warning"], m["psi_alert"]):
            fig.add_vline(x=x, line=dict(color="#475569", dash="dot", width=1))
        layout(fig, height=520, title=f"PSI vs. training reference · {s['window']}",
               xtitle="PSI")
        c1, c2 = st.columns([1.4, 1])
        c1.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        c2.dataframe(drift[["feature", "psi", "status_label", "reference", "current"]], hide_index=True,
                     use_container_width=True, height=520,
                     column_config={"psi": st.column_config.NumberColumn("PSI", format="%.3f"),
                                    "status_label": "Status"})
        st.caption("Dotted lines mark the warning (0.10) and alert (0.25) thresholds. PSI < 0.10 stable · 0.10-0.25 moderate · > 0.25 significant. Drift is a reason to "
                   "investigate (see docs/runbooks.md), not an automatic retraining trigger. "
                   "Try Batch scoring → 'Drifted population' to see alerts fire.")

with tab2:
    preds = db.query_df("SELECT substr(scored_at,1,10) AS day, source, COUNT(*) AS n, "
                        "AVG(churn_probability) AS mean_p, "
                        "AVG(CASE WHEN data_quality_status != 'passed' THEN 1.0 ELSE 0 END) AS dq "
                        "FROM predictions WHERE org_id = :org GROUP BY day, source ORDER BY day")
    if preds.empty:
        common.empty_state("No predictions yet", "")
    else:
        daily = preds.groupby("day").agg(n=("n", "sum")).reset_index()
        mean = preds.assign(w=preds["mean_p"] * preds["n"]).groupby("day").agg(w=("w", "sum"), n=("n", "sum"))
        daily["mean_p"] = (mean["w"] / mean["n"]).values
        c1, c2 = st.columns(2)
        fig = go.Figure(go.Bar(x=pd.to_datetime(daily["day"]), y=daily["n"],
                               marker=dict(color=SERIES[0], cornerradius=4),
                               hovertemplate="%{x|%d %b}: %{y:,} predictions<extra></extra>"))
        layout(fig, height=280, title="Predictions per day", ytitle="Predictions")
        c1.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        fig = go.Figure(go.Scatter(x=pd.to_datetime(daily["day"]), y=daily["mean_p"], mode="lines+markers",
                                   line=dict(color=SERIES[0], width=2), marker=dict(size=8),
                                   hovertemplate="%{x|%d %b}: mean %{y:.3f}<extra></extra>"))
        if svc.reference_profile:
            fig.add_hline(y=svc.reference_profile["score"]["mean"], line=dict(color="#475569", dash="dot", width=1))
            fig.add_annotation(x=1, xref="paper", y=svc.reference_profile["score"]["mean"], text="training mean",
                               showarrow=False, xanchor="right", yanchor="bottom", font=dict(size=11))
        layout(fig, height=280, title="Mean estimated risk per day", ytitle="Mean score")
        fig.update_yaxes(range=[0, max(0.6, daily["mean_p"].max() * 1.2)])
        c2.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    jobs = db.batch_jobs()
    st.markdown("**Batch jobs**")
    if jobs.empty:
        st.caption("No stored batch jobs yet.")
    else:
        st.dataframe(jobs, hide_index=True, use_container_width=True)

with tab3:
    if perf["status"] != "ok":
        common.empty_state("Waiting for outcome labels",
                           f"{perf['n']} labelled customers so far; at least "
                           f"{m['min_labels_for_performance']} are needed.")
    else:
        rows = [("ROC-AUC", perf["roc_auc"], ref["roc_auc"]), ("PR-AUC", perf["pr_auc"], ref["pr_auc"]),
                ("Precision @ threshold", perf["precision"], ref["at_threshold"]["precision"]),
                ("Recall @ threshold", perf["recall"], ref["at_threshold"]["recall"])]
        fig = go.Figure()
        fig.add_bar(x=[r[0] for r in rows], y=[r[2] for r in rows], name="Test baseline (training time)",
                    marker=dict(color=SERIES[0], cornerradius=4), hovertemplate="%{x}: %{y:.3f}<extra>baseline</extra>")
        fig.add_bar(x=[r[0] for r in rows], y=[r[1] for r in rows], name=f"Delayed labels (n={perf['n']:,})",
                    marker=dict(color=SERIES[1], cornerradius=4), hovertemplate="%{x}: %{y:.3f}<extra>live</extra>")
        layout(fig, height=320, legend=True)
        fig.update_layout(barmode="group", bargroupgap=0.08)
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption(f"Observed churn among labelled customers {perf['observed_churn_rate']:.1%} vs mean "
                   f"estimated risk {perf['mean_predicted']:.1%}. Labels arrive after the score; only "
                   "customers with a label are included.")
    if common.can("manager"):
        with st.expander("Import outcome labels (CSV: customerID, churned)"):
            up = st.file_uploader("Labels file", type=["csv"], key="labels_upload")
            if up is not None:
                lab = pd.read_csv(up, dtype=str)
                cols = {c.lower().replace("_", ""): c for c in lab.columns}
                if "customerid" not in cols or not ({"churned", "churn"} & set(cols)):
                    st.error("Need columns customerID and churned (1/0 or Yes/No).")
                else:
                    target = cols.get("churned") or cols.get("churn")
                    vals = lab[target].str.strip().str.lower().map({"1": 1, "yes": 1, "0": 0, "no": 0})
                    ok = lab[vals.notna()]
                    if st.button(f"Import {len(ok):,} labels ({(vals.isna()).sum()} invalid skipped)"):
                        n = db.upsert_labels(list(zip(ok[cols['customerid']].str.strip(), vals[vals.notna()].astype(int), strict=True)),
                                             source=f"upload:{html.escape(up.name)[:60]}", actor=common.actor())
                        st.success(f"Imported {n:,} labels")
                        st.rerun()

with tab4:
    from retentioniq.experiments.registry import Experiments
    common.note("Impact is measured only through randomised experiments (control customers are held out). "
                "Model accuracy is not impact. Full analysis lives in <b>Experiments &amp; uplift</b>.")
    ex = Experiments(db)
    lst = ex.list()
    if lst.empty:
        common.empty_state("No experiments", "Create one in Experiments &amp; uplift.")
    else:
        rows = []
        for e in lst.itertuples():
            a = ex.analyze(e.experiment_id)
            t = a["test"]
            rows.append({"experiment": e.name, "status": e.status, "verdict": a["verdict"].replace("_", " "),
                         "treatment outcomes": a["arms"]["treatment"]["with_outcome"],
                         "control outcomes": a["arms"]["control"]["with_outcome"],
                         "difference": t.get("difference"), "p_value": t.get("p_value")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                     column_config={"difference": st.column_config.NumberColumn(format="%+.3f"),
                                    "p_value": st.column_config.NumberColumn(format="%.3f")})
    wd = s.get("webhook_deliveries", {})
    st.markdown(f"**Integrations:** {wd.get('succeeded', 0):,} delivered · {wd.get('pending', 0):,} pending · "
                f"{wd.get('dead', 0):,} dead-lettered")
