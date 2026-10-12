import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from retentioniq import config
from retentioniq.decisioning.risk import tier_cutoffs
from ui import common
from ui.theme import RISK, RISK_BAR, SERIES, layout

svc = common.service()
db = common.db()
common.header("Executive overview",
              "Who is at risk, how much recurring revenue is exposed, and what the team is doing about it.")

latest = db.latest_scores()
if latest.empty:
    common.empty_state("No customers scored yet", "Upload a file in <b>Batch scoring</b> to get started.")
    st.stop()

cut = tier_cutoffs(svc.model.threshold)
horizon = config.settings()["value"]["horizon_months"]
delta = config.settings()["decision"]["risk_rising_delta"]
latest["revenue_at_risk"] = latest["churn_probability"] * latest["MonthlyCharges"] * horizon
tiers = latest["risk_tier"].value_counts().reindex(["high", "medium", "low"], fill_value=0)
actions = db.actions()
open_actions = actions[actions["status"].isin(["recommended", "approved", "in_progress"])] if not actions.empty else actions
pending_approval = (actions[(actions["status"] == "recommended") & (actions["requires_approval"] == 1)]
                    if not actions.empty else actions)
dq_rate = (latest["data_quality_status"] != "passed").mean()
fresh = pd.to_datetime(latest["scored_at"]).max()
labels = db.labels()

common.kpi_row([
    common.kpi("Customers scored", f"{len(latest):,}", f"latest score {fresh:%d %b %Y %H:%M} UTC"),
    common.kpi("High-risk customers", f"{tiers['high']:,}",
               f"{tiers['high'] / len(latest):.1%} of portfolio · p ≥ {cut['high']:.2f}"),
    common.kpi("Revenue at risk (est.)", common.money(latest["revenue_at_risk"].sum()),
               f"Σ p(churn) × monthly charges × {horizon} mo",
               "Estimate, not a forecast: probability-weighted annualised charges."),
    common.kpi("Open retention actions", f"{len(open_actions):,}",
               f"{len(pending_approval)} awaiting manager approval"),
    common.kpi("Rising risk", f"{int((latest['risk_change'] >= delta).sum()):,}",
               f"score up ≥ {delta:.2f} since previous scoring"),
    common.kpi("Data-quality warnings", f"{dq_rate:.1%}", "of scored customers"),
    common.kpi("Observed churn (labels)", f"{labels['churned'].mean():.1%}" if len(labels) else "—",
               f"{len(labels):,} delayed labels received" if len(labels) else "no outcome labels yet"),
])

if (latest["source"] == "demo_seed").any():
    common.note("Demo portfolio: the 1,409 held-out <b>test</b> customers from the public IBM Telco "
                "sample, scored with spread-out timestamps. Revenue figures are estimates based on "
                "<code>MonthlyCharges</code>.", "info")

c1, c2 = st.columns([1, 1.35])
with c1:
    fig = go.Figure(go.Bar(
        x=[RISK[t]["label"] for t in tiers.index], y=tiers.values,
        marker=dict(color=[RISK_BAR[t] for t in tiers.index], cornerradius=4),
        text=[f"{RISK[t]['icon']} {v:,}" for t, v in tiers.items()], textposition="outside",
        hovertemplate="%{x} risk: %{y:,} customers<extra></extra>"))
    layout(fig, title="Customers by risk tier", ytitle="Customers")
    fig.update_yaxes(range=[0, tiers.max() * 1.18])
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(f"Tiers: high ≥ {cut['high']:.2f}, medium ≥ {cut['medium']:.2f} (model operating "
               "threshold), low below.")
with c2:
    fig = go.Figure(go.Histogram(
        x=latest["churn_probability"], xbins=dict(start=0, end=1, size=0.05),
        marker=dict(color=SERIES[0], line=dict(color="#FFFFFF", width=2)),
        hovertemplate="score %{x}<br>%{y:,} customers<extra></extra>"))
    for name, x, y in (("operating threshold", cut["medium"], 1.0), ("high tier", cut["high"], 0.88)):
        fig.add_vline(x=x, line=dict(color="#475569", width=1, dash="dot"))
        fig.add_annotation(x=x, y=y, yref="paper", text=name, showarrow=False, xanchor="left",
                           yanchor="top", xshift=4, font=dict(size=11, color="#475569"),
                           bgcolor="rgba(255,255,255,0.85)")
    layout(fig, title="Distribution of estimated churn risk", xtitle="Estimated churn probability",
           ytitle="Customers")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

c3, c4 = st.columns([1.35, 1])
with c3:
    min_n = config.settings()["segments"]["min_cohort_size"]
    seg = (latest.groupby(["Contract", "InternetService"])
           .agg(customers=("customer_id", "size"), avg_risk=("churn_probability", "mean"),
                revenue_at_risk=("revenue_at_risk", "sum"))
           .reset_index())
    seg = seg[seg["customers"] >= min_n].sort_values("revenue_at_risk", ascending=False).head(6)
    seg["segment"] = seg["Contract"] + " · " + seg["InternetService"].replace({"No": "No internet"})
    seg = seg.iloc[::-1]
    fig = go.Figure(go.Bar(
        y=seg["segment"], x=seg["revenue_at_risk"], orientation="h",
        marker=dict(color=SERIES[0], cornerradius=4),
        customdata=seg[["customers", "avg_risk"]].values,
        text=[common.money(v) for v in seg["revenue_at_risk"]], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>Revenue at risk %{x:$,.0f}<br>%{customdata[0]:,} customers · "
                      "avg risk %{customdata[1]:.0%}<extra></extra>"))
    layout(fig, title="Top segments by estimated revenue at risk", xtitle="Annualised revenue at risk (est.)")
    fig.update_xaxes(range=[0, seg["revenue_at_risk"].max() * 1.3])
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(f"Segments with fewer than {min_n} customers are hidden.")
with c4:
    st.markdown("<div class='card'><h4>Action pipeline</h4>", unsafe_allow_html=True)
    if actions.empty:
        common.empty_state("No actions yet", "Recommendations appear in the risk work queue.")
    else:
        counts = actions["status"].value_counts()
        for status in ["recommended", "approved", "in_progress", "completed", "rejected"]:
            st.markdown(f"<div class='kv'><div>{status.replace('_', ' ').title()}</div>"
                        f"<div style='text-align:right'><b>{int(counts.get(status, 0)):,}</b></div></div>",
                        unsafe_allow_html=True)
        top = actions["recommendation_name"].value_counts().head(4)
        st.markdown("<div class='muted' style='margin-top:10px'>Most recommended</div>", unsafe_allow_html=True)
        for name, n in top.items():
            st.markdown(f"<div class='kv'><div>{name}</div><div style='text-align:right'>{n}</div></div>",
                        unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    if st.button("Open risk work queue →", use_container_width=True, type="primary"):
        st.switch_page("app_pages/risk_queue.py")

with st.expander("Metric definitions"):
    st.markdown(f"""
- **Customers scored** - distinct customers with at least one stored score; the latest score per customer is used.
- **Revenue at risk (est.)** - Σ estimated churn probability × `MonthlyCharges` × {horizon}. A prioritisation
  estimate; it is not a forecast of lost revenue and assumes no intervention.
- **Observed churn** - share of churners among customers whose outcome label has arrived (delayed labels).
- **Data-quality warnings** - records accepted with warnings (e.g. blank `TotalCharges`, inconsistent totals).
- Model quality (PR-AUC/ROC-AUC) is shown in the Model lab and is **not** evidence of retention impact.
""")
