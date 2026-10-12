import html

import plotly.graph_objects as go
import streamlit as st

from retentioniq import config
from retentioniq.assistant.semantic import DIMENSIONS, METRICS, ask
from ui import common
from ui.theme import SERIES, layout

db = common.db()
common.header("Analytics assistant", "Ask questions in plain English. Answers come only from approved metrics, "
                                     "scoped to your organisation.")
common.note(f"<b>How it stays safe:</b> your question is translated into a restricted query (one approved metric, "
            f"an optional dimension, simple filters). No generated SQL is executed, groups smaller than "
            f"{config.settings()['segments']['min_cohort_size']} are suppressed, and every question is audited.")

examples = ["Revenue at risk by contract", "How many high risk fiber customers?", "Rising risk customers by internet",
            "Average risk for month-to-month by tenure", "Top 3 payment methods by revenue at risk",
            "Retention rate by experiment arm"]
cols = st.columns(3)
for i, ex in enumerate(examples):
    if cols[i % 3].button(ex, use_container_width=True, key=f"ex{i}"):
        st.session_state["assistant_q"] = ex
question = st.text_input("Your question", key="assistant_q", placeholder="e.g. revenue at risk by payment method")

if question:
    out = ask(db, question, common.actor())
    if out["status"] != "ok":
        st.error(out["error"])
        with st.expander("What can I ask?"):
            st.markdown("**Metrics:** " + ", ".join(f"`{m}`" for m in METRICS) +
                        "  \n**Dimensions:** " + ", ".join(f"`{d}`" for d in DIMENSIONS))
        st.stop()
    res = out["result"]
    st.markdown(f"<div class='card'><div class='muted'>Interpreted as · {out['parser']}</div>"
                f"<b>{html.escape(out['interpretation'])}</b></div>", unsafe_allow_html=True)
    metric = out["spec"].metric
    fmt = (lambda v: "—" if v is None else common.money(v)) if metric == "revenue_at_risk" else \
        (lambda v: "—" if v is None else f"{v:.1%}") if metric in ("avg_risk", "observed_churn_rate", "retention_rate") \
        else (lambda v: "—" if v is None else f"{v:,.0f}")
    common.kpi_row([common.kpi("Overall", fmt(res["value"]), f"based on {res['n']:,} records")])
    table = res["table"]
    if not table.empty:
        dim = out["spec"].group_by
        shown = table.dropna(subset=["value"]).iloc[::-1]
        fig = go.Figure(go.Bar(y=shown[dim], x=shown["value"], orientation="h",
                               marker=dict(color=SERIES[0], cornerradius=4),
                               text=[fmt(v) for v in shown["value"]], textposition="outside", cliponaxis=False,
                               customdata=shown["n"], hovertemplate="%{y}<br>%{text}<br>n = %{customdata:,}<extra></extra>"))
        layout(fig, height=max(220, 44 * len(shown) + 60), xtitle=METRICS[metric])
        fig.update_xaxes(range=[0, shown["value"].max() * 1.25] if len(shown) else None)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.dataframe(table.assign(value=[fmt(v) if v == v else "suppressed" for v in table["value"]]),
                     hide_index=True, use_container_width=True)
        if res["suppressed"]:
            st.caption(f"{res['suppressed']} group(s) suppressed for having too few records.")

with st.expander("Recent questions in your organisation"):
    hist = db.query_df("SELECT asked_at, actor, question, parser, status FROM assistant_queries WHERE org_id = :org "
                       "ORDER BY asked_at DESC LIMIT 50")
    st.dataframe(hist, hide_index=True, use_container_width=True)
