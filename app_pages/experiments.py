import plotly.graph_objects as go
import streamlit as st

from retentioniq import config
from retentioniq.data.schema import FEATURE_COLUMNS
from retentioniq.experiments import stats, uplift
from retentioniq.experiments.registry import Experiments
from ui import common
from ui.theme import INK_2, SERIES, layout

svc = common.service()
db = common.db()
exps = Experiments(db)
common.header("Experiments & uplift", "Measure whether retention actions work - with randomised control groups, "
                                      "power analysis and uplift modelling.")
common.note("A churn model predicts <b>who</b> may leave. Only a randomised comparison shows whether an "
            "<b>action</b> changes the outcome. Customers assigned to a control arm are held out (not contacted); "
            "assignment is a stable hash of experiment and customer, so it is reproducible and unbiased.")

VERDICTS = {
    "collecting": ("◆", "#B7791F", "Collecting outcomes"),
    "underpowered": ("◆", "#B7791F", "Not enough outcomes yet - no conclusion"),
    "positive": ("▲", "#0B7A0B", "Significant improvement in retention"),
    "negative": ("▼", "#D03B3B", "Significant harm - consider stopping"),
    "no_detectable_effect": ("●", "#475569", "No detectable effect at the planned power"),
}

tab1, tab2, tab3 = st.tabs(["Experiments", "New experiment", "Uplift lab"])
with tab1:
    lst = exps.list()
    if lst.empty:
        common.empty_state("No experiments", "Create one in the next tab.")
    else:
        st.dataframe(lst[["name", "status", "action_code", "control_share", "min_detectable_effect", "started_at",
                          "created_by"]].fillna({"action_code": "all actions"}),
                     hide_index=True, use_container_width=True)
        pick = st.selectbox("Analyse", lst["experiment_id"].tolist(),
                            format_func=lambda i: lst.set_index("experiment_id").loc[i, "name"])
        a = exps.analyze(pick)
        exp, arms, test = a["experiment"], a["arms"], a["test"]
        icon, color, label = VERDICTS[a["verdict"]]
        st.markdown(f"<div class='card'><h4>{exp['name']}</h4><div class='muted'>{exp['hypothesis'] or ''}</div>"
                    f"<div style='margin-top:8px;font-weight:700;color:{color}'>{icon} {label}</div></div>",
                    unsafe_allow_html=True)
        t, c = arms["treatment"], arms["control"]
        common.kpi_row([
            common.kpi("Treatment retention", f"{test['treatment_rate']:.1%}" if test.get("status") == "ok" else "—",
                       f"{t['retained']}/{t['with_outcome']} with outcome · {t['assigned']} assigned"),
            common.kpi("Control retention", f"{test['control_rate']:.1%}" if test.get("status") == "ok" else "—",
                       f"{c['retained']}/{c['with_outcome']} with outcome · {c['assigned']} assigned"),
            common.kpi("Difference (95% CI)", f"{test['difference']:+.1%}" if test.get("status") == "ok" else "—",
                       f"{test['ci_low']:+.1%} to {test['ci_high']:+.1%} · p={test['p_value']:.3f}"
                       if test.get("status") == "ok" else ""),
            common.kpi("Sample needed", f"{a['required']['treatment']:,} / {a['required']['control']:,}",
                       f"treatment / control outcomes for MDE {exp['min_detectable_effect']:.0%}"),
            common.kpi("Achieved power", f"{a['achieved_power']:.0%}", f"target {exp['power']:.0%}"),
        ])
        if test.get("status") == "ok":
            fig = go.Figure()
            fig.add_vline(x=0, line=dict(color="#C3C2B7", width=1))
            fig.add_trace(go.Scatter(x=[test["difference"]], y=["treatment − control"], mode="markers",
                                     marker=dict(size=12, color=SERIES[0]),
                                     error_x=dict(type="data", symmetric=False,
                                                  array=[test["ci_high"] - test["difference"]],
                                                  arrayminus=[test["difference"] - test["ci_low"]],
                                                  color=SERIES[0], thickness=2, width=8),
                                     hovertemplate="Δ %{x:+.1%}<extra></extra>"))
            fig.add_vrect(x0=-exp["min_detectable_effect"], x1=exp["min_detectable_effect"],
                          fillcolor="#EEF2F7", line_width=0, layer="below")
            layout(fig, height=160, xtitle="Difference in retention rate (shaded: smaller than the MDE)")
            fig.update_xaxes(tickformat="+.0%", zeroline=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        if t["guardrail_events"] or c["guardrail_events"]:
            common.note(f"Guardrail events (opt-outs / declines): treatment {t['guardrail_events']}, "
                        f"control {c['guardrail_events']}.", "warn")
        if common.can("manage_experiments"):
            s1, s2, s3 = st.columns(3)
            if exp["status"] != "running" and s1.button("Start / resume"):
                try:
                    exps.set_status(pick, "running", common.actor())
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
            if exp["status"] == "running" and s2.button("Stop"):
                exps.set_status(pick, "stopped", common.actor())
                st.rerun()
            if exp["status"] in ("running", "stopped") and s3.button("Mark completed"):
                exps.set_status(pick, "completed", common.actor())
                st.rerun()
        st.caption("Two-sided two-proportion z-test with a Wald interval. Verdicts stay 'not enough outcomes' "
                   "until the planned sample is reached, which prevents stopping early on a lucky streak. In the "
                   "demo, nobody was actually contacted and outcomes are the customers' real historical labels, so "
                   "the honest answer is that no effect can be claimed.")

with tab2:
    if not common.can("manage_experiments"):
        st.info("Manager role required to create experiments.")
    else:
        codes = [a["code"] for a in config.playbooks()["actions"] if not a.get("special")]
        with st.form("new_exp", border=True):
            name = st.text_input("Name", placeholder="Billing support email test")
            hypothesis = st.text_area("Hypothesis", placeholder="Customers paying by electronic check retain better "
                                                                "after an autopay-assistance email.")
            c1, c2 = st.columns(2)
            scope = c1.selectbox("Applies to", ["All recommended actions", *codes])
            control = c2.slider("Control share", 0.05, 0.5, 0.2, 0.05)
            c3, c4, c5 = st.columns(3)
            mde = c3.number_input("Minimum detectable effect (abs. retention lift)", 0.01, 0.3, 0.05, 0.01)
            alpha = c4.selectbox("Significance level", [0.05, 0.01, 0.1])
            power = c5.selectbox("Power", [0.8, 0.9])
            baseline = st.slider("Expected baseline retention among targeted customers", 0.3, 0.95, 0.7, 0.05)
            n = stats.required_sample_size(baseline, mde, alpha, power)
            total = stats.unequal_allocation_total(n, control)
            st.markdown(f"**Required outcomes:** about **{total:,}** in total "
                        f"({int(total * (1 - control)):,} treatment / {int(total * control):,} control).")
            start = st.checkbox("Start immediately", value=True)
            if st.form_submit_button("Create experiment", type="primary"):
                try:
                    exps.create(name=name or "Untitled experiment", actor=common.actor(), hypothesis=hypothesis,
                                action_code=None if scope.startswith("All") else scope, control_share=control,
                                min_detectable_effect=mde, alpha=alpha, power=power, start=start)
                    st.success("Experiment created")
                except ValueError as exc:
                    st.error(str(exc))

with tab3:
    common.note("<b>Simulation lab.</b> Generates a synthetic randomised campaign over your scored portfolio with a "
                "<i>known</i> effect: larger for new month-to-month customers and fiber customers without support, "
                "and <i>negative</i> for long-tenure two-year customers ('sleeping dogs'). The portfolio is bootstrapped to "
                "20,000 synthetic customers (effects of a few points need large samples); a T-learner is trained on "
                "60% and evaluated on 40%. This validates the method; it is <b>not</b> evidence about real customers.",
                "warn")
    latest = db.latest_scores()
    real = uplift.real_outcome_data(db)
    n_t = int(real["treatment"].sum()) if not real.empty else 0
    n_c = int(len(real) - n_t) if not real.empty else 0
    st.markdown(f"Real randomised outcomes available: **{n_t}** treatment / **{n_c}** control "
                f"(a real uplift model needs ≥ {uplift.MIN_PER_ARM} per arm).")
    if latest.empty:
        common.empty_state("No scored customers", "")
    elif st.button("Run uplift simulation", type="primary"):
        with st.spinner("Simulating campaign and fitting T-learner…"):
            st.session_state["uplift_sim"] = uplift.run_simulation(latest[FEATURE_COLUMNS],
                                                                   latest["churn_probability"].to_numpy())
    sim = st.session_state.get("uplift_sim")
    if sim:
        common.kpi_row([
            common.kpi("Correlation with true effect", f"{sim['correlation_with_truth']:.2f}",
                       f"test split · {sim['n_test']:,} of {sim['population']:,} simulated"),
            common.kpi("Qini gain - uplift model", f"{sim['auuc_gain']:.1f}", "area above random targeting"),
            common.kpi("Qini gain - churn-risk ranking", f"{sim['auuc_gain_risk']:.1f}",
                       "targeting the riskiest instead"),
            common.kpi("Qini gain - oracle", f"{sim['auuc_gain_oracle']:.1f}", "true effect known"),
            common.kpi("Average effect", f"{sim['observed_ate']:+.1%}", f"simulated truth {sim['true_ate']:+.1%}"),
        ])
        fig = go.Figure()
        for name, curve, color, dash in (("Uplift model", sim["curve"], SERIES[0], None),
                                         ("Rank by churn risk", sim["risk_curve"], SERIES[1], None),
                                         ("Oracle (true effect)", sim["oracle"], SERIES[2], "dot"),
                                         ("Random targeting", sim["curve"].assign(incremental_retained=sim["curve"]["random"]),
                                          INK_2, "dash")):
            fig.add_scatter(x=curve["fraction_targeted"], y=curve["incremental_retained"], mode="lines", name=name,
                            line=dict(color=color, width=2, dash=dash),
                            hovertemplate=f"{name}<br>%{{x:.0%}} targeted → %{{y:.1f}} extra retained<extra></extra>")
        layout(fig, height=380, legend=True, title="Qini curve (simulated test split)",
               xtitle="Share of customers targeted", ytitle="Incremental customers retained")
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.dataframe(sim["segments"], hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(format="%.3f")
                                    for c in ["avg_predicted_uplift", "avg_true_uplift", "avg_base_churn"]})
        st.caption("Persuadables respond to outreach; sleeping dogs are made worse by it; lost causes and sure "
                   "things are unaffected. Targeting by churn risk alone spends effort on lost causes and can wake "
                   "sleeping dogs - which is why uplift targeting is the long-term goal once real experiment data exists.")
