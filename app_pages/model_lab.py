import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from retentioniq.models.evaluate import expected_cost, threshold_metrics
from ui import common
from ui.theme import INK_2, SERIES, layout

svc = common.service()
meta, ev = svc.model.metadata, svc.model.evaluation
t = ev["test"]
common.header("Model lab", f"Held-out evaluation of {svc.model.version}. The test split was never used "
                           "for model selection or threshold tuning.")

split = meta["split"]
common.kpi_row([
    common.kpi("Test ROC-AUC", f"{t['roc_auc']:.3f}", f"95% CI {t['ci_95']['roc_auc'][0]:.3f}-{t['ci_95']['roc_auc'][1]:.3f}"),
    common.kpi("Test PR-AUC", f"{t['pr_auc']:.3f}",
               f"CI {t['ci_95']['pr_auc'][0]:.3f}-{t['ci_95']['pr_auc'][1]:.3f} · random = {t['prevalence']:.3f}"),
    common.kpi("Brier score", f"{t['brier']:.3f}", "lower is better (calibration + sharpness)"),
    common.kpi("Operating threshold", f"{meta['operating_threshold']:.2f}",
               f"min cost, FN:FP = {meta['cost_assumptions']['false_negative']:g}:{meta['cost_assumptions']['false_positive']:g}"),
    common.kpi("Test set", f"{split['test']:,}", f"prevalence {t['prevalence']:.1%} · {split['method']}"),
])

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Threshold trade-offs", "Curves & calibration", "Lift & capacity",
                                        "Model comparison", "Fairness & importance"])
y = np.array(ev["test_predictions"]["y"])
p = np.array(ev["test_predictions"]["p"])

with tab1:
    c1, c2 = st.columns([1, 1.6])
    thr = c1.slider("Threshold", 0.05, 0.95, float(meta["operating_threshold"]), 0.01,
                    help="Explore only - the production threshold is set by the training pipeline.")
    tm = threshold_metrics(y, p, thr)
    cfp, cfn = meta["cost_assumptions"]["false_positive"], meta["cost_assumptions"]["false_negative"]
    c1.markdown(f"""
<div class='card'><div class='kv'>
<div><span>Precision</span></div><div><b>{tm['precision']:.3f}</b></div>
<div><span>Recall</span></div><div><b>{tm['recall']:.3f}</b></div>
<div><span>F1</span></div><div>{tm['f1']:.3f}</div>
<div><span>Accuracy</span></div><div>{tm['accuracy']:.3f}</div>
<div><span>Flagged for outreach</span></div><div>{tm['flagged_share']:.1%} ({tm['tp'] + tm['fp']:,})</div>
<div><span>Expected cost / customer</span></div><div>{expected_cost(y, p, thr, cfp, cfn):.3f}</div>
</div></div>""", unsafe_allow_html=True)
    cm = np.array([[tm["tn"], tm["fp"]], [tm["fn"], tm["tp"]]])
    fig = go.Figure(go.Heatmap(z=cm, x=["Predicted: stay", "Predicted: churn"], y=["Actual: stay", "Actual: churn"],
                               colorscale=[[0, "#EAF2FC"], [1, "#256ABF"]], showscale=False, xgap=2, ygap=2,
                               text=cm, texttemplate="<b>%{text}</b>", textfont=dict(size=16),
                               hovertemplate="%{y} / %{x}: %{z}<extra></extra>"))
    layout(fig, height=230, title="Confusion matrix (test)")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    c1.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    rows = [threshold_metrics(y, p, g) for g in grid]
    fig = go.Figure()
    for i, (key, name) in enumerate([("precision", "Precision"), ("recall", "Recall"),
                                     ("flagged_share", "Share flagged")]):
        fig.add_scatter(x=grid, y=[r[key] for r in rows], mode="lines", name=name,
                        line=dict(color=SERIES[i], width=2, dash="dot" if key == "flagged_share" else None),
                        hovertemplate=f"{name} %{{y:.2f}} at %{{x:.2f}}<extra></extra>")
    fig.add_vline(x=thr, line=dict(color=INK_2, width=1))
    fig.add_vline(x=meta["operating_threshold"], line=dict(color=INK_2, width=1, dash="dot"))
    fig.add_annotation(x=meta["operating_threshold"], y=1.04, yref="paper", text="production", showarrow=False,
                       font=dict(size=11, color=INK_2))
    layout(fig, height=420, legend=True, title="Precision, recall and outreach volume by threshold",
           xtitle="Threshold", ytitle="Rate")
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(range=[0, 1.02])
    c2.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with tab2:
    c1, c2, c3 = st.columns(3)
    roc = ev["curves"]["roc"]
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#C3C2B7", dash="dot", width=1),
                    hoverinfo="skip", name="random")
    fig.add_scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", line=dict(color=SERIES[0], width=2),
                    name="model", hovertemplate="FPR %{x:.2f} · TPR %{y:.2f}<extra></extra>")
    layout(fig, height=320, title=f"ROC curve · AUC {t['roc_auc']:.3f}", xtitle="False positive rate",
           ytitle="True positive rate")
    c1.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    pr = ev["curves"]["pr"]
    fig = go.Figure()
    fig.add_hline(y=t["prevalence"], line=dict(color="#C3C2B7", dash="dot", width=1))
    fig.add_annotation(x=1, y=t["prevalence"], text="random (prevalence)", showarrow=False, xanchor="right",
                       yanchor="bottom", font=dict(size=11, color=INK_2))
    fig.add_scatter(x=pr["recall"], y=pr["precision"], mode="lines", line=dict(color=SERIES[0], width=2),
                    hovertemplate="Recall %{x:.2f} · Precision %{y:.2f}<extra></extra>")
    layout(fig, height=320, title=f"Precision-recall · AP {t['pr_auc']:.3f}", xtitle="Recall", ytitle="Precision")
    fig.update_yaxes(range=[0, 1.02])
    c2.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    cal = pd.DataFrame(ev["calibration"])
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#C3C2B7", dash="dot", width=1),
                    hoverinfo="skip")
    fig.add_scatter(x=cal["mean_predicted"], y=cal["observed_rate"], mode="lines+markers",
                    line=dict(color=SERIES[0], width=2), marker=dict(size=8, line=dict(color="#fff", width=2)),
                    customdata=cal["count"],
                    hovertemplate="Predicted %{x:.2f} → observed %{y:.2f}<br>n = %{customdata}<extra></extra>")
    layout(fig, height=320, title=f"Calibration (deciles) · Brier {t['brier']:.3f}",
           xtitle="Mean predicted probability", ytitle="Observed churn rate")
    c3.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("Points on the diagonal mean predicted probabilities match observed frequencies on the test set.")

with tab3:
    lift = pd.DataFrame(ev["lift"])
    c1, c2 = st.columns([1.4, 1])
    fig = go.Figure(go.Bar(x=lift["decile"], y=lift["lift"], marker=dict(color=SERIES[0], cornerradius=4),
                           text=[f"{v:.1f}×" for v in lift["lift"]], textposition="outside",
                           customdata=lift[["churn_rate", "cumulative_capture"]].values,
                           hovertemplate="Decile %{x}<br>Lift %{y:.2f}×<br>Churn rate %{customdata[0]:.1%}"
                                         "<br>Cumulative churners captured %{customdata[1]:.0%}<extra></extra>"))
    fig.add_hline(y=1, line=dict(color="#C3C2B7", dash="dot", width=1))
    layout(fig, height=340, title="Lift by risk decile (1 = highest risk)", xtitle="Decile", ytitle="Lift vs. random")
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(range=[0, lift["lift"].max() * 1.2])
    c1.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    c2.markdown("**Precision at top-K** (outreach capacity)")
    pk = pd.DataFrame([{"K (customers contacted)": int(k), "share who actually churned": v}
                       for k, v in t["precision_at_k"].items()])
    c2.dataframe(pk, hide_index=True, use_container_width=True,
                 column_config={"share who actually churned": st.column_config.ProgressColumn(
                     format="%.2f", min_value=0, max_value=1)})
    top2 = lift.iloc[1]["cumulative_capture"]
    c2.markdown(f"Contacting the top 20% riskiest test customers would reach **{top2:.0%}** of actual churners.")

with tab4:
    comp = pd.DataFrame(ev["comparison"])
    st.dataframe(comp, hide_index=True, use_container_width=True, column_config={
        "cv_pr_auc_mean": st.column_config.NumberColumn("CV PR-AUC (train)", format="%.3f"),
        "cv_pr_auc_std": st.column_config.NumberColumn("± sd", format="%.3f"),
        "val_pr_auc": st.column_config.NumberColumn("Val PR-AUC", format="%.3f"),
        "val_roc_auc": st.column_config.NumberColumn("Val ROC-AUC", format="%.3f"),
        "val_brier": st.column_config.NumberColumn("Val Brier", format="%.3f"),
        "selected": st.column_config.CheckboxColumn("Selected")})
    st.caption("Selection rule: highest validation PR-AUC (rounded to 3 dp), ties broken by lower Brier score. "
               "Differences within the CV standard deviation are not meaningful - the transparent logistic "
               "baseline performs comparably here.")

with tab5:
    c1, c2 = st.columns([1.2, 1])
    sub = pd.DataFrame(ev["subgroups"])
    c1.markdown("**Subgroup performance (test set)**")
    c1.dataframe(sub, hide_index=True, use_container_width=True,
                 column_config={k: st.column_config.NumberColumn(format="%.3f")
                                for k in ["prevalence", "roc_auc", "recall", "precision", "flagged_share"]})
    if "recall" in sub:
        gaps = sub.groupby("attribute")["recall"].agg(lambda s: s.max() - s.min())
        for attr, gap in gaps.items():
            kind = "warn" if gap > 0.10 else "ok"
            common.note(f"Recall gap across <b>{attr}</b>: {gap:.3f} "
                        f"({'review before use' if gap > 0.10 else 'within 0.10 tolerance'}).", kind)
    gi = pd.DataFrame(ev["global_importance"]).head(12).iloc[::-1]
    fig = go.Figure(go.Bar(y=gi["label"], x=gi["mean_abs_contribution"], orientation="h",
                           marker=dict(color=SERIES[0], cornerradius=4),
                           hovertemplate="%{y}<br>mean |contribution| %{x:.3f}<extra></extra>"))
    layout(fig, height=420, title="Global importance (mean |contribution|, test sample)", xtitle="Mean |contribution|")
    c2.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    c2.caption(meta["explanation_method"] + ". Importance reflects the model, not causal effect.")
