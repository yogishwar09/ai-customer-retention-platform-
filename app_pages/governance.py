import json
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from retentioniq import config
from retentioniq.data.schema import SCHEMA_VERSION, data_dictionary
from retentioniq.decisioning.risk import tier_cutoffs
from retentioniq.jobs.queue import JobQueue
from retentioniq.models import registry
from retentioniq.monitoring.summary import shadow_summary
from ui import common
from ui.theme import SERIES, layout

svc = common.service()
db = common.db()
common.header("Governance", "Model lifecycle and approvals, model cards, policy and thresholds, data contract, audit trail.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Model lifecycle", "Model card", "Policy & thresholds",
                                        "Data contract", "Audit log"])
with tab1:
    reg = registry.read_registry()
    models = pd.DataFrame(reg["models"]).sort_values("registered_at", ascending=False)
    status_order = {"production": 0, "staging": 1, "candidate": 2, "archived": 3, "rejected": 4}
    models["_o"] = models["status"].map(status_order)
    show = models.sort_values(["_o", "registered_at"], ascending=[True, False])
    st.dataframe(show[["version", "status", "algorithm", "test_pr_auc", "test_roc_auc", "brier", "threshold",
                       "gate_passed", "feedback_rows", "registered_at"]] if "feedback_rows" in show else
                 show.drop(columns=["_o", "history"], errors="ignore"),
                 hide_index=True, use_container_width=True,
                 column_config={"gate_passed": st.column_config.CheckboxColumn("Gate"),
                                "test_pr_auc": st.column_config.NumberColumn("Test PR-AUC", format="%.3f"),
                                "test_roc_auc": st.column_config.NumberColumn("Test ROC-AUC", format="%.3f"),
                                "brier": st.column_config.NumberColumn(format="%.3f"),
                                "feedback_rows": st.column_config.NumberColumn("Feedback rows")})
    common.note("Lifecycle: <b>candidate</b> → <b>staging</b> (scored in shadow next to production) → "
                "<b>production</b> (requires an approval reason). Only models that passed the release gate can be "
                "staged or promoted. Promoting an archived model is the rollback path.")

    c1, c2 = st.columns([1.3, 1])
    with c1.container(border=True):
        st.markdown("**Change model status**")
        target = st.selectbox("Model version", show["version"].tolist(), key="gov_target")
        entry = next(m for m in reg["models"] if m["version"] == target)
        new_status = st.selectbox("New status", ["staging", "production", "archived", "candidate"], key="gov_status")
        reason = st.text_input("Approval reason / change note", max_chars=300,
                               placeholder="e.g. Challenger matched champion PR-AUC on 2 weeks of shadow scores")
        allowed = common.can("manage_models")
        if st.button("Apply change", type="primary", disabled=not allowed,
                     help=None if allowed else "Admin role required"):
            try:
                registry.set_status(target, new_status, common.actor(), reason)
            except (registry.PromotionError, registry.ModelNotAvailable) as exc:
                st.error(str(exc))
            else:
                db.audit(common.actor(), f"model.{new_status}", "model", target, {"reason": reason})
                if new_status == "production":
                    db.emit("model.promoted", {"version": target, "by": common.actor(), "reason": reason})
                st.cache_resource.clear()
                st.success(f"{target} → {new_status}")
                st.rerun()
        if entry.get("history"):
            st.markdown("<div class='muted'>History</div>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(entry["history"]).iloc[::-1], hide_index=True, use_container_width=True)
    with c2.container(border=True):
        st.markdown("**Retrain**")
        st.caption("Queues a training job on the worker: reproduces the pipeline, optionally adds this "
                   "organisation's outcome labels (excluding any that overlap the held-out test set), and "
                   "registers the result as a candidate. It never auto-promotes.")
        tune = st.checkbox("Hyperparameter search (slower)")
        feedback = st.checkbox("Include this organisation's feedback labels", value=True)
        if st.button("Queue retraining job", disabled=not common.can("manage_models")):
            jid = JobQueue().enqueue("retrain_model", {"tune": tune, "use_feedback": feedback,
                                                       "actor": common.actor()},
                                     org_id=db.org_id, created_by=common.actor(), max_attempts=1)
            db.audit(common.actor(), "model.retrain_requested", "job", jid, {"tune": tune, "feedback": feedback})
            st.success(f"Queued `{jid}` - follow it in Operations.")

    st.subheader("Champion vs challenger (shadow scoring)")
    sh = shadow_summary(db, db.labels()[["customer_id", "churned"]])
    if sh["status"] != "ok":
        common.empty_state("No challenger running",
                           "Move a gate-passing candidate to <b>staging</b>; new scores will be computed by both "
                           "models (only the champion's score is used for decisions).")
    else:
        common.kpi_row([
            common.kpi("Shadow-scored", f"{sh['n']:,}", sh["challenger_version"]),
            common.kpi("Mean |difference|", f"{sh['mean_abs_difference']:.3f}", "probability points"),
            common.kpi("Correlation", f"{sh['correlation']:.3f}" if sh.get("correlation") is not None else "—"),
            common.kpi("PR-AUC champion / challenger",
                       f"{sh['champion_pr_auc']:.3f} / {sh['challenger_pr_auc']:.3f}" if "champion_pr_auc" in sh
                       else "—", f"on {sh['labelled']:,} labelled" if "labelled" in sh else "needs ≥ 50 labels"),
        ])
        comp = db.shadow_comparison().tail(2000)
        fig = go.Figure(go.Scatter(x=comp["champion"], y=comp["challenger"], mode="markers",
                                   marker=dict(size=8, color=SERIES[0], opacity=0.5,
                                               line=dict(color="#FFFFFF", width=1)),
                                   hovertemplate="champion %{x:.2f}<br>challenger %{y:.2f}<extra></extra>"))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#C3C2B7", dash="dot", width=1),
                        hoverinfo="skip")
        layout(fig, height=320, xtitle="Champion probability", ytitle="Challenger probability")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with st.expander("Production model metadata"):
        st.json(svc.model.metadata)

with tab2:
    versions = show["version"].tolist()
    pick = st.selectbox("Model card for", versions, index=versions.index(svc.model.version)
                        if svc.model.version in versions else 0)
    card = config.MODELS_DIR / pick / "model_card.md"
    report = config.MODELS_DIR / pick / "evaluation_report.md"
    if not card.exists():
        card, report = config.REPORTS_DIR / "model_card.md", config.REPORTS_DIR / "evaluation_report.md"
    if card.exists():
        text = re.sub(r"^(#+) ", lambda mt: "#" * min(len(mt.group(1)) + 2, 6) + " ",
                      card.read_text(encoding="utf-8"), flags=re.M)
        st.markdown(text)
        c1, c2 = st.columns(2)
        c1.download_button("Download model card (.md)", card.read_bytes(), f"model_card_{pick}.md")
        if report.exists():
            c2.download_button("Download evaluation report (.md)", report.read_bytes(), f"evaluation_{pick}.md")
    else:
        common.empty_state("No model card", "Run <code>make train</code> to generate it.")

with tab3:
    pb = config.playbooks()
    cut = tier_cutoffs(svc.model.threshold)
    st.markdown(f"**Policy version** `{pb.get('policy_version')}` · **risk tiers** high ≥ {cut['high']:.2f}, "
                f"medium ≥ {cut['medium']:.2f} (operating threshold)")
    acts = pd.DataFrame([{
        "code": a["code"], "name": a["name"], "channel": a["channel"], "cost": a.get("cost", 0),
        "assumed effectiveness": a.get("assumed_effectiveness", 0.0),
        "requires approval": a.get("requires_approval", False), "min tier": a.get("min_risk_tier", "—"),
        "conditions": "; ".join(f"{c['feature']} {c['op']} {c['value']}" for c in a.get("all_of", []))
                      + (" AND any of: " + ", ".join(f"{c['feature']} {c['op']} {c['value']}"
                                                     for c in a["any_of"]) if a.get("any_of") else ""),
    } for a in pb["actions"]])
    st.dataframe(acts, hide_index=True, use_container_width=True)
    st.caption("Rules live in configs/playbooks.yaml and are validated against the data contract in CI. "
               "Effectiveness values are assumptions until measured in Experiments.")
    st.markdown("**Runtime settings**")
    st.code(json.dumps(config.settings(), indent=2), language="json")

with tab4:
    st.markdown(f"Schema version **{SCHEMA_VERSION}**. Every channel (UI, batch, API, jobs, training) validates "
                "through the same contract.")
    st.dataframe(data_dictionary(), hide_index=True, use_container_width=True)

with tab5:
    log = db.audit_log(2000)
    if log.empty:
        common.empty_state("No audit events", "")
    else:
        f1, f2 = st.columns(2)
        types = f1.multiselect("Event types", sorted(log["event_type"].unique()))
        actors = f2.multiselect("Actors", sorted(log["actor"].unique()))
        if types:
            log = log[log["event_type"].isin(types)]
        if actors:
            log = log[log["actor"].isin(actors)]
        st.dataframe(log.drop(columns=["org_id"]), hide_index=True, use_container_width=True, height=420)
        if common.can("export"):
            st.download_button("Export audit log (CSV)", log.to_csv(index=False).encode(), "audit_log.csv", "text/csv")
        st.caption("Audit events store actor, action and target ids only - not customer feature rows. "
                   "Scoped to your organisation.")
    if common.can("reset_demo"):
        with st.expander("Admin: reset demo data"):
            st.warning("Reseeds every demo organisation: deletes stored predictions, actions, outcomes, experiments, "
                       "webhooks and audit events, regenerates demo users' passwords and the demo API key.")
            if st.checkbox("I understand") and st.button("Reset and reseed"):
                from scripts.seed_demo import main as seed
                st.cache_resource.clear()
                seed(reset=True)
                common.logout("Demo data was reset. New sign-in details are in artifacts/demo_credentials.txt.")
                st.rerun()
