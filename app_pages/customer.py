import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from retentioniq.data.schema import FEATURE_COLUMNS, FIELD_BY_NAME
from retentioniq.decisioning import policy, workflow
from retentioniq.storage.repository import OUTCOME_TYPES
from ui import common
from ui.theme import SERIES, layout

svc = common.service()
db = common.db()
common.header("Customer detail", "Risk, drivers, recommended playbook and action history for one customer.")

ids = db.customer_ids()
if not ids:
    common.empty_state("No stored customers", "Score customers with <b>Batch scoring</b> first.")
    st.stop()

default = st.session_state.get("customer_id")
cid = st.selectbox("Customer", ids, index=ids.index(default) if default in ids else 0)
st.session_state["customer_id"] = cid

history = db.customer_history(cid)
latest = db.get_prediction(history.iloc[-1]["prediction_id"])
features = latest["features"]
X = pd.DataFrame([features])[FEATURE_COLUMNS]
drivers = [d.as_dict() for d in svc.explainer.drivers(X, top_n=6)[0]]
stability = svc.explainer.stability(X)[0]
current_p = float(svc.model.pipeline.predict_proba(X)[0, 1])
rec = policy.recommend(features, latest["churn_probability"], latest["risk_tier"],
                       quality_status=latest["data_quality_status"],
                       borderline=bool(latest["borderline"]), customer_id=cid)

left, mid, right = st.columns([1, 1.25, 1.25])
with left:
    st.markdown(
        f"<div class='card'><div class='muted'>Estimated churn risk</div>"
        f"<div class='big-score'>{latest['churn_probability']:.0%}</div>"
        f"<div style='margin:8px 0'>{common.risk_badge(latest['risk_tier'])}</div>"
        f"<div class='muted'>Scored {latest['scored_at'][:16].replace('T', ' ')} UTC<br>"
        f"Model <code>{html.escape(latest['model_version'])}</code><br>"
        f"Data quality: {common.status_chip('ok' if latest['data_quality_status'] == 'passed' else 'warning')}"
        f"{' · borderline score' if latest['borderline'] else ''}</div></div>",
        unsafe_allow_html=True)
    if latest["model_version"] != svc.model.version:
        common.note(f"Production model is now <code>{svc.model.version}</code>; it estimates "
                    f"<b>{current_p:.0%}</b> for this profile. Rescore to store a new score.", "warn")
    for w in latest["data_quality_warnings"]:
        common.note(f"⚠ {html.escape(w)}", "warn")
    def fmt(c, v):
        if v is None:
            return "not recorded"
        if c == "tenure":
            return f"{int(v)} months"
        if c in ("MonthlyCharges", "TotalCharges"):
            return f"${v:,.2f}"
        return str(v)

    profile = "".join(
        f"<div><span>{FIELD_BY_NAME[c].label}</span></div><div>{html.escape(fmt(c, features[c]))}</div>"
        for c in ["tenure", "Contract", "MonthlyCharges", "TotalCharges", "InternetService",
                  "PaymentMethod", "TechSupport", "OnlineSecurity", "PaperlessBilling"])
    st.markdown(f"<div class='card'><h4>Profile</h4><div class='kv'>{profile}</div></div>",
                unsafe_allow_html=True)

with mid:
    stab = ("stable under small input changes" if stability["stable"]
            else "unstable - treat the ranking with caution")
    st.markdown(f"<div class='card'><h4>Why the model scores this customer this way</h4>"
                f"{common.driver_rows(drivers)}"
                f"<div class='muted' style='margin-top:10px'>Method: {html.escape(svc.explainer.method)}. "
                f"Top-3 driver stability {stability['score']:.2f} ({stab}). Contributions explain the "
                f"model's output, not what causes churn.</div></div>", unsafe_allow_html=True)
    if len(history) > 1:
        fig = go.Figure(go.Scatter(x=pd.to_datetime(history["scored_at"]), y=history["churn_probability"],
                                   mode="lines+markers", line=dict(color=SERIES[0], width=2),
                                   marker=dict(size=8), hovertemplate="%{x|%d %b %H:%M}<br>%{y:.1%}<extra></extra>"))
        layout(fig, title="Score history", height=220, ytitle="Est. churn risk")
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.caption("One score stored so far - history appears after rescoring.")

with right:
    a = rec.assumptions
    alt = "".join(f"<div><span>{html.escape(x['name'])}</span></div><div>EV ${x['expected_value']:,.0f}</div>"
                  for x in rec.eligible_alternatives[:3]) or "<div><span>None</span></div><div></div>"
    approval = ("<b style='color:#B7791F'>◆ Manager approval required</b>" if rec.requires_approval
                else "No approval required")
    from retentioniq.experiments.registry import Experiments
    exp_id, exp_arm = Experiments(db).assignment(rec.code, cid)
    arm = ("No running experiment covers this action" if exp_id is None else
           "<b>Control group</b> - will be held out (not contacted) to measure impact"
           if exp_arm == "control" else "Treatment group of a running experiment")
    st.markdown(
        f"<div class='card'><h4>Recommended next-best action</h4>"
        f"<div style='font-size:1.05rem;font-weight:700;color:#0F172A'>{html.escape(rec.name)}</div>"
        f"<div class='muted' style='margin:4px 0 8px'>{html.escape(rec.code)} · {html.escape(rec.channel)} · "
        f"policy {html.escape(rec.policy_version)}</div>"
        f"<div style='font-size:.88rem'>{html.escape(rec.description)}</div>"
        f"<div class='muted' style='margin-top:8px'>{html.escape(rec.rationale)}</div>"
        f"<div style='margin-top:10px;font-size:.86rem'>{approval}<br>{arm}</div>"
        f"<div class='kv' style='margin-top:10px'>"
        f"<div><span>Expected value (est.)</span></div><div><b>${rec.expected_value:,.2f}</b></div>"
        f"<div><span>p(churn)</span></div><div>{a['churn_probability']:.2f}</div>"
        f"<div><span>Annual value</span></div><div>${a['annual_value']:,.0f}</div>"
        f"<div><span>Assumed effectiveness</span></div><div>{a['assumed_effectiveness']:.0%} (assumption)</div>"
        f"<div><span>Action cost</span></div><div>${a['action_cost']:,.0f}</div></div>"
        f"<div class='muted' style='margin-top:10px'>Other eligible actions</div><div class='kv'>{alt}</div>"
        f"</div>", unsafe_allow_html=True)

st.subheader("Action timeline")
actions = db.actions(cid)
c1, c2 = st.columns([1, 2])
with c1:
    with st.container(border=True):
        st.markdown("**Log recommended action**")
        note = st.text_input("Note (optional)", max_chars=300, key="new_action_note")
        disabled = not common.can("agent") or rec.code == "MONITOR"
        if st.button("Create action from recommendation", disabled=disabled, type="primary",
                     help="Agent role required" if not common.can("agent") else None):
            act = workflow.create_action(db, latest, common.actor(), note, recommendation=rec.as_dict())
            st.toast("Logged - customer is in the control group and will not be contacted"
                     if act["arm"] == "control" else "Action logged")
            st.rerun()
with c2:
    if actions.empty:
        common.empty_state("No actions yet", "Log the recommendation to start tracking it.")
    for row in actions.to_dict(orient="records"):
        with st.container(border=True):
            top = st.columns([3, 1.2])
            top[0].markdown(f"**{row['recommendation_name']}** · `{row['status']}` · arm `{row['experiment_arm']}`  \n"
                            f"<span class='muted'>{row['created_at'][:16].replace('T', ' ')} by {row['owner']}"
                            f"{' · ' + html.escape(row['note']) if row['note'] else ''}</span>",
                            unsafe_allow_html=True)
            if row.get("outcome_type"):
                top[1].markdown(f"Outcome: **{row['outcome_type']}**")
            key = row["action_id"]
            btns = st.columns(4)
            if row["status"] == "recommended":
                needs = bool(row["requires_approval"])
                ok = common.can("manager") if needs else common.can("agent")
                if btns[0].button("Approve" if needs else "Start", key=f"ap{key}", disabled=not ok,
                                  help=None if ok else "Manager approval required"):
                    db.update_action_status(key, "approved" if needs else "in_progress", common.actor())
                    st.rerun()
                if btns[1].button("Reject", key=f"rj{key}", disabled=not common.can("manager")):
                    db.update_action_status(key, "rejected", common.actor())
                    st.rerun()
            if row["status"] == "held_out":
                btns[0].markdown("<span class='pill'>control · not contacted</span>", unsafe_allow_html=True)
            if row["status"] in ("approved", "in_progress", "recommended", "held_out") and not row.get("outcome_type"):
                outcome = btns[2].selectbox("Outcome", OUTCOME_TYPES, key=f"oc{key}", label_visibility="collapsed")
                blocked = row["status"] == "recommended" and bool(row["requires_approval"])
                if btns[3].button("Record outcome", key=f"ro{key}",
                                  disabled=not common.can("agent") or blocked,
                                  help="Approve the action first" if blocked else None):
                    db.record_outcome(key, outcome, common.actor(), window_days=30)
                    st.rerun()

if st.button("Rescore with current production model"):
    if common.can("agent"):
        svc.score(pd.DataFrame([{"customerID": cid, **features}]), source="rescore", repo=db, persist=True)
        db.audit(common.actor(), "prediction.rescored", "customer", cid)
        st.rerun()
    else:
        st.warning("Agent role required.")
