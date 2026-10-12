import html

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from retentioniq.data.schema import FIELD_BY_NAME, INTERNET_ADDON_COLUMNS, NO_INTERNET, NO_PHONE
from ui import common
from ui.theme import SERIES, layout

svc = common.service()
db = common.db()
common.header("Score a customer",
              "Validated single-record scoring with drivers, recommendation and what-if sensitivity.")


def pick(col, name, index=0):
    f = FIELD_BY_NAME[name]
    return col.selectbox(f.label, list(f.allowed), index=index, key=f"in_{name}")


with st.container(border=True):
    st.markdown("**Account**")
    a1, a2, a3, a4 = st.columns(4)
    cid = a1.text_input("Customer ID (optional)", max_chars=64, placeholder="e.g. NEW-0001")
    tenure = a2.number_input("Tenure (months)", 0, 120, 4)
    contract = pick(a3, "Contract")
    payment = pick(a4, "PaymentMethod")
    b1, b2, b3, b4 = st.columns(4)
    monthly = b1.number_input("Monthly charges ($)", 0.0, 1000.0, 79.5, step=0.5)
    total_default = float(round(monthly * tenure, 2))
    total_blank = b2.checkbox("Total charges unknown", value=False)
    total = None if total_blank else b2.number_input("Total charges ($)", 0.0, 200_000.0, total_default)
    paperless = pick(b3, "PaperlessBilling")
    st.markdown("**Demographics**")
    d1, d2, d3, d4 = st.columns(4)
    gender = pick(d1, "gender")
    senior = d2.selectbox("Senior citizen", [0, 1], format_func=lambda v: "Yes" if v else "No")
    partner = pick(d3, "Partner", 1)
    dependents = pick(d4, "Dependents", 1)
    st.markdown("**Services**")
    s1, s2, s3 = st.columns(3)
    phone = pick(s1, "PhoneService")
    multiple = (s2.selectbox("Multiple lines", ["Yes", "No"], index=1) if phone == "Yes"
                else s2.selectbox("Multiple lines", [NO_PHONE], disabled=True))
    internet = s3.selectbox("Internet service", list(FIELD_BY_NAME["InternetService"].allowed), index=1)
    addon_cols = st.columns(6)
    addons = {}
    for col, name in zip(addon_cols, INTERNET_ADDON_COLUMNS, strict=True):
        if internet == "No":
            addons[name] = col.selectbox(FIELD_BY_NAME[name].label, [NO_INTERNET], disabled=True)
        else:
            addons[name] = col.selectbox(FIELD_BY_NAME[name].label, ["No", "Yes"])
    persist = st.toggle("Save this score to the customer record", value=bool(cid),
                        disabled=not cid or not common.can("agent"),
                        help="Requires a customer ID and the agent role.")
    go_btn = st.button("Score customer", type="primary")

record = {"customerID": cid or None, "gender": gender, "SeniorCitizen": senior, "Partner": partner,
          "Dependents": dependents, "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple,
          "InternetService": internet, **addons, "Contract": contract, "PaperlessBilling": paperless,
          "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total}
if not cid:
    record.pop("customerID")

if go_btn:
    st.session_state["last_score_input"] = record
    batch = svc.score(record, source="ui_single", repo=db, persist=persist and bool(cid))
    st.session_state["last_score"] = batch
    if persist and cid and batch.records:
        db.audit(common.actor(), "prediction.created", "customer", cid)

batch = st.session_state.get("last_score")
if batch is None:
    st.stop()
if not batch.records:
    st.error("The record failed validation:")
    for err in batch.validation.rejected["errors"].iloc[0].split("; "):
        st.markdown(f"- {html.escape(err)}")
    st.stop()

r = batch.records[0]
rec = r["recommendation"]
st.divider()
c1, c2, c3 = st.columns([1, 1.3, 1.3])
with c1:
    st.markdown(
        f"<div class='card'><div class='muted'>Estimated churn risk</div>"
        f"<div class='big-score'>{r['churn_probability']:.0%}</div>"
        f"<div style='margin:8px 0'>{common.risk_badge(r['risk_tier'])}</div>"
        f"<div class='muted'>{'Above' if r['above_operating_threshold'] else 'Below'} operating threshold "
        f"{svc.model.threshold:.2f}{' (borderline)' if r['borderline'] else ''}<br>"
        f"Model <code>{html.escape(r['model_version'])}</code><br>"
        f"Latency {batch.latency_ms:.0f} ms · "
        f"{common.status_chip('ok' if r['data_quality']['status'] == 'passed' else 'warning')}</div></div>",
        unsafe_allow_html=True)
    for w in r["data_quality"]["warnings"]:
        common.note(f"⚠ {html.escape(w)}", "warn")
with c2:
    st.markdown(f"<div class='card'><h4>Top drivers</h4>{common.driver_rows(r['explanations'])}"
                f"<div class='muted' style='margin-top:8px'>{html.escape(r['explanation_method'])}. "
                "Not causal.</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown(
        f"<div class='card'><h4>Next-best action</h4><b>{html.escape(rec['name'])}</b>"
        f"<div class='muted' style='margin:4px 0 8px'>{rec['code']} · EV ${rec['expected_value']:,.2f} (est.)"
        f"{' · needs approval' if rec['requires_approval'] else ''}</div>"
        f"<div style='font-size:.88rem'>{html.escape(rec['description'])}</div>"
        f"<div class='muted' style='margin-top:8px'>{html.escape(rec['rationale'])}</div></div>",
        unsafe_allow_html=True)

st.subheader("What-if: model sensitivity")
common.note("Shows how the <b>model's estimate</b> changes when one input changes and everything else "
            "stays fixed. It is not a prediction of what would happen if the customer actually changed.")
base = st.session_state["last_score_input"]
w1, w2 = st.columns([1, 3])
feature = w1.radio("Vary", ["Contract", "tenure", "MonthlyCharges", "PaymentMethod", "TechSupport"],
                   format_func=lambda f: FIELD_BY_NAME[f].label)
if feature in ("tenure", "MonthlyCharges"):
    values = list(range(0, 73, 3)) if feature == "tenure" else list(np.round(np.arange(20, 121, 5), 2))
    base_wi = dict(base, TotalCharges=None)
    curve = svc.what_if(base_wi, feature, values)
    fig = go.Figure(go.Scatter(x=curve[feature], y=curve["churn_probability"], mode="lines",
                               line=dict(color=SERIES[0], width=2),
                               hovertemplate=f"{FIELD_BY_NAME[feature].label} %{{x}}<br>%{{y:.1%}}<extra></extra>"))
    fig.add_vline(x=base[feature], line=dict(color="#475569", dash="dot", width=1))
    fig.add_annotation(x=base[feature], y=1, yref="paper", text="current", showarrow=False,
                       xanchor="left", xshift=4, font=dict(size=11, color="#475569"))
    w2.caption("TotalCharges is left blank (imputed) while varying this input to keep the profile consistent.")
else:
    if feature == "TechSupport" and base["InternetService"] == "No":
        w2.info("Tech support is not applicable without internet service.")
        st.stop()
    options = [v for v in FIELD_BY_NAME[feature].allowed if v != NO_INTERNET]
    curve = svc.what_if(base, feature, options)
    fig = go.Figure(go.Bar(x=curve[feature], y=curve["churn_probability"],
                           marker=dict(color=[SERIES[1] if v == base[feature] else SERIES[0]
                                              for v in curve[feature]], cornerradius=4),
                           text=[f"{p:.0%}" for p in curve["churn_probability"]], textposition="outside",
                           hovertemplate="%{x}<br>%{y:.1%}<extra></extra>"))
    w2.caption("Orange = the customer's current value.")
layout(fig, height=300, ytitle="Est. churn risk", xtitle=FIELD_BY_NAME[feature].label)
fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
w2.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with st.expander("Raw response (same contract as POST /v1/predictions)"):
    st.json({k: v for k, v in r.items() if k not in ("features",)})
