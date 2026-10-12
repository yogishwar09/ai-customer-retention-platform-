import pandas as pd
import streamlit as st

from retentioniq import config
from retentioniq.decisioning import policy
from ui import common

svc = common.service()
db = common.db()
common.header("Risk work queue",
              "Prioritised customers with reason codes and an advisory next-best action.")

latest = db.latest_scores()
if latest.empty:
    common.empty_state("Queue is empty", "Score customers via <b>Batch scoring</b> first.")
    st.stop()

horizon = config.settings()["value"]["horizon_months"]
capacity = config.settings()["decision"]["outreach_capacity"]
latest["annual_value"] = latest["MonthlyCharges"] * horizon
latest["revenue_at_risk"] = latest["churn_probability"] * latest["annual_value"]

recs = []
for row in latest.to_dict(orient="records"):
    feats = {k: row[k] for k in svc.model.metadata["feature_columns"]}
    r = policy.recommend(feats, row["churn_probability"], row["risk_tier"],
                         quality_status=row["data_quality_status"], borderline=bool(row["borderline"]),
                         customer_id=row["customer_id"])
    recs.append((r.code, r.name, r.expected_value, r.requires_approval, r.channel))
latest[["action_code", "action_name", "expected_value", "requires_approval", "channel"]] = pd.DataFrame(
    recs, index=latest.index)
delta = config.settings()["decision"]["risk_rising_delta"]

actions = db.actions()
if not actions.empty:
    last_action = actions.sort_values("created_at").groupby("customer_id").tail(1)
    latest = latest.merge(last_action[["customer_id", "status", "experiment_arm"]].rename(
        columns={"status": "action_status", "experiment_arm": "arm"}), on="customer_id", how="left")
else:
    latest["action_status"], latest["arm"] = None, None
latest["arm"] = latest["arm"].fillna("—")
latest["action_status"] = latest["action_status"].fillna("none")

with st.container(border=True):
    f1, f2, f3, f4, f5 = st.columns([1.1, 1.2, 1.2, 1.1, 1.3])
    tiers = f1.multiselect("Risk tier", ["high", "medium", "low"], default=["high", "medium"],
                           format_func=str.title)
    contracts = f2.multiselect("Contract", sorted(latest["Contract"].unique()))
    action_codes = f3.multiselect("Recommended action", sorted(latest["action_code"].unique()))
    status = f4.multiselect("Action status", sorted(latest["action_status"].unique()))
    sort_by = f5.selectbox("Prioritise by", ["Expected value of action", "Churn probability",
                                             "Revenue at risk", "Customer value"])
    s1, s2 = st.columns([2, 1])
    search = s1.text_input("Search customer ID", placeholder="e.g. 7590-VHVEG")
    t1, t2 = s2.columns(2)
    rising_only = t1.toggle("Rising risk only", value=False, help=f"Score rose ≥ {delta:.2f} since the previous scoring")
    hide_control = t2.toggle("Hide control group", value=False,
                             help="Control-group customers are held out from outreach to measure impact.")

q = latest
if tiers:
    q = q[q["risk_tier"].isin(tiers)]
if contracts:
    q = q[q["Contract"].isin(contracts)]
if action_codes:
    q = q[q["action_code"].isin(action_codes)]
if status:
    q = q[q["action_status"].isin(status)]
if search:
    q = q[q["customer_id"].str.contains(search.strip(), case=False, regex=False)]
if hide_control:
    q = q[q["arm"] != "control"]
if rising_only:
    q = q[q["risk_change"] >= delta]
sort_col = {"Expected value of action": "expected_value", "Churn probability": "churn_probability",
            "Revenue at risk": "revenue_at_risk", "Customer value": "annual_value"}[sort_by]
q = q.sort_values(sort_col, ascending=False).reset_index(drop=True)
q.insert(0, "priority", range(1, len(q) + 1))

common.kpi_row([
    common.kpi("Customers in view", f"{len(q):,}", f"of {len(latest):,} scored"),
    common.kpi("Within outreach capacity", f"{min(len(q), capacity):,}", f"capacity {capacity} per cycle"),
    common.kpi("Revenue at risk in view (est.)", common.money(q["revenue_at_risk"].sum())),
    common.kpi("Expected value, top capacity (est.)",
               common.money(q.head(capacity)["expected_value"].clip(lower=0).sum()),
               "uses assumed action effectiveness"),
])

view = q[["priority", "customer_id", "churn_probability", "risk_change", "risk_tier", "top_reasons", "action_name",
          "expected_value", "requires_approval", "action_status", "arm", "Contract", "tenure",
          "MonthlyCharges", "revenue_at_risk", "data_quality_status", "scored_at"]]
event = st.dataframe(
    view, hide_index=True, use_container_width=True, height=470,
    on_select="rerun", selection_mode="single-row",
    column_config={
        "priority": st.column_config.NumberColumn("#", width="small"),
        "customer_id": "Customer",
        "churn_probability": st.column_config.ProgressColumn("Est. churn risk", format="%.2f",
                                                             min_value=0, max_value=1),
        "risk_change": st.column_config.NumberColumn("Δ vs prev", format="%+.2f",
                                                     help="Change since the previous score"),
        "risk_tier": st.column_config.TextColumn("Tier"),
        "top_reasons": st.column_config.TextColumn("Top risk drivers", width="large"),
        "action_name": st.column_config.TextColumn("Next-best action", width="medium"),
        "expected_value": st.column_config.NumberColumn("EV (est.)", format="$%.0f"),
        "requires_approval": st.column_config.CheckboxColumn("Needs approval"),
        "action_status": "Action status",
        "arm": "Arm",
        "tenure": st.column_config.NumberColumn("Tenure", format="%d mo"),
        "MonthlyCharges": st.column_config.NumberColumn("Monthly", format="$%.2f"),
        "revenue_at_risk": st.column_config.NumberColumn("Rev. at risk (est.)", format="$%.0f"),
        "data_quality_status": "Data quality",
        "scored_at": st.column_config.TextColumn("Scored at"),
    })

selected = event.selection.rows if event and event.selection else []
b1, b2, b3 = st.columns([1.2, 1.4, 3])
if selected:
    cid = view.iloc[selected[0]]["customer_id"]
    if b1.button(f"Open {cid} →", type="primary"):
        common.open_customer(cid)
else:
    b1.button("Open customer →", disabled=True, help="Select a row first")

if common.can("export"):
    from retentioniq.integrations.crm_export import TEMPLATES
    from retentioniq.integrations.crm_export import export as crm_export
    export = q.copy()
    export["model_version"] = svc.model.version
    export["policy_version"] = config.playbooks().get("policy_version")
    template = b3.selectbox("Export format", list(TEMPLATES), label_visibility="collapsed",
                            format_func=lambda t: {"generic": "Generic CSV", "salesforce_task": "Salesforce tasks",
                                                   "hubspot_ticket": "HubSpot tickets"}[t])
    out = crm_export(export, template)
    b2.download_button("Export queue (CSV)", out.to_csv(index=False).encode(), f"risk_queue_{template}.csv",
                       "text/csv", on_click=lambda: db.audit(common.actor(), "queue.exported",
                                                             "risk_queue", None, {"rows": len(out),
                                                                                  "template": template}))
else:
    b2.button("Export queue (CSV)", disabled=True, help="Manager role required")

st.caption("Expected value ≈ p(churn) × annual value × assumed effectiveness − action cost. "
           "Effectiveness values are assumptions in configs/playbooks.yaml until measured by experiment. "
           "Recommendations are advisory; actions marked 'Needs approval' require a manager.")
