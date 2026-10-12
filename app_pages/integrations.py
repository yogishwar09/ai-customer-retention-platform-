import json

import pandas as pd
import streamlit as st

from retentioniq import config
from retentioniq.integrations.webhooks import MAX_ATTEMPTS, WebhookError, Webhooks, dispatch_due, summary
from retentioniq.storage.engine import get_engine
from retentioniq.storage.repository import EVENT_TYPES
from ui import common

db = common.db()
hooks = Webhooks(db)
common.header("Integrations", "Signed, retried, idempotent webhooks to your CRM or helpdesk, plus scheduled file imports.")
common.require("manage_integrations")

s = summary(db)
common.kpi_row([
    common.kpi("Active webhooks", f"{int((hooks.list()['active'] == 1).sum()) if not hooks.list().empty else 0}"),
    common.kpi("Delivered", f"{s.get('succeeded', 0):,}"),
    common.kpi("Pending / retrying", f"{s.get('pending', 0):,}", "delivered by the worker every 15 s"),
    common.kpi("Dead-lettered", f"{s.get('dead', 0):,}", f"after {MAX_ATTEMPTS} attempts or a 4xx"),
])

tab1, tab2, tab3, tab4 = st.tabs(["Webhooks", "Delivery log", "Sandbox CRM inbox", "Scheduled imports"])
with tab1:
    df = hooks.list()
    if df.empty:
        common.empty_state("No webhooks yet", "Create one below. Use <code>sandbox://crm</code> to try it safely.")
    else:
        st.dataframe(df, hide_index=True, use_container_width=True,
                     column_config={"active": st.column_config.CheckboxColumn("Active")})
        c1, c2, c3 = st.columns([2, 1, 1])
        pick = c1.selectbox("Webhook", df["webhook_id"].tolist(),
                            format_func=lambda i: df.set_index("webhook_id").loc[i, "name"])
        active = bool(df.set_index("webhook_id").loc[pick, "active"])
        if c2.button("Disable" if active else "Enable"):
            hooks.set_active(pick, not active, common.actor())
            st.rerun()
        if c3.button("Send test event"):
            ev = hooks.send_test(common.actor())
            st.toast("Test event queued" if ev else "No active webhook subscribes to alert.raised")

    with st.form("new_webhook", border=True):
        st.markdown("**New webhook**")
        name = st.text_input("Name", placeholder="Salesforce retention queue")
        url = st.text_input("Endpoint URL", placeholder="https://hooks.example.com/retentioniq  or  sandbox://crm")
        events = st.multiselect("Events", EVENT_TYPES, default=["customer.high_risk", "action.approved"])
        if st.form_submit_button("Create webhook", type="primary"):
            try:
                wid, secret = hooks.create(name.strip() or "Webhook", url.strip(), events, common.actor())
            except WebhookError as exc:
                st.error(str(exc))
            else:
                st.success(f"Created `{wid}`.")
                st.code(secret, language=None)
                st.warning("Copy the signing secret now - it is not shown again.")
    with st.expander("How receivers verify requests"):
        st.markdown("""
Each POST carries:
- `X-RetentionIQ-Event` - event type, `X-RetentionIQ-Delivery` - delivery id
- `Idempotency-Key` - the event id; **store it and ignore repeats** (delivery is at-least-once)
- `X-RetentionIQ-Signature: t=<unix>,v1=<hex>` where `v1 = HMAC-SHA256(secret, f"{t}.{body}")`

Reject signatures older than 5 minutes. Return 2xx quickly; 5xx/timeouts are retried with exponential backoff
(30 s, 60 s, 120 s …), 400/401/403/404/410 are not. Non-public addresses are blocked (SSRF protection).
""")
        st.code('''import hmac, hashlib, time
def verify(secret: str, body: bytes, header: str) -> bool:
    parts = dict(p.split("=", 1) for p in header.split(","))
    if abs(time.time() - int(parts["t"])) > 300:
        return False
    mac = hmac.new(secret.encode(), f"{parts['t']}.".encode() + body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(mac, parts["v1"])''', language="python")

with tab2:
    d = hooks.deliveries(500)
    if d.empty:
        common.empty_state("No deliveries yet", "Events appear here when subscribed activity happens.")
    else:
        status = st.multiselect("Status", sorted(d["status"].unique()), default=[])
        view = d[d["status"].isin(status)] if status else d
        st.dataframe(view, hide_index=True, use_container_width=True, height=380)
        dead = d[d["status"] == "dead"]
        c1, c2 = st.columns([2, 1])
        if not dead.empty:
            target = c1.selectbox("Replay a dead-lettered delivery", dead["delivery_id"].tolist())
            if c2.button("Replay"):
                hooks.replay(target, common.actor())
                st.rerun()
        if st.button("Dispatch due deliveries now"):
            st.write(dispatch_due(get_engine(), org_id=db.org_id))

with tab3:
    inbox = hooks.sandbox_inbox()
    common.note("The sandbox CRM is an in-process receiver for <code>sandbox://</code> URLs. It verifies the HMAC "
                "signature and de-duplicates on the idempotency key exactly as a production receiver should - "
                "<code>delivery_count &gt; 1</code> shows retries being absorbed safely. Set "
                "<code>RETENTIONIQ_SANDBOX_FAILURE_RATE=0.3</code> to rehearse retries and dead-lettering.")
    if inbox.empty:
        common.empty_state("Inbox empty", "Create a sandbox webhook and trigger an event.")
    else:
        inbox["summary"] = inbox["payload"].map(lambda p: json.dumps(json.loads(p)["data"])[:140])
        st.dataframe(inbox[["received_at", "event_type", "signature_valid", "delivery_count", "idempotency_key",
                            "summary"]], hide_index=True, use_container_width=True, height=380,
                     column_config={"signature_valid": st.column_config.CheckboxColumn("Signature OK")})

with tab4:
    from retentioniq.storage.accounts import Accounts
    org = Accounts().get_org(db.org_id)
    folder = config.PROJECT_ROOT / config.settings()["jobs"]["inbox_dir"] / org["slug"]
    st.markdown(f"Drop CSV files into **`{folder.relative_to(config.PROJECT_ROOT)}/`**. The worker checks every "
                f"{config.settings()['scheduler']['inbox_import']} s, stores each file, queues an async scoring job "
                "and moves the file to `processed/`. Results, error reports and audit events appear in Operations.")
    processed = sorted((folder / "processed").glob("*.csv"), reverse=True) if (folder / "processed").exists() else []
    if processed:
        st.dataframe(pd.DataFrame({"processed file": [p.name for p in processed[:50]]}), hide_index=True,
                     use_container_width=True)
    if st.button("Copy the sample file into the inbox (demo)"):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "sample_import.csv").write_bytes((config.SAMPLES_DIR / "batch_clean.csv").read_bytes())
        st.success("Queued for the next import tick.")
