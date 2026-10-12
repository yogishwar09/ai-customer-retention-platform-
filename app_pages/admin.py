import pandas as pd
import streamlit as st

from retentioniq.auth.passwords import generate_password
from retentioniq.auth.rbac import PERMISSIONS, ROLES
from ui import common

db = common.db()
acc = common.accounts()
me = common.current()
common.header("Administration", "Users, roles and API keys for your organisation.")
common.require("manage_users")

tab1, tab2, tab3 = st.tabs(["Users", "API keys", "Roles & permissions"])
with tab1:
    users = pd.DataFrame(acc.list_users(me.org_id))
    st.dataframe(users.drop(columns=["user_id"]), hide_index=True, use_container_width=True,
                 column_config={"active": st.column_config.CheckboxColumn()})
    c1, c2 = st.columns(2)
    with c1.form("invite", border=True):
        st.markdown("**Add user**")
        email = st.text_input("Email")
        role = st.selectbox("Role", ROLES, index=1)
        if st.form_submit_button("Create user", type="primary"):
            temp = generate_password()
            try:
                uid = acc.create_user(me.org_id, email, temp, role)
            except Exception as exc:
                st.error(str(exc))
            else:
                db.audit(me.actor, "user.created", "user", uid, {"email": email.lower(), "role": role})
                st.success("User created. Share this one-time password through a secure channel:")
                st.code(temp, language=None)
    with c2.container(border=True):
        st.markdown("**Update user**")
        if not users.empty:
            pick = st.selectbox("User", users["user_id"].tolist(),
                                format_func=lambda u: users.set_index("user_id").loc[u, "email"])
            row = users.set_index("user_id").loc[pick]
            new_role = st.selectbox("Role ", ROLES, index=ROLES.index(row["role"]))
            active = st.checkbox("Active", value=bool(row["active"]))
            b1, b2 = st.columns(2)
            if b1.button("Save changes"):
                if pick == me.user_id and (new_role != "admin" or not active):
                    st.error("You can't remove your own admin access.")
                else:
                    acc.update_user(me.org_id, pick, role=new_role, active=active)
                    db.audit(me.actor, "user.updated", "user", pick, {"role": new_role, "active": active})
                    st.rerun()
            if b2.button("Reset password"):
                temp = generate_password()
                acc.update_user(me.org_id, pick, password=temp)
                db.audit(me.actor, "user.password_reset", "user", pick)
                st.code(temp, language=None)
                st.caption("One-time password - share securely. Lockouts are cleared.")

with tab2:
    keys = pd.DataFrame(acc.list_api_keys(me.org_id))
    if keys.empty:
        common.empty_state("No API keys", "")
    else:
        st.dataframe(keys.drop(columns=["key_id"]), hide_index=True, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1.form("newkey", border=True):
        st.markdown("**Create API key**")
        name = st.text_input("Name", placeholder="crm-sync")
        role = st.selectbox("Role", ROLES, index=1, key="keyrole")
        if st.form_submit_button("Create key", type="primary"):
            kid, token = acc.create_api_key(me.org_id, name or "api-key", role, me.actor)
            db.audit(me.actor, "api_key.created", "api_key", kid, {"name": name, "role": role})
            st.success("Copy the key now - only its hash is stored.")
            st.code(token, language=None)
    with c2.container(border=True):
        st.markdown("**Revoke API key**")
        active = keys[keys["revoked_at"].isna()] if not keys.empty else keys
        if active.empty:
            st.caption("No active keys.")
        else:
            kid = st.selectbox("Key", active["key_id"].tolist(),
                               format_func=lambda k: f"{active.set_index('key_id').loc[k, 'name']} "
                                                     f"(riq_{active.set_index('key_id').loc[k, 'prefix']}_…)")
            if st.button("Revoke"):
                acc.revoke_api_key(me.org_id, kid)
                db.audit(me.actor, "api_key.revoked", "api_key", kid)
                st.rerun()
    st.caption("Use keys as `Authorization: Bearer riq_...`. Requests are scoped to this organisation and the key's "
               "role, rate-limited, and logged with the key name.")

with tab3:
    rank = {r: i for i, r in enumerate(ROLES)}
    matrix = pd.DataFrame([{"permission": p.replace("_", " "),
                            **{r: "✓" if rank[r] >= rank[m] else "" for r in ROLES}}
                           for p, m in PERMISSIONS.items()])
    st.dataframe(matrix, hide_index=True, use_container_width=True)
    st.caption("Permissions are enforced server-side in the API and repository layer; hidden buttons are only a "
               "convenience.")
