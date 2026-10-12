"""Shared Streamlit helpers: authenticated session, tenant repository, cached
service, trust panel and small components."""

from __future__ import annotations

import html
import os
import time
from pathlib import Path
from typing import Any

import streamlit as st

from retentioniq import config
from retentioniq.auth.rbac import ROLES, AuthError, Principal
from retentioniq.models import registry
from retentioniq.service import PredictionService
from retentioniq.storage.accounts import Accounts
from retentioniq.storage.repository import Repository
from ui.theme import RISK, STATUS

SESSION_IDLE_MINUTES = 60
_RANK = {r: i for i, r in enumerate(ROLES)}


# ------------------------------------------------------------- session
@st.cache_resource(show_spinner=False)
def accounts() -> Accounts:
    return Accounts()


def principal() -> Principal | None:
    p = st.session_state.get("principal")
    if p is None:
        return None
    last = st.session_state.get("last_activity", 0)
    if time.time() - last > SESSION_IDLE_MINUTES * 60:
        logout("Your session expired after inactivity. Please sign in again.")
        return None
    st.session_state["last_activity"] = time.time()
    return p


def login_view() -> None:
    inject_css()
    left, mid, right = st.columns([1, 1.2, 1])
    with mid:
        st.markdown("<div class='login-brand'>RetentionIQ</div>"
                    "<div class='login-sub'>Customer retention decision intelligence</div>", unsafe_allow_html=True)
        if msg := st.session_state.pop("login_message", None):
            st.info(msg)
        with st.form("login", border=True):
            email = st.text_input("Username or email", autocomplete="username", placeholder="admin")
            password = st.text_input("Password", type="password", autocomplete="current-password")
            submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
        if submitted:
            try:
                p = accounts().authenticate(email, password)
            except AuthError as exc:
                st.error(str(exc))
            else:
                st.session_state["principal"] = p
                st.session_state["last_activity"] = time.time()
                Repository(p.org_id).audit(p.actor, "auth.login", "user", p.user_id)
                st.rerun()
        if os.environ.get("RETENTIONIQ_ENV") != "production":
            st.markdown(
                "<div class='card'><h4>Demo accounts</h4>"
                "<div class='kv'>"
                "<div><span>Username</span></div><div><span>Access</span></div>"
                "<div><b>admin</b></div><div>Demo Telco · everything</div>"
                "<div><b>manager</b></div><div>Approvals, exports, experiments</div>"
                "<div><b>agent</b></div><div>Scoring, actions, outcomes</div>"
                "<div><b>viewer</b></div><div>Read-only</div>"
                "<div><b>northwind</b></div><div>Second organisation (admin)</div>"
                "</div><div class='muted' style='margin-top:8px'>Password for all: <b>demo123</b> · "
                "local demo only</div></div>", unsafe_allow_html=True)


def logout(message: str | None = None) -> None:
    p = st.session_state.get("principal")
    if p is not None and message is None:
        Repository(p.org_id).audit(p.actor, "auth.logout", "user", p.user_id)
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    if message:
        st.session_state["login_message"] = message


def current() -> Principal:
    p = principal()
    if p is None:
        st.stop()
    return p


def db() -> Repository:
    """The signed-in user's tenant repository (all reads/writes are org-scoped)."""
    return Repository(current().org_id)


def role() -> str:
    return current().role


def actor() -> str:
    return current().actor


def can(min_role_or_permission: str) -> bool:
    p = current()
    if min_role_or_permission in _RANK:
        return _RANK[p.role] >= _RANK[min_role_or_permission]
    return p.can(min_role_or_permission)


def require(permission: str) -> None:
    if not can(permission):
        st.error(f"Your role ({role()}) does not have access to this page.")
        st.stop()


# ------------------------------------------------------------- service
@st.cache_resource(show_spinner="Loading model…")
def _service(version: str, staging: str | None) -> PredictionService:
    return PredictionService(model=registry.load(version))


def service() -> PredictionService:
    version = registry.production_version()
    if not version or not (config.MODELS_DIR / version / "model.joblib").exists():
        with st.status("No trained model found - training one now (first start only)…", expanded=True) as status:
            from retentioniq.models.train import train
            version = train(log=st.write)["version"]
            status.update(label=f"Trained {version}", state="complete")
    try:
        return _service(version, registry.staging_version())
    except registry.ArtifactIntegrityError as exc:
        st.error(f"Model artifact failed its integrity check: {exc}")
        st.stop()


@st.cache_resource(show_spinner=False)
def start_worker() -> bool:
    if os.environ.get("RETENTIONIQ_INPROCESS_WORKER", "1") == "1":
        from retentioniq.jobs.worker import start_background_thread
        start_background_thread()
        return True
    return False


def ensure_demo_data() -> None:
    if not accounts().list_orgs():
        with st.spinner("First start: training a model and seeding the demo organisations…"):
            service()
            from scripts.seed_demo import main as seed
            seed(reset=False)


# ------------------------------------------------------------- layout
def inject_css() -> None:
    css = Path(config.PROJECT_ROOT / "styles" / "app.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def sidebar() -> None:
    p = current()
    with st.sidebar:
        st.markdown(
            f"<div class='who'><div class='who-org'>{html.escape(p.org_name)}</div>"
            f"<div class='who-user'>{html.escape(p.actor)}</div>"
            f"<span class='who-role'>{html.escape(p.role)}</span></div>", unsafe_allow_html=True)
        if st.button("Sign out", use_container_width=True):
            logout()
            st.rerun()
        trust_panel()


def trust_panel() -> None:
    svc = service()
    m = svc.model.metadata
    ts = db().scalar("SELECT MAX(scored_at) FROM predictions WHERE org_id = :org")
    t = svc.model.evaluation.get("test", {})
    gate = "passed" if m["release_gate"]["passed"] else "failed"
    rows = [
        ("Model", f"<code>{html.escape(svc.model.version)}</code>"),
        *([("Challenger", f"<code>{html.escape(svc.challenger.version)}</code>")] if svc.challenger else []),
        ("Algorithm", html.escape(m["algorithm"].replace("_", " "))),
        ("Trained", m["trained_at"][:10]),
        ("Test PR-AUC", f"{t.get('pr_auc', 0):.3f} · ROC-AUC {t.get('roc_auc', 0):.3f}"),
        ("Release gate", gate),
        ("Threshold", f"{svc.model.threshold:.2f}"),
        ("Policy", html.escape(str(config.playbooks().get("policy_version")))),
        ("Last score", (ts or "—")[:16].replace("T", " ")),
    ]
    body = "".join(f"<div><span>{k}</span>{v}</div>" for k, v in rows)
    st.markdown("<div class='trust'><div class='trust-h'>Trust panel</div>" + body +
                "<div class='trust-note'>Scores are estimated risk, not certainty. Drivers explain the model, "
                "not causes. Actions are advisory.</div></div>", unsafe_allow_html=True)


def header(title: str, subtitle: str = "") -> None:
    st.markdown(f"<div class='page-h'><h1>{html.escape(title)}</h1>"
                f"<p>{html.escape(subtitle)}</p></div>", unsafe_allow_html=True)


def risk_badge(tier: str) -> str:
    r = RISK[tier]
    return (f"<span class='badge' style='color:{r['color']};background:{r['bg']}'>"
            f"{r['icon']} {r['label']} risk</span>")


def status_chip(status: str) -> str:
    color, icon, label = STATUS.get(status, ("#475569", "•", status.replace("_", " ").title()))
    return f"<span class='chip' style='color:{color};border-color:{color}33'>{icon} {label}</span>"


def kpi(label: str, value: str, note: str = "", help_text: str = "") -> str:
    tip = f" title='{html.escape(help_text)}'" if help_text else ""
    return (f"<div class='kpi'{tip}><div class='kpi-l'>{html.escape(label)}</div>"
            f"<div class='kpi-v'>{value}</div><div class='kpi-n'>{note}</div></div>")


def kpi_row(items: list[str]) -> None:
    st.markdown(f"<div class='kpi-row'>{''.join(items)}</div>", unsafe_allow_html=True)


def note(text: str, kind: str = "info") -> None:
    st.markdown(f"<div class='note note-{kind}'>{text}</div>", unsafe_allow_html=True)


def money(v: float) -> str:
    cur = config.settings()["value"]["currency"]
    if abs(v) >= 1_000_000:
        return f"{cur}{v / 1_000_000:.2f}M"
    if abs(v) >= 10_000:
        return f"{cur}{v / 1_000:.1f}k"
    return f"{cur}{v:,.0f}"


def empty_state(title: str, body: str) -> None:
    st.markdown(f"<div class='empty'><b>{html.escape(title)}</b><br>{body}</div>", unsafe_allow_html=True)


def open_customer(customer_id: str) -> None:
    st.session_state["customer_id"] = customer_id
    st.switch_page("app_pages/customer.py")


def driver_rows(drivers: list[dict[str, Any]]) -> str:
    rows = []
    max_abs = max((abs(d["contribution"]) for d in drivers), default=1) or 1
    for d in drivers:
        up = d["direction"] == "increases_risk"
        width = 100 * abs(d["contribution"]) / max_abs
        color = "#D03B3B" if up else "#2A78D6"
        arrow = "↑ raises risk" if up else "↓ lowers risk"
        rows.append(
            f"<div class='drv'><div class='drv-t'>{html.escape(d['reason'])}"
            f"<span style='color:{color}'>{arrow}</span></div>"
            f"<div class='drv-bar'><div style='width:{width:.0f}%;background:{color}'></div></div></div>")
    return "".join(rows)
