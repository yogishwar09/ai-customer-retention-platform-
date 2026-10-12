"""RetentionIQ - Streamlit entry point.

    streamlit run app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (ROOT / "src", ROOT):  # works without `pip install -e .` (e.g. Streamlit Cloud)
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="RetentionIQ", page_icon="📉", layout="wide", initial_sidebar_state="expanded")

from ui import common  # noqa: E402

common.inject_css()
common.ensure_demo_data()
common.start_worker()

if common.principal() is None:
    common.login_view()
    st.stop()


def page(path: str, title: str, icon: str, default: bool = False):
    return st.Page(path, title=title, icon=icon, default=default)


pages = {
    "Overview": [page("app_pages/overview.py", "Executive overview", ":material/dashboard:", default=True),
                 page("app_pages/assistant.py", "Ask the data", ":material/forum:")],
    "Operate": [
        page("app_pages/risk_queue.py", "Risk work queue", ":material/format_list_numbered:"),
        page("app_pages/customer.py", "Customer detail", ":material/person_search:"),
        page("app_pages/score.py", "Score a customer", ":material/calculate:"),
        page("app_pages/batch.py", "Batch scoring", ":material/upload_file:"),
    ],
    "Analyze": [
        page("app_pages/segments.py", "Segments & cohorts", ":material/donut_small:"),
        page("app_pages/model_lab.py", "Model lab", ":material/science:"),
        page("app_pages/experiments.py", "Experiments & uplift", ":material/experiment:"),
    ],
    "Govern": [
        page("app_pages/monitoring.py", "Monitoring", ":material/monitor_heart:"),
        page("app_pages/governance.py", "Governance", ":material/verified_user:"),
        page("app_pages/operations.py", "Operations", ":material/schedule:"),
    ],
}
admin_pages = []
if common.can("manage_integrations"):
    admin_pages.append(page("app_pages/integrations.py", "Integrations", ":material/hub:"))
if common.can("manage_users"):
    admin_pages.append(page("app_pages/admin.py", "Users & API keys", ":material/admin_panel_settings:"))
if admin_pages:
    pages["Admin"] = admin_pages

nav = st.navigation(pages)
common.sidebar()
nav.run()
