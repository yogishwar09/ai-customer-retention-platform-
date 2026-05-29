import streamlit as st
import os

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="AI Churn SaaS Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CSS LOADER
# =========================
def load_css():
    css_path = "styles/style.css"
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# =========================
# SIDEBAR LOGIC
# =========================
st.sidebar.title("AI Churn Platform")

menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Dashboard", "Single Prediction", "Batch Prediction"]
)

st.sidebar.markdown("---")
st.sidebar.info("ML-powered customer churn system")

# =========================
# HOME
# =========================
if menu == "Home":
    st.title("📊 AI Customer Churn SaaS Platform")

    st.markdown("""
    Enterprise ML system for:
    - Churn prediction
    - Customer analytics
    - Batch prediction
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Status", "ACTIVE")
    col2.metric("Model", "READY")
    col3.metric("Version", "v1.0")

# =========================
# DASHBOARD
# =========================
elif menu == "Dashboard":
    try:
        from pages import dashboard
    except Exception as e:
        st.error("Dashboard error")
        st.code(str(e))

# =========================
# SINGLE PREDICTION
# =========================
elif menu == "Single Prediction":
    try:
        from pages import prediction
    except Exception as e:
        st.error("Prediction error")
        st.code(str(e))

# =========================
# BATCH PREDICTION
# =========================
elif menu == "Batch Prediction":
    try:
        from pages import batch_prediction
    except Exception as e:
        st.error("Batch prediction error")
        st.code(str(e))

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("🚀 Built with Streamlit + Machine Learning")