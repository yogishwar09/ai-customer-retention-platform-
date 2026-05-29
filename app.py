import streamlit as st
import os

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="AI Churn SaaS Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# LOAD CSS SAFELY
# =====================================================
def load_css():
    css_path = "styles/style.css"
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# =====================================================
# SAFE LOGO HANDLER
# =====================================================
def show_logo():
    logo_path = "assets/logo.png"
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, width=120)
    else:
        st.sidebar.markdown("### 📊 Churn AI Platform")

show_logo()

# =====================================================
# SIDEBAR NAVIGATION (NO REDIRECTS)
# =====================================================
st.sidebar.title("Navigation")

menu = st.sidebar.radio(
    "Go to",
    ["Home", "Dashboard", "Single Prediction", "Batch Prediction"]
)

st.sidebar.markdown("---")
st.sidebar.info("AI-powered churn prediction system")

# =====================================================
# HOME PAGE
# =====================================================
if menu == "Home":
    st.markdown("""
    <div class="hero-box">
        <h1>📊 AI Customer Churn SaaS Platform</h1>
        <p>
        Enterprise-grade machine learning system for predicting customer churn,
        analyzing behavior, and improving retention strategy.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    col1.metric("System Status", "ACTIVE")
    col2.metric("Model", "READY")
    col3.metric("Platform", "STABLE")

    st.success("Select a module from the sidebar to continue.")

# =====================================================
# DASHBOARD PAGE
# =====================================================
elif menu == "Dashboard":
    try:
        import pages.dashboard
    except Exception as e:
        st.error("Dashboard failed to load")
        st.exception(e)

# =====================================================
# SINGLE PREDICTION PAGE
# =====================================================
elif menu == "Single Prediction":
    try:
        import pages.prediction
    except Exception as e:
        st.error("Prediction page failed to load")
        st.exception(e)

# =====================================================
# BATCH PREDICTION PAGE
# =====================================================
elif menu == "Batch Prediction":
    try:
        import pages.batch_prediction
    except Exception as e:
        st.error("Batch prediction page failed to load")
        st.exception(e)

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<div class="footer">
    🚀 AI SaaS Platform | Built with Streamlit + Machine Learning
</div>
""", unsafe_allow_html=True)