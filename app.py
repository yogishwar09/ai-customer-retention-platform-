import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Customer Retention Platform",
    page_icon="📊",
    layout="wide"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 AI Retention System")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "📈 Analytics", "🤖 Prediction", "🧠 Explainability"]
)

# ---------------- OVERVIEW ----------------
if page == "🏠 Overview":

    st.title("AI Customer Retention Platform")
    st.markdown("### Predict customer churn using Machine Learning + Explainable AI")

    st.markdown("---")

    # KPI SECTION
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Total Customers", value="10,000")

    with col2:
        st.metric(label="Churn Rate", value="23%")

    with col3:
        st.metric(label="Model Accuracy", value="89%")

    with col4:
        st.metric(label="High-Risk Customers", value="1,240")

    st.markdown("---")

    st.subheader("📌 System Overview")
    st.write("""
    This platform uses Machine Learning to analyze customer behavior and predict churn probability.
    It helps businesses identify at-risk customers early and take proactive retention actions.
    """)

    st.success("Use the sidebar to explore analytics, predictions, and model insights.")

# ---------------- ANALYTICS ----------------
elif page == "📈 Analytics":

    st.title("📈 Analytics Dashboard")

    st.markdown("### Customer behavior insights will be displayed here")

    st.info("Next step: We will add interactive charts (Plotly) for churn distribution, trends, and feature importance.")

    # placeholder layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Distribution")
        st.write("Chart coming soon...")

    with col2:
        st.subheader("Feature Importance")
        st.write("Chart coming soon...")

# ---------------- PREDICTION ----------------
elif page == "🤖 Prediction":

    st.title("🤖 Churn Prediction Engine")

    st.markdown("### Enter customer details to predict churn probability")

    st.info("Next step: We will connect your Random Forest model here")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", 18, 100, 30)
            tenure = st.number_input("Tenure (months)", 0, 120, 12)

        with col2:
            balance = st.number_input("Account Balance", 0, 100000, 5000)
            products = st.number_input("Number of Products", 1, 10, 1)

        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        st.success("Prediction system will be integrated in next step 🚀")

# ---------------- EXPLAINABILITY ----------------
elif page == "🧠 Explainability":

    st.title("🧠 Model Explainability")

    st.markdown("### Understanding why the model makes predictions")

    st.info("Next step: Feature importance + SHAP analysis")

    st.subheader("Feature Importance")
    st.write("Placeholder for bar chart")

    st.subheader("Business Interpretation")
    st.write("""
    Explainability helps stakeholders understand which factors influence customer churn most.
    This improves trust and decision-making.
    """)