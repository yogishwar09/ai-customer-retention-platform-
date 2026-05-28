import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Customer Churn AI Platform",
    layout="wide"
)

st.title("📊 Customer Churn Prediction & Analytics System")
st.markdown("AI-powered SaaS platform for predicting and analyzing telecom customer churn")

# =====================================================
# LOAD MODEL + PREPROCESSOR (SAFE)
# =====================================================
MODEL_PATH = os.path.join("model", "churn_model.pkl")
PREPROCESSOR_PATH = os.path.join("model", "preprocessor.pkl")

@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found")
        st.stop()

    if not os.path.exists(PREPROCESSOR_PATH):
        st.error("❌ Preprocessor file not found")
        st.stop()

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return model, preprocessor

model, preprocessor = load_assets()

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🔮 Single Prediction", "📦 Batch Prediction", "📊 System Info"]
)

# =====================================================
# HOME PAGE
# =====================================================
if page == "🏠 Home":

    st.subheader("Welcome to AI Churn Intelligence Platform 🚀")

    st.markdown("""
    This system helps telecom businesses:

    ✔ Predict customer churn using Machine Learning  
    ✔ Analyze customer behavior patterns  
    ✔ Identify high-risk customers early  
    ✔ Improve customer retention strategy  
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Model", "XGBoost")
    col2.metric("Pipeline", "Preprocess + ML")
    col3.metric("Status", "Active")

    st.info("Use sidebar to navigate between prediction tools and system info.")

# =====================================================
# SINGLE PREDICTION (UPGRADED UI)
# =====================================================
elif page == "🔮 Single Prediction":

    st.subheader("🔮 Customer Churn Prediction")

    st.markdown("Enter customer details to estimate churn probability")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input("📅 Tenure (months)", 0, 100, 12)
        monthly_charges = st.number_input("💰 Monthly Charges", 0.0, 200.0, 70.0)

    with col2:
        total_charges = st.number_input("🧾 Total Charges", 0.0, 10000.0, 1000.0)

    input_df = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    st.divider()

    if st.button("🚀 Predict Churn", use_container_width=True):

        try:
            processed = preprocessor.transform(input_df)

            prediction = model.predict(processed)[0]
            probability = model.predict_proba(processed)[0][1]

            st.subheader("📊 Prediction Result")

            col1, col2 = st.columns(2)

            col1.metric("Churn Probability", f"{probability:.2f}")
            col2.metric("Prediction", "Churn ❌" if prediction == 1 else "No Churn ✅")

            if probability > 0.7:
                st.error("⚠️ High Risk Customer - Immediate attention required")
            elif probability > 0.4:
                st.warning("⚠️ Medium Risk Customer")
            else:
                st.success("✅ Low Risk Customer")

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")

# =====================================================
# BATCH PREDICTION INFO PAGE
# =====================================================
elif page == "📦 Batch Prediction":

    st.subheader("📦 Batch Prediction System")

    st.markdown("""
    ### How to use:
    1. Go to batch prediction page  
    2. Upload CSV file with customer data  
    3. Run prediction  
    4. Download results  

    👉 Full batch processing available in `pages/batch_prediction.py`
    """)

    st.info("Upload multiple customers for bulk churn prediction")

# =====================================================
# SYSTEM INFO PAGE
# =====================================================
elif page == "📊 System Info":

    st.subheader("📊 Model Information")

    st.markdown("""
    ✔ Model Type: XGBoost Classifier  
    ✔ Preprocessing: ColumnTransformer Pipeline  
    ✔ Handling: SMOTE for imbalance  
    ✔ Deployment: Streamlit Cloud Ready  
    """)

    st.markdown("### Features Used")

    st.markdown("""
    - Tenure  
    - Monthly Charges  
    - Total Charges  
    - Contract Type  
    - Internet Services  
    - Customer Demographics  
    """)

    st.success("System is fully operational 🚀")