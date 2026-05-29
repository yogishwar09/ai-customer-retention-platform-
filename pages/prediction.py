import streamlit as st
import pandas as pd
from utils.model_loader import get_model

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Churn Predictor", layout="wide")

st.title("📊 Customer Churn Prediction System")
st.markdown("Predict customer churn using trained ML model")

# =========================
# LOAD MODEL (SAFE)
# =========================
model = get_model()

# =========================
# INPUT FORM
# =========================
with st.form("churn_form"):

    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)

    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox("Multiple Lines", ["Yes", "No"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    security = st.selectbox("Online Security", ["Yes", "No"])
    backup = st.selectbox("Online Backup", ["Yes", "No"])
    device = st.selectbox("Device Protection", ["Yes", "No"])
    tech = st.selectbox("Tech Support", ["Yes", "No"])

    tv = st.selectbox("Streaming TV", ["Yes", "No"])
    movies = st.selectbox("Streaming Movies", ["Yes", "No"])

    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    payment = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    )

    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)
    total = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=100.0)

    submitted = st.form_submit_button("Predict Churn")

# =========================
# PREDICTION LOGIC
# =========================
if submitted:

    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaymentMethod": payment,
        "PaperlessBilling": paperless,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")

        st.metric("Churn Probability", f"{probability * 100:.2f}%")

        if prediction == 1:
            st.error("🔴 HIGH RISK: Customer likely to churn")
        else:
            st.success("🟢 LOW RISK: Customer will stay")

    except Exception as e:
        st.error("Prediction failed due to model or input mismatch")
        st.code(str(e))