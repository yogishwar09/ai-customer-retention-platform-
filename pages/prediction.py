import streamlit as st
import pandas as pd
from utils.model_loader import get_model

st.set_page_config(page_title="Churn Prediction", layout="wide")

st.title("📊 Customer Churn Prediction")

model = get_model()

with st.form("form"):

    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure", 0, 100)

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

    monthly = st.number_input("Monthly Charges", 0.0, 200.0)
    total = st.number_input("Total Charges", 0.0, 10000.0)

    submit = st.form_submit_button("Predict")

if submit:

    if model is None:
        st.error("Model not loaded. Check deployment.")
        st.stop()

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
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        st.metric("Churn Probability", f"{prob*100:.2f}%")

        if pred == 1:
            st.error("🔴 HIGH RISK CUSTOMER")
        else:
            st.success("🟢 LOW RISK CUSTOMER")

    except Exception as e:
        st.error("Prediction error")
        st.code(str(e))