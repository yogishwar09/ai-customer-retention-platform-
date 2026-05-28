import streamlit as st
import pandas as pd

from utils.predict import predict_churn

# =====================================================
# PAGE TITLE
# =====================================================

st.title("🔮 Customer Churn Prediction")

st.write(
    """
Predict customer churn probability
using AI-powered machine learning.
"""
)

# =====================================================
# INPUTS
# =====================================================

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    tenure = st.slider(
        "Tenure",
        0,
        72,
        12
    )

    monthly = st.number_input(
        "Monthly Charges",
        0.0,
        200.0,
        70.0
    )

with col2:

    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    total = st.number_input(
        "Total Charges",
        0.0,
        10000.0,
        1000.0
    )

# =====================================================
# PREDICT BUTTON
# =====================================================

if st.button("🚀 Predict Customer Churn"):

    input_data = pd.DataFrame([{

        "gender": gender,

        "SeniorCitizen": senior,

        "Partner": partner,

        "Dependents": "No",

        "tenure": tenure,

        "PhoneService": "Yes",

        "MultipleLines": "No",

        "InternetService": internet,

        "OnlineSecurity": "No",

        "OnlineBackup": "No",

        "DeviceProtection": "No",

        "TechSupport": "No",

        "StreamingTV": "No",

        "StreamingMovies": "No",

        "Contract": contract,

        "PaperlessBilling": "Yes",

        "PaymentMethod": payment,

        "MonthlyCharges": monthly,

        "TotalCharges": total
    }])

    prediction, probability = predict_churn(
        input_data
    )

    churn_percent = round(
        probability * 100,
        2
    )

    confidence = round(
        max(probability, 1 - probability) * 100,
        2
    )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Churn Probability",
            f"{churn_percent}%"
        )

    with col2:

        st.metric(
            "Model Confidence",
            f"{confidence}%"
        )

    st.write("")

    if churn_percent >= 70:

        st.error(
            "⚠ High churn risk customer detected."
        )

    elif churn_percent >= 40:

        st.warning(
            "⚠ Medium churn risk customer."
        )

    else:

        st.success(
            "✅ Low churn risk customer."
        )