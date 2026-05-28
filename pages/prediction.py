import streamlit as st
from utils.predict import predict_churn

# =====================================================
# PAGE TITLE
# =====================================================
st.title("🔮 Customer Churn Prediction")

st.write("Predict whether a customer is likely to churn using AI-powered ML model.")

# =====================================================
# INPUT LAYOUT
# =====================================================
col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox("Gender", ["Male", "Female"])

    senior = st.selectbox("Senior Citizen", [0, 1])

    partner = st.selectbox("Partner", ["Yes", "No"])

    dependents = st.selectbox("Dependents", ["Yes", "No"])

    tenure = st.slider("Tenure (months)", 0, 72, 12)

    monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)

with col2:

    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
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

    total = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

# =====================================================
# PREDICT BUTTON
# =====================================================
if st.button("🚀 Predict Churn"):

    # Input dictionary (must match training columns conceptually)
    input_data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
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
    }

    # Prediction
    prediction, probability = predict_churn(input_data)

    churn_percent = round(probability * 100, 2)
    confidence = round(max(probability, 1 - probability) * 100, 2)

    # =================================================
    # RESULTS UI
    # =================================================
    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Churn Probability", f"{churn_percent}%")

    with col2:
        st.metric("Model Confidence", f"{confidence}%")

    st.write("")

    # Risk classification
    if churn_percent >= 70:
        st.error("⚠ High Risk: Customer likely to churn")

    elif churn_percent >= 40:
        st.warning("⚠ Medium Risk: Monitor customer")

    else:
        st.success("✅ Low Risk: Customer is stable")