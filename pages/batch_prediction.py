import streamlit as st
import pandas as pd
import joblib
import os

# =========================
# LOAD MODEL
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "churn_model.pkl")

model = joblib.load(MODEL_PATH)

st.title("📦 Batch Churn Prediction")

file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    st.subheader("Input Data Preview")
    st.dataframe(df.head())

    # =========================
    # VALIDATION STEP (IMPORTANT)
    # =========================
    try:
        # Predict probabilities
        probs = model.predict_proba(df)[:, 1]
        preds = (probs > 0.5).astype(int)

        df["Churn_Probability"] = probs
        df["Churn_Prediction"] = preds

        st.subheader("Results")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Predictions",
            csv,
            "churn_results.csv",
            "text/csv"
        )

    except Exception as e:
        st.error("❌ Batch prediction failed")
        st.write("Error details:", str(e))
        st.info("Make sure uploaded CSV has SAME columns as training data")