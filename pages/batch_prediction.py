import streamlit as st
import pandas as pd
import os
from utils.model_loader import get_model

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Batch Churn Prediction", layout="wide")

st.title("📦 Batch Churn Prediction System")
st.markdown("Upload a CSV file to predict customer churn in bulk")

# =========================
# LOAD MODEL (SAFE)
# =========================
model = get_model()

# =========================
# FILE UPLOAD
# =========================
file = st.file_uploader("Upload CSV file", type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    st.subheader("📄 Input Data Preview")
    st.dataframe(df.head())

    try:
        # =========================
        # SAFE PREDICTION
        # =========================

        # Keep only columns model expects (prevents crash)
        expected_features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else df.columns

        # Align dataset with training features
        df_model = df.reindex(columns=expected_features, fill_value=0)

        # Predict
        probs = model.predict_proba(df_model)[:, 1]
        preds = (probs >= 0.5).astype(int)

        # Add results
        df["Churn_Probability"] = probs
        df["Churn_Prediction"] = preds

        # =========================
        # OUTPUT
        # =========================
        st.subheader("📊 Prediction Results")
        st.dataframe(df)

        # Download CSV
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇ Download Results",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error("❌ Batch prediction failed")

        st.markdown("### Debug Info")
        st.code(str(e))

        st.info("""
        Fix checklist:
        - CSV must contain correct feature columns
        - No missing required columns
        - Same format as training dataset
        """)