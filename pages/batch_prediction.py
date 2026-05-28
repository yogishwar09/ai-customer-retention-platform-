import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Batch Churn Prediction", layout="wide")

st.title("📦 Customer Churn Batch Prediction System")
st.markdown("Upload customer dataset to generate churn predictions using ML model")

# =====================================================
# LOAD MODEL + PREPROCESSOR (SAFE)
# =====================================================
MODEL_PATH = os.path.join("model", "churn_model.pkl")
PREPROCESSOR_PATH = os.path.join("model", "preprocessor.pkl")

@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file missing")
        st.stop()

    if not os.path.exists(PREPROCESSOR_PATH):
        st.error("Preprocessor file missing")
        st.stop()

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return model, preprocessor

model, preprocessor = load_assets()

# =====================================================
# FILE UPLOAD
# =====================================================
uploaded_file = st.file_uploader("📤 Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    # =====================================================
    # PREDICTION BUTTON
    # =====================================================
    if st.button("🚀 Run Batch Prediction"):

        try:
            data = df.copy()

            # =================================================
            # CLEANING
            # =================================================
            data = data.replace(" ", np.nan)
            data = data.fillna(0)

            # Drop ID column if present
            if "customerID" in data.columns:
                data = data.drop("customerID", axis=1)

            # =================================================
            # APPLY PREPROCESSOR (IMPORTANT FIX)
            # =================================================
            data_processed = preprocessor.transform(data)

            # =================================================
            # PREDICTIONS
            # =================================================
            predictions = model.predict(data_processed)

            probabilities = (
                model.predict_proba(data_processed)[:, 1]
                if hasattr(model, "predict_proba")
                else np.zeros(len(predictions))
            )

            # =================================================
            # RESULT DATAFRAME
            # =================================================
            result_df = df.copy()
            result_df["Churn Prediction"] = predictions
            result_df["Churn Probability"] = probabilities

            st.subheader("📌 Prediction Results")
            st.dataframe(result_df)

            # =================================================
            # DOWNLOAD CSV
            # =================================================
            csv_buffer = io.StringIO()
            result_df.to_csv(csv_buffer, index=False)

            st.download_button(
                label="📥 Download Results CSV",
                data=csv_buffer.getvalue(),
                file_name="churn_predictions.csv",
                mime="text/csv"
            )

            # =================================================
            # ANALYTICS
            # =================================================
            st.subheader("📊 Prediction Analytics")

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Customers", len(result_df))
            col2.metric("Churned", int(sum(result_df["Churn Prediction"])))
            col3.metric("Retention", int(len(result_df) - sum(result_df["Churn Prediction"])))

            fig, ax = plt.subplots()
            result_df["Churn Prediction"].value_counts().plot(kind="bar", ax=ax)
            ax.set_xticklabels(["No Churn", "Churn"], rotation=0)

            st.pyplot(fig)

            # =================================================
            # HIGH RISK CUSTOMERS
            # =================================================
            st.subheader("⚠️ High Risk Customers")

            high_risk = result_df[result_df["Churn Probability"] > 0.7]

            if len(high_risk) > 0:
                st.dataframe(high_risk)
            else:
                st.success("No high-risk customers found 🎉")

            st.success("Batch prediction completed successfully!")

        except Exception as e:
            st.error(f"Error occurred: {str(e)}")