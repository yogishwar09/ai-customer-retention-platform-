import streamlit as st
import pandas as pd
from utils.model_loader import get_model

st.set_page_config(page_title="Batch Prediction", layout="wide")

st.title("📦 Batch Churn Prediction System")

model = get_model()

if model is None:
    st.stop()

file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    st.subheader("Input Preview")
    st.dataframe(df.head())

    try:
        probs = model.predict_proba(df)[:, 1]
        preds = (probs >= 0.5).astype(int)

        df["Churn_Probability"] = probs
        df["Churn_Prediction"] = preds

        st.subheader("Results")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Results",
            csv,
            "churn_predictions.csv",
            "text/csv"
        )

    except Exception as e:
        st.error("Batch prediction failed")
        st.code(str(e))