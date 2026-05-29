import streamlit as st
import pandas as pd
from utils.model_loader import get_model 
import streamlit as st
st.set_page_config(layout="wide")

st.set_page_config(page_title="Batch Prediction", layout="wide")

st.title("📦 Batch Churn Prediction")

model = get_model()

file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:

    df = pd.read_csv(file)
    st.dataframe(df.head())

    if model is None:
        st.error("Model not available")
        st.stop()

    try:
        expected_cols = model.feature_names_in_ if hasattr(model, "feature_names_in_") else df.columns

        df_model = df.reindex(columns=expected_cols, fill_value=0)

        probs = model.predict_proba(df_model)[:, 1]
        preds = (probs >= 0.5).astype(int)

        df["Churn_Probability"] = probs
        df["Churn_Prediction"] = preds

        st.success("Prediction completed")
        st.dataframe(df)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "churn_predictions.csv",
            "text/csv"
        )

    except Exception as e:
        st.error("Prediction failed")
        st.code(str(e))