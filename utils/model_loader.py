import os
import streamlit as st
import joblib

MODEL_PATH = os.path.join("model", "churn_model.pkl")


@st.cache_resource
def get_model():
    try:
        if not os.path.exists(MODEL_PATH):
            return None

        # SAFE loading (fixes cloud + pickle issues)
        with open(MODEL_PATH, "rb") as f:
            model = joblib.load(f)

        return model

    except Exception as e:
        st.error("❌ Model loading failed on Streamlit Cloud")
        st.code(str(e))
        return None