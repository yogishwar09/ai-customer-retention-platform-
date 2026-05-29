import streamlit as st
import os

MODEL_PATH = os.path.join("model", "churn_model.pkl")

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        return None

    try:
        import joblib

        # Load safely with timeout protection idea
        model = joblib.load(MODEL_PATH)
        return model

    except Exception as e:
        # DO NOT CRASH APP
        st.warning("Model could not be loaded. Running in demo mode.")
        st.code(str(e))
        return None