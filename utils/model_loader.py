import joblib
import os
import streamlit as st

MODEL_PATH = os.path.join("model", "churn_model.pkl")

@st.cache_resource
def get_model():
    try:
        if not os.path.exists(MODEL_PATH):
            return None

        model = joblib.load(MODEL_PATH)
        return model

    except Exception as e:
        st.error("❌ Model loading failed")
        st.code(str(e))
        return None