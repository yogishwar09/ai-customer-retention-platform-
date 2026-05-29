import joblib
import os

# Path to your model
MODEL_PATH = os.path.join("model", "churn_model.pkl")

_model = None

def get_model():
    """
    Loads model ONLY ONCE (prevents Streamlit reloading crashes)
    """
    global _model

    if _model is None:
        try:
            _model = joblib.load(MODEL_PATH)
        except Exception as e:
            raise RuntimeError(
                "Model loading failed. Check compatibility of churn_model.pkl"
            ) from e

    return _model