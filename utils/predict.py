import joblib
import pandas as pd

from utils.preprocess import engineer_features

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load(
    "model/churn_model.pkl"
)

preprocessor = joblib.load(
    "model/preprocessor.pkl"
)

# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_churn(input_df):

    # ==========================================
    # FEATURE ENGINEERING
    # ==========================================

    input_df = engineer_features(input_df)

    # ==========================================
    # PREPROCESS
    # ==========================================

    processed_data = preprocessor.transform(
        input_df
    )

    # ==========================================
    # PREDICTION
    # ==========================================

    prediction = model.predict(
        processed_data
    )[0]

    probability = model.predict_proba(
        processed_data
    )[0][1]

    return prediction, probability