import joblib
import pandas as pd

# =====================================================
# LOAD FINAL PIPELINE MODEL
# =====================================================
model = joblib.load("model/final_model.pkl")


# =====================================================
# PREDICTION FUNCTION
# =====================================================
def predict_churn(input_data):

    # Convert input dict → DataFrame
    df = pd.DataFrame([input_data])

    # =================================================
    # DIRECT PREDICTION (PIPELINE HANDLES EVERYTHING)
    # =================================================
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return prediction, probability