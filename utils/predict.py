import joblib
import pandas as pd

# ===============================
# LOAD MODEL + PREPROCESSOR + FEATURES
# ===============================
model = joblib.load("model/churn_model.pkl")
preprocessor = joblib.load("model/preprocessor.pkl")
features = joblib.load("model/features.pkl")


# ===============================
# MAIN PREDICTION FUNCTION
# ===============================
def predict_churn(input_data):
    """
    input_data: dict (single customer data)
    """

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])

    # ===============================
    # ALIGN COLUMNS WITH TRAINING DATA
    # ===============================
    input_df = input_df.reindex(columns=features, fill_value=0)

    # ===============================
    # PREPROCESS
    # ===============================
    processed_data = preprocessor.transform(input_df)

    # ===============================
    # PREDICT
    # ===============================
    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0][1]

    # ✅ FIX: return clean tuple
    return prediction, probability