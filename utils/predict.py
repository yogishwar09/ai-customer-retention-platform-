import joblib
import pandas as pd

# =====================================================
# LOAD MODEL + PREPROCESSOR + FEATURES
# =====================================================
model = joblib.load("model/churn_model.pkl")
preprocessor = joblib.load("model/preprocessor.pkl")
features = joblib.load("model/features.pkl")


# =====================================================
# PREDICTION FUNCTION
# =====================================================
def predict_churn(input_data):

    df = pd.DataFrame([input_data])

    # =================================================
    # FORCE EXACT FEATURE ALIGNMENT
    # =================================================
    df = df.reindex(columns=features)

    # Fill missing numeric values safely
    df = df.fillna(0)

    # Convert all object columns safely (VERY IMPORTANT FIX)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)

    # =================================================
    # PREPROCESS
    # =================================================
    processed = preprocessor.transform(df)

    prediction = model.predict(processed)[0]
    probability = model.predict_proba(processed)[0][1]

    return prediction, probability