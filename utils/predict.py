import joblib
import pandas as pd

# load artifacts
model = joblib.load("model/churn_model.pkl")
preprocessor = joblib.load("model/preprocessor.pkl")
features = joblib.load("model/features.pkl")


def predict_churn(input_data):

    df = pd.DataFrame([input_data])

    # align columns exactly
    df = df.reindex(columns=features)

    # ===============================
    # FIX MISSING VALUES ONLY (NO TYPE BREAKAGE)
    # ===============================
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Missing")
        else:
            df[col] = df[col].fillna(0)

    # ===============================
    # IMPORTANT: USE PREPROCESSOR (THIS IS REQUIRED)
    # ===============================
    processed = preprocessor.transform(df)

    # convert to numeric-safe format for xgboost
    processed = pd.DataFrame(processed)

    # ===============================
    # PREDICT
    # ===============================
    prediction = model.predict(processed)[0]
    probability = model.predict_proba(processed)[0][1]

    return prediction, probability