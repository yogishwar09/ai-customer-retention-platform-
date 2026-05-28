import joblib
import pandas as pd

# Load artifacts
model = joblib.load("model/churn_model.pkl")
preprocessor = joblib.load("model/preprocessor.pkl")
features = joblib.load("model/features.pkl")


def predict_churn(input_data):

    df = pd.DataFrame([input_data])

    # Align columns
    df = df.reindex(columns=features)

    # ===============================
    # 🚨 FAST EMERGENCY FIX (CRITICAL)
    # ===============================

    # convert everything to string-safe format first
    for col in df.columns:
        df[col] = df[col].astype(str)

    # replace missing values safely
    df = df.replace("nan", "Missing")

    # ===============================
    # TRANSFORM
    # ===============================
    processed = preprocessor.transform(df)

    # ===============================
    # PREDICT
    # ===============================
    prediction = model.predict(processed)[0]
    probability = model.predict_proba(processed)[0][1]

    return prediction, probability