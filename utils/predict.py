import joblib
import pandas as pd

model = joblib.load("model/churn_model.pkl")
features = joblib.load("model/features.pkl")


def predict_churn(input_data):

    df = pd.DataFrame([input_data])

    # align columns
    df = df.reindex(columns=features)

    # ===============================
    # SAFE CLEANING (NO sklearn dependency)
    # ===============================

    for col in df.columns:

        # convert everything to string (safe for encoding pipelines)
        df[col] = df[col].astype(str)

        # fix missing values
        df[col] = df[col].replace("nan", "Missing")

    # ===============================
    # DIRECT MODEL PREDICTION
    # ===============================
    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0][1]

    return prediction, probability