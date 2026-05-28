import joblib
import pandas as pd
import numpy as np

# =====================================================
# LOAD ONLY ONE MODEL (NO PREPROCESSOR)
# =====================================================
model = joblib.load("model/final_model.pkl")


# =====================================================
# FEATURE ENGINEERING (ONLY IF USED IN TRAINING)
# =====================================================
def feature_engineering(df):

    df = df.copy()

    df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["IsLongTerm"] = np.where(df["tenure"] > 24, 1, 0)

    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-1 Year", "1-2 Years", "2-4 Years", "4-6 Years"]
    )

    df["HighMonthlyCharges"] = np.where(df["MonthlyCharges"] > 70, 1, 0)

    service_cols = [
        "PhoneService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    df["TotalServices"] = (df[service_cols] == "Yes").sum(axis=1)

    return df


# =====================================================
# PREDICTION FUNCTION
# =====================================================
def predict_churn(input_data):

    df = pd.DataFrame([input_data])

    # feature engineering
    df = feature_engineering(df)

    # FORCE SAFE TYPES (IMPORTANT FIX)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)

    # replace NaN
    df = df.fillna(0)

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return prediction, probability