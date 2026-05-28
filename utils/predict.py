import joblib
import pandas as pd
import numpy as np

# =====================================================
# LOAD ARTIFACTS
# =====================================================
model = joblib.load("model/churn_model.pkl")
preprocessor = joblib.load("model/preprocessor.pkl")


# =====================================================
# FEATURE ENGINEERING (MUST MATCH TRAINING)
# =====================================================
def feature_engineering(df):

    df = df.copy()

    # Average monthly spend
    df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)

    # Long-term customer
    df["IsLongTerm"] = np.where(df["tenure"] > 24, 1, 0)

    # Tenure groups
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-1 Year", "1-2 Years", "2-4 Years", "4-6 Years"]
    )

    # High charges
    df["HighMonthlyCharges"] = np.where(df["MonthlyCharges"] > 70, 1, 0)

    # Total services
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
# PREDICT FUNCTION
# =====================================================
def predict_churn(input_data):

    df = pd.DataFrame([input_data])

    # add engineered features
    df = feature_engineering(df)

    # preprocess
    processed = preprocessor.transform(df)

    prediction = model.predict(processed)[0]
    probability = model.predict_proba(processed)[0][1]

    return prediction, probability