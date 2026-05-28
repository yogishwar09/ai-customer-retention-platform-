import joblib
import pandas as pd

# ===============================
# LOAD ARTIFACTS
# ===============================
model = joblib.load("model/churn_model.pkl")
preprocessor = joblib.load("model/preprocessor.pkl")
features = joblib.load("model/features.pkl")


# ===============================
# MAIN FUNCTION
# ===============================
def predict_churn(input_data):

    df = pd.DataFrame([input_data])

    # ===============================
    # FORCE ALL MISSING COLUMNS
    # ===============================
    df = df.reindex(columns=features)

    # ===============================
    # FIX DATA TYPES (VERY IMPORTANT)
    # ===============================
    for col in df.columns:

        # numeric handling
        if df[col].dtype != "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # fill missing values
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Missing")
        else:
            df[col] = df[col].fillna(0)

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