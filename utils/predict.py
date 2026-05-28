import joblib
import pandas as pd

# =====================================================
# LOAD ARTIFACTS
# =====================================================
model = joblib.load("model/churn_model.pkl")
preprocessor = joblib.load("model/preprocessor.pkl")
features = joblib.load("model/features.pkl")


# =====================================================
# PREDICTION FUNCTION (STABLE VERSION)
# =====================================================
def predict_churn(input_data):

    # Convert input → DataFrame
    df = pd.DataFrame([input_data])

    # =================================================
    # ALIGN COLUMNS EXACTLY AS TRAINING
    # =================================================
    df = df.reindex(columns=features)

    # Fill missing values safely
    df = df.fillna(0)

    # Convert all values to string-safe format (prevents sklearn crash)
    for col in df.columns:
        df[col] = df[col].astype(str)

    # =================================================
    # TRANSFORM + PREDICT
    # =================================================
    processed = preprocessor.transform(df)

    prediction = model.predict(processed)[0]
    probability = model.predict_proba(processed)[0][1]

    return prediction, probability