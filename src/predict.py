import joblib
import pandas as pd

model = joblib.load("model/churn_model.pkl")
preprocessor = joblib.load("model/preprocessor.pkl")
features = joblib.load("model/features.pkl")


def predict_churn(input_data):

    df = pd.DataFrame([input_data])

    # align columns
    df = df.reindex(columns=features, fill_value=0)

    # preprocess
    processed = preprocessor.transform(df)

    prediction = model.predict(processed)[0]
    probability = model.predict_proba(processed)[0][1]

    return prediction, probability