import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "churn_model.pkl")

model = joblib.load(MODEL_PATH)

def predict_churn(input_data):

    df = pd.DataFrame([input_data])

    prob = model.predict_proba(df)[0][1]
    pred = int(prob > 0.5)

    return pred, prob