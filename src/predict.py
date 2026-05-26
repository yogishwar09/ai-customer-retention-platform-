import pickle
import pandas as pd


# Load model and scaler
model = pickle.load(open('model/churn_model.pkl', 'rb'))

scaler = pickle.load(open('model/scaler.pkl', 'rb'))


FEATURE_ORDER = [
    'gender',
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'tenure',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod',
    'MonthlyCharges',
    'TotalCharges'
]


def predict_churn(input_dict):

    input_df = pd.DataFrame([input_dict])

    # Match training feature order
    input_df = input_df[FEATURE_ORDER]

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Probability
    probability = model.predict_proba(input_scaled)[0][1]

    return prediction, probability