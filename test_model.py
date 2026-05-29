import joblib

model = joblib.load("model/churn_model.pkl")
print(type(model))