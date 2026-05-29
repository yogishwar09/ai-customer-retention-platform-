import joblib

model = joblib.load("model/churn_model.pkl")

joblib.dump(model, "model/churn_model.pkl", protocol=4)

print("Model re-saved successfully in safe format")