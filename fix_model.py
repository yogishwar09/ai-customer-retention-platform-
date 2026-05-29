import joblib
from sklearn.pipeline import Pipeline

model = joblib.load("model/churn_model.pkl")

# re-save in safe format
joblib.dump(model, "model/churn_model.pkl", protocol=4)

print("Model re-saved safely for cloud compatibility")