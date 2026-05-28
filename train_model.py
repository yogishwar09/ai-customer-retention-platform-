import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# =====================================================
# CLEAN DATA
# =====================================================
df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# =====================================================
# FEATURE ENGINEERING
# =====================================================
df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)
df["IsLongTerm"] = np.where(df["tenure"] > 24, 1, 0)

df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-1 Year", "1-2 Years", "2-4 Years", "4-6 Years"]
)

df["HighMonthlyCharges"] = np.where(df["MonthlyCharges"] > 70, 1, 0)

service_cols = [
    "PhoneService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
]

df["TotalServices"] = (df[service_cols] == "Yes").sum(axis=1)

# =====================================================
# SPLIT DATA
# =====================================================
X = df.drop("Churn", axis=1)
y = df["Churn"]

# =====================================================
# COLUMN TYPES
# =====================================================
categorical_cols = X.select_dtypes(include=["object", "category"]).columns
numeric_cols = X.select_dtypes(exclude=["object", "category"]).columns

# =====================================================
# PREPROCESSOR
# =====================================================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# =====================================================
# FINAL MODEL PIPELINE
# =====================================================
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        eval_metric="auc"
    ))
])

# =====================================================
# TRAIN TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# TRAIN MODEL
# =====================================================
print("Training model...")
model.fit(X_train, y_train)

# =====================================================
# SAVE MODEL
# =====================================================
joblib.dump(model, "model/final_model.pkl")

print("✅ FINAL MODEL SAVED SUCCESSFULLY!")