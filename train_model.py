import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# =====================================================
# CLEANING
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
    "PhoneService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

df["TotalServices"] = (df[service_cols] == "Yes").sum(axis=1)

# =====================================================
# SPLIT FEATURES
# =====================================================
X = df.drop("Churn", axis=1)
y = df["Churn"]

categorical_cols = X.select_dtypes(include=["object", "category"]).columns
numeric_cols = X.select_dtypes(exclude=["object", "category"]).columns

# =====================================================
# PREPROCESSOR
# =====================================================
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# =====================================================
# SPLIT DATA
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# TRANSFORM + SMOTE
# =====================================================
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(
    X_train_processed,
    y_train
)

# =====================================================
# MODEL
# =====================================================
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    random_state=42
)

print("Training model...")
model.fit(X_train_smote, y_train_smote)

# =====================================================
# SAVE ARTIFACTS (CLEAN ONLY ONCE)
# =====================================================
joblib.dump(model, "model/churn_model.pkl")
joblib.dump(preprocessor, "model/preprocessor.pkl")
joblib.dump(list(X.columns), "model/features.pkl")

print("✅ Model training completed and saved successfully!")