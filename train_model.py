import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# =====================================================
# DATA CLEANING
# =====================================================

# Remove customer ID
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Convert target variable
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# =====================================================
# FEATURE ENGINEERING
# =====================================================

# Average monthly spend
df["AvgMonthlySpend"] = (
    df["TotalCharges"] / (df["tenure"] + 1)
)

# Long-term customer
df["IsLongTerm"] = np.where(
    df["tenure"] > 24,
    1,
    0
)

# Tenure groups
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=[
        "0-1 Year",
        "1-2 Years",
        "2-4 Years",
        "4-6 Years"
    ]
)

# High monthly charges
df["HighMonthlyCharges"] = np.where(
    df["MonthlyCharges"] > 70,
    1,
    0
)

# Total services used
service_cols = [
    "PhoneService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

df["TotalServices"] = (
    df[service_cols] == "Yes"
).sum(axis=1)

# =====================================================
# FEATURES & TARGET
# =====================================================

X = df.drop("Churn", axis=1)
y = df["Churn"]

# =====================================================
# COLUMN TYPES
# =====================================================

categorical_cols = X.select_dtypes(include=["object", "category"]).columns
numeric_cols = X.select_dtypes(exclude=["object", "category"]).columns

# =====================================================
# PREPROCESSING PIPELINES
# =====================================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# APPLY PREPROCESSING
# =====================================================

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# =====================================================
# HANDLE CLASS IMBALANCE
# =====================================================

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_processed,
    y_train
)

# =====================================================
# ADVANCED XGBOOST MODEL
# =====================================================

model = XGBClassifier(
    n_estimators=1200,
    learning_rate=0.01,
    max_depth=8,

    min_child_weight=2,
    gamma=0.2,

    subsample=0.8,
    colsample_bytree=0.8,

    scale_pos_weight=3,

    reg_alpha=0.5,
    reg_lambda=1.5,

    objective='binary:logistic',

    random_state=42,
    eval_metric='auc'
)

# =====================================================
# TRAIN MODEL
# =====================================================

print("\nTraining model...\n")

model.fit(
    X_train_smote,
    y_train_smote
)

# =====================================================
# PREDICTIONS
# =====================================================

y_pred = model.predict(X_test_processed)

y_prob = model.predict_proba(
    X_test_processed
)[:, 1]

# =====================================================
# EVALUATION
# =====================================================

accuracy = accuracy_score(y_test, y_pred)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

print("=" * 50)

print(f"\nAccuracy: {round(accuracy * 100, 2)} %")

print(f"ROC-AUC: {round(roc_auc * 100, 2)} %")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

print("=" * 50)

# =====================================================
# SAVE MODEL
# =====================================================

joblib.dump(
    model,
    "model/churn_model.pkl"
)

joblib.dump(
    preprocessor,
    "model/preprocessor.pkl"
)

print("\n✅ Model Saved Successfully!")

import joblib

joblib.dump(model, "model/churn_model.pkl")
joblib.dump(preprocessor, "model/preprocessor.pkl")

# ✅ SAVE FEATURE NAMES
joblib.dump(list(X.columns), "model/features.pkl")

print("✅ Model, Preprocessor & Features saved successfully!")