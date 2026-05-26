import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score, classification_report

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

import pickle


# Load dataset
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')


# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')


# Remove missing values
df.dropna(inplace=True)


# Remove customerID column
df.drop('customerID', axis=1, inplace=True)


# Convert target variable
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})


# Encode categorical columns
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])


# Features and target
X = df.drop('Churn', axis=1)
print(X.columns.tolist())

y = df['Churn']


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Apply SMOTE
X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(
    X_train,
    y_train
)


# Feature scaling
scaler = StandardScaler()

X_train_sc = scaler.fit_transform(X_train_res)

X_test_sc = scaler.transform(X_test)


# Test output
print("Preprocessing completed successfully")

print(X_train_sc.shape)

print(X_test_sc.shape) 

rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(X_train_res, y_train_res)

rf_preds = rf_model.predict(X_test)

print("Random Forest Results")

print(classification_report(y_test, rf_preds))

import os

os.makedirs('model', exist_ok=True)

pickle.dump(rf_model, open('model/churn_model.pkl', 'wb'))

pickle.dump(scaler, open('model/scaler.pkl', 'wb'))

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})

feature_importance.to_csv(
    'model/feature_importance.csv',
    index=False
)

print("Feature importance saved successfully")