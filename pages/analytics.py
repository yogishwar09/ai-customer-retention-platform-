import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Churn Analytics Dashboard", layout="wide")

st.title("📊 Customer Churn Analytics Dashboard")
st.markdown("AI-powered business intelligence system for telecom churn analysis")

# =====================================================
# DATA PATH
# =====================================================
DATA_PATH = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error("❌ Dataset not found")
        st.stop()
    return pd.read_csv(DATA_PATH)

df = load_data()

# =====================================================
# CLEAN DATA
# =====================================================
df = df.replace(" ", np.nan)
df = df.fillna(0)

if "Churn" in df.columns:
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

st.success("✅ Dataset loaded successfully!")

# =====================================================
# LOAD MODEL (OPTIONAL)
# =====================================================
MODEL_PATH = os.path.join("model", "churn_model.pkl")
model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

# =====================================================
# SIDEBAR FILTERS
# =====================================================
st.sidebar.header("🔍 Filters")

if "Contract" in df.columns:
    contracts = df["Contract"].dropna().unique()
    selected_contracts = st.sidebar.multiselect(
        "Contract Type",
        contracts,
        default=contracts
    )
    df = df[df["Contract"].isin(selected_contracts)]

# =====================================================
# KPI SECTION
# =====================================================
st.subheader("📌 Executive KPIs")

total_customers = len(df)

churn_rate = df["Churn"].mean() * 100 if "Churn" in df.columns else 0
active_customers = len(df[df["Churn"] == 0]) if "Churn" in df.columns else 0

col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", total_customers)
col2.metric("Churn Rate (%)", f"{churn_rate:.2f}")
col3.metric("Active Customers", active_customers)

st.divider()

# =====================================================
# CHURN DISTRIBUTION
# =====================================================
st.subheader("📊 Churn Distribution")

if "Churn" in df.columns:
    fig, ax = plt.subplots()
    df["Churn"].value_counts().plot(kind="bar", color=["green", "red"], ax=ax)
    ax.set_xticklabels(["No Churn", "Churn"], rotation=0)
    ax.set_ylabel("Count")
    st.pyplot(fig)

# =====================================================
# CONTRACT ANALYSIS
# =====================================================
st.subheader("📄 Contract Type vs Churn")

if "Contract" in df.columns and "Churn" in df.columns:
    contract_churn = df.groupby("Contract")["Churn"].mean() * 100

    fig, ax = plt.subplots()
    contract_churn.plot(kind="bar", ax=ax, color="orange")
    ax.set_ylabel("Churn Rate (%)")
    st.pyplot(fig)

# =====================================================
# MONTHLY CHARGES IMPACT
# =====================================================
st.subheader("💰 Monthly Charges Impact on Churn")

if "MonthlyCharges" in df.columns and "Churn" in df.columns:
    fig, ax = plt.subplots()
    sns.boxplot(x="Churn", y="MonthlyCharges", data=df, ax=ax)
    ax.set_xticklabels(["No Churn", "Churn"])
    st.pyplot(fig)

# =====================================================
# TENURE ANALYSIS
# =====================================================
st.subheader("⏳ Customer Tenure Analysis")

if "tenure" in df.columns and "Churn" in df.columns:
    fig, ax = plt.subplots()
    sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack", ax=ax)
    st.pyplot(fig)

# =====================================================
# CORRELATION HEATMAP
# =====================================================
st.subheader("🔗 Feature Correlation Heatmap")

numeric_df = df.select_dtypes(include=[np.number])

if numeric_df.shape[1] > 1:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# =====================================================
# FEATURE IMPORTANCE
# =====================================================
st.subheader("📌 Feature Importance (Model Insights)")

try:
    if model is not None and hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_)
        importances = importances.sort_values(ascending=False)

        fig, ax = plt.subplots()
        importances.head(10).plot(kind="bar", ax=ax, color="purple")
        ax.set_title("Top 10 Important Features")
        st.pyplot(fig)
    else:
        st.info("Feature importance not available for this model.")
except:
    st.warning("Could not load feature importance safely.")

# =====================================================
# BUSINESS INSIGHTS (UPGRADED UI)
# =====================================================
st.subheader("🧠 Business Insights")

st.success("✔ Month-to-month contracts have highest churn risk")
st.info("✔ Customers with low tenure are most likely to leave")
st.warning("✔ Higher monthly charges increase churn probability")
st.success("✔ Long-term contracts significantly reduce churn")
st.info("✔ Early customer engagement improves retention")

# =====================================================
# RAW DATA
# =====================================================
with st.expander("📄 View Raw Data"):
    st.dataframe(df)