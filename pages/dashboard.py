import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Churn Dashboard", layout="wide")

# =====================================================
# CSS (SaaS STYLE UI)
# =====================================================
st.markdown("""
<style>

.hero-box {
    background: linear-gradient(90deg, #1f1f2e, #2c2c3d);
    padding: 25px;
    border-radius: 15px;
    color: white;
    margin-bottom: 20px;
}

.metric-card {
    background-color: #1e1e2f;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
}

.metric-card h2 {
    color: #4CAF50;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================
st.markdown("""
<div class='hero-box'>
    <h1>📊 Enterprise AI Churn Dashboard</h1>
    <p>AI-powered analytics system for customer churn prediction and business intelligence</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA (SAFE)
# =====================================================
DATA_PATH = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error("Dataset not found")
        st.stop()
    return pd.read_csv(DATA_PATH)

df = load_data()

# Clean
df = df.replace(" ", np.nan)
df = df.fillna(0)

if "Churn" in df.columns:
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# =====================================================
# MODEL LOAD (OPTIONAL)
# =====================================================
MODEL_PATH = os.path.join("model", "churn_model.pkl")
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# =====================================================
# FILTERS
# =====================================================
st.sidebar.header("Filters")

if "Contract" in df.columns:
    contracts = df["Contract"].dropna().unique()
    selected = st.sidebar.multiselect("Contract Type", contracts, default=contracts)
    df = df[df["Contract"].isin(selected)]

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
    df["Churn"].value_counts().plot(kind="bar", ax=ax, color=["green", "red"])
    ax.set_xticklabels(["No Churn", "Churn"], rotation=0)
    st.pyplot(fig)

# =====================================================
# CONTRACT ANALYSIS
# =====================================================
st.subheader("📄 Contract vs Churn")

if "Contract" in df.columns and "Churn" in df.columns:
    contract_churn = df.groupby("Contract")["Churn"].mean() * 100

    fig, ax = plt.subplots()
    contract_churn.plot(kind="bar", ax=ax, color="orange")
    ax.set_ylabel("Churn %")
    st.pyplot(fig)

# =====================================================
# MONTHLY CHARGES
# =====================================================
st.subheader("💰 Monthly Charges Impact")

if "MonthlyCharges" in df.columns and "Churn" in df.columns:
    fig, ax = plt.subplots()
    sns.boxplot(x="Churn", y="MonthlyCharges", data=df, ax=ax)
    ax.set_xticklabels(["No Churn", "Churn"])
    st.pyplot(fig)

# =====================================================
# TENURE ANALYSIS
# =====================================================
st.subheader("⏳ Tenure Analysis")

if "tenure" in df.columns and "Churn" in df.columns:
    fig, ax = plt.subplots()
    sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack", ax=ax)
    st.pyplot(fig)

# =====================================================
# CORRELATION
# =====================================================
st.subheader("🔗 Correlation Heatmap")

num_df = df.select_dtypes(include=[np.number])

if num_df.shape[1] > 1:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(num_df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# =====================================================
# FEATURE IMPORTANCE
# =====================================================
st.subheader("📌 Feature Importance")

try:
    if model is not None and hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_)
        imp = imp.sort_values(ascending=False)

        fig, ax = plt.subplots()
        imp.head(10).plot(kind="bar", ax=ax, color="purple")
        st.pyplot(fig)
    else:
        st.info("Feature importance not available")
except:
    st.warning("Could not load feature importance")

# =====================================================
# REVENUE LOSS ANALYSIS
# =====================================================
st.subheader("💰 Revenue Impact Analysis")

if "Churn" in df.columns and "MonthlyCharges" in df.columns:

    churned = df[df["Churn"] == 1]

    revenue_loss = churned["MonthlyCharges"].sum()
    avg_loss = churned["MonthlyCharges"].mean()

    col1, col2 = st.columns(2)

    col1.metric("Total Revenue Loss", f"${revenue_loss:,.2f}")
    col2.metric("Avg Loss per Customer", f"${avg_loss:,.2f}")

# =====================================================
# RISK SEGMENTATION
# =====================================================
st.subheader("🎯 Risk Segmentation")

if "Churn" in df.columns:

    df["Risk"] = df["Churn"].apply(lambda x: "High Risk" if x == 1 else "Low Risk")

    risk = df["Risk"].value_counts().reset_index()
    risk.columns = ["Risk", "Customers"]

    fig = px.bar(risk, x="Risk", y="Customers", color="Risk")
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# INSIGHTS
# =====================================================
st.subheader("🧠 Business Insights")

st.success("Month-to-month contracts show highest churn risk")
st.info("Low tenure customers are most vulnerable")
st.warning("Higher charges increase churn probability")
st.success("Long-term contracts improve retention significantly")
st.info("Early engagement reduces churn drastically")

# =====================================================
# RAW DATA
# =====================================================
with st.expander("📄 View Data"):
    st.dataframe(df)