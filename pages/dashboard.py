import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Dashboard", layout="wide")

st.title("📊 Customer Churn Dashboard")

# =========================
# DATA PATH SAFETY
# =========================
DATA_PATH = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error("❌ Dataset not found in data folder")
        st.stop()
    return pd.read_csv(DATA_PATH)

df = load_data()

# =========================
# CLEANING (SAFE)
# =========================
df = df.replace(" ", np.nan)

# Fill only numeric-safe missing values
df = df.fillna(0)

# Convert churn safely
if "Churn" in df.columns:
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# =========================
# KPI SECTION (SAFE)
# =========================
st.subheader("📌 KPIs")

total_customers = len(df)

churn_rate = df["Churn"].mean() * 100 if "Churn" in df.columns else 0

active_customers = (
    len(df[df["Churn"] == 0]) if "Churn" in df.columns else 0
)

col1, col2, col3 = st.columns(3)

col1.metric("Customers", total_customers)
col2.metric("Churn Rate (%)", f"{churn_rate:.2f}")
col3.metric("Active Customers", active_customers)

st.divider()

# =========================
# CHURN DISTRIBUTION
# =========================
st.subheader("📊 Churn Distribution")

if "Churn" in df.columns:
    fig, ax = plt.subplots()

    df["Churn"].value_counts().sort_index().plot(
        kind="bar",
        ax=ax,
        color=["green", "red"]
    )

    ax.set_xticklabels(["No Churn", "Churn"], rotation=0)
    ax.set_ylabel("Count")

    st.pyplot(fig)
else:
    st.warning("Churn column not available")

# =========================
# CONTRACT IMPACT
# =========================
st.subheader("📄 Contract Impact")

if "Contract" in df.columns and "Churn" in df.columns:
    fig, ax = plt.subplots()

    df.groupby("Contract")["Churn"].mean().plot(
        kind="bar",
        ax=ax,
        color="orange"
    )

    ax.set_ylabel("Churn Rate")

    st.pyplot(fig)
else:
    st.info("Contract data not available")

# =========================
# CORRELATION HEATMAP
# =========================
st.subheader("🔗 Correlation Analysis")

num_df = df.select_dtypes(include=np.number)

if num_df.shape[1] > 1:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(num_df.corr(), ax=ax, cmap="coolwarm")
    st.pyplot(fig)
else:
    st.warning("Not enough numeric data for correlation")

# =========================
# FOOTER INFO
# =========================
st.success("Dashboard loaded successfully 🚀")