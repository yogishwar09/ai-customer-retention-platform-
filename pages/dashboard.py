import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils.model_loader import get_model 
import streamlit as st
st.set_page_config(layout="wide")

st.set_page_config(page_title="Dashboard", layout="wide")

st.title("📊 Customer Churn Dashboard")

DATA_PATH = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

df = df.replace(" ", np.nan).fillna(0)

if "Churn" in df.columns:
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

model = get_model()

st.subheader("KPIs")

col1, col2, col3 = st.columns(3)

col1.metric("Customers", len(df))
col2.metric("Churn Rate", f"{df['Churn'].mean()*100:.2f}%")
col3.metric("Active", len(df[df["Churn"] == 0]))

st.subheader("Churn Distribution")

fig, ax = plt.subplots()
df["Churn"].value_counts().plot(kind="bar", ax=ax)
st.pyplot(fig)

st.subheader("Contract Impact")

fig, ax = plt.subplots()
df.groupby("Contract")["Churn"].mean().plot(kind="bar", ax=ax)
st.pyplot(fig)

st.subheader("Correlation")

num_df = df.select_dtypes(include=np.number)

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(num_df.corr(), ax=ax)
st.pyplot(fig)