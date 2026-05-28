import streamlit as st
import plotly.express as px
import pandas as pd

st.title("📊 Churn Analytics Dashboard")

st.markdown("""
<div class='hero-box'>
    <h1>Enterprise AI Churn Dashboard</h1>
    <p>
        Monitor customer churn insights, predictions,
        retention trends, and AI-driven analytics.
    </p>
</div>
""", unsafe_allow_html=True)

# KPI CARDS

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class='metric-card'>
        <h3>Accuracy</h3>
        <h1>75.66%</h1>
        <p>Model Performance</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='metric-card'>
        <h3>ROC-AUC</h3>
        <h1>82.97%</h1>
        <p>Prediction Quality</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='metric-card'>
        <h3>Recall</h3>
        <h1>72%</h1>
        <p>Churn Detection</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class='metric-card'>
        <h3>Status</h3>
        <h1>ACTIVE</h1>
        <p>Production Ready</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# CHARTS

chart1, chart2 = st.columns(2)

with chart1:

    churn_data = pd.DataFrame({
        "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
        "Churn": [12, 19, 15, 22, 18, 25]
    })

    fig = px.line(
        churn_data,
        x="Month",
        y="Churn",
        markers=True,
        title="Monthly Churn Trend"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig, use_container_width=True)

with chart2:

    risk_data = pd.DataFrame({
        "Risk": ["Low", "Medium", "High"],
        "Customers": [420, 230, 150]
    })

    fig2 = px.pie(
        risk_data,
        names="Risk",
        values="Customers",
        hole=0.5,
        title="Customer Risk Distribution"
    )

    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig2, use_container_width=True)