import streamlit as st
import pandas as pd
import plotly.express as px 

from src.predict import predict_churn


# PAGE CONFIGURATION
st.set_page_config(
    page_title="AI Customer Retention Platform",
    page_icon="📊",
    layout="wide"
)


# CUSTOM CSS STYLING 

# LOAD DATA
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

importance_df = pd.read_csv('model/feature_importance.csv')
st.markdown("""
<style>

/* Main App Background */
.stApp {
    background-color: #0E1117;
    color: white;
}

/* Headings */
h1, h2, h3, h4 {
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161A23;
}

/* Metric Cards */
div[data-testid="metric-container"] {
    background: linear-gradient(145deg, #1f2937, #111827);
    border: 1px solid #374151;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.4);
}

/* Buttons */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #FF4B4B, #FF6B6B);
    color: white;
    border: none;
    border-radius: 12px;
    height: 3.2em;
    font-size: 18px;
    font-weight: bold;
}

/* Button Hover */
.stButton > button:hover {
    background: linear-gradient(90deg, #FF2E2E, #FF4B4B);
    color: white;
}

/* Success Box */
.stSuccess {
    border-radius: 12px;
}

/* Error Box */
.stAlert {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# HEADER
st.title("📊 AI Customer Retention Intelligence Platform")

st.markdown("""
### Predict telecom customer churn using Machine Learning intelligence.
""")


# SIDEBAR
st.sidebar.header("Customer Profile")


# INPUTS
gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male"]
)

senior = st.sidebar.selectbox(
    "Senior Citizen",
    [0, 1]
)

partner = st.sidebar.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.sidebar.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.sidebar.slider(
    "Tenure (Months)",
    0,
    72,
    12
)

monthly_charges = st.sidebar.slider(
    "Monthly Charges",
    0,
    150,
    70
)

total_charges = st.sidebar.number_input(
    "Total Charges",
    0.0,
    10000.0,
    1000.0
)


# ENCODING
gender = 1 if gender == "Male" else 0
partner = 1 if partner == "Yes" else 0
dependents = 1 if dependents == "Yes" else 0


# INPUT DATA
input_dict = {
    'gender': gender,
    'SeniorCitizen': senior,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': 1,
    'MultipleLines': 0,
    'InternetService': 0,
    'OnlineSecurity': 0,
    'OnlineBackup': 0,
    'DeviceProtection': 0,
    'TechSupport': 0,
    'StreamingTV': 0,
    'StreamingMovies': 0,
    'Contract': 0,
    'PaperlessBilling': 1,
    'PaymentMethod': 0,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}


# PREDICTION BUTTON
if st.button("🚀 Predict Churn"):

    prediction, probability = predict_churn(input_dict)

    st.subheader("📈 Prediction Analysis")

    # METRICS
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )

    with col2:
        st.metric(
            "Retention Probability",
            f"{(1 - probability):.2%}"
        )

    # HIGH RISK
    if prediction == 1:

        st.error("⚠️ High Risk Customer")

        st.markdown("""
        ## Recommended Retention Actions

        - Offer personalized discounts
        - Provide loyalty incentives
        - Improve customer engagement
        - Offer premium support services
        - Provide long-term subscription benefits
        """)

    # LOW RISK
    else:

        st.success("✅ Low Churn Risk")

        st.markdown("""
        ## Positive Indicators

        - Stable customer relationship
        - Lower churn probability
        - Good retention likelihood
        - Strong customer engagement
        - Healthy customer lifecycle
        """)
        
        # ANALYTICS SECTION
st.markdown("---")

st.header("📊 Customer Analytics Dashboard")


# CHURN DISTRIBUTION
churn_count = df['Churn'].value_counts().reset_index()

fig1 = px.pie(
    churn_count,
    names='Churn',
    values='count',
    title='Customer Churn Distribution'
)

st.plotly_chart(fig1, use_container_width=True)


# MONTHLY CHARGES VS CHURN
fig2 = px.box(
    df,
    x='Churn',
    y='MonthlyCharges',
    color='Churn',
    title='Monthly Charges vs Churn'
)

st.plotly_chart(fig2, use_container_width=True)


# TENURE DISTRIBUTION
fig3 = px.histogram(
    df,
    x='tenure',
    color='Churn',
    title='Customer Tenure Distribution',
    barmode='overlay'
)

st.plotly_chart(fig3, use_container_width=True)

# FEATURE IMPORTANCE SECTION
st.markdown("---")

st.header("🧠 Explainable AI Insights")


# SORT FEATURES
importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)


# FEATURE IMPORTANCE CHART
fig4 = px.bar(
    importance_df.head(10),
    x='Importance',
    y='Feature',
    orientation='h',
    title='Top Features Influencing Customer Churn'
)

st.plotly_chart(fig4, use_container_width=True)


# BUSINESS INSIGHTS
st.markdown("""
## Key Business Insights

- Customers with shorter tenure are more likely to churn
- Higher monthly charges increase churn risk
- Contract type strongly impacts retention
- Tech support and online security improve customer retention
- Long-term customers show better loyalty patterns
""")