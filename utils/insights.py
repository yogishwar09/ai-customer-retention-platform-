def generate_insights(data):

    insights = []

    # Contract risk
    if data.get("Contract") == "Month-to-month":
        insights.append("⚠ Month-to-month contract increases churn risk")

    # Charges risk
    if data.get("MonthlyCharges", 0) > 80:
        insights.append("⚠ High monthly charges increase churn probability")

    # Tenure risk
    if data.get("tenure", 0) < 12:
        insights.append("⚠ New customers are more likely to churn")

    # Tech support
    if data.get("TechSupport") == "No":
        insights.append("⚠ Lack of tech support increases churn risk")

    # Security
    if data.get("OnlineSecurity") == "No":
        insights.append("⚠ No online security increases churn risk")

    if len(insights) == 0:
        insights.append("✅ Customer profile looks stable")

    return insights