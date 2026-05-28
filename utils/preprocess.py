import pandas as pd
import numpy as np

# =====================================================
# FEATURE ENGINEERING
# =====================================================

def engineer_features(df):

    df = df.copy()

    # ==========================================
    # TOTAL CHARGES FIX
    # ==========================================

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # ==========================================
    # AVG MONTHLY SPEND
    # ==========================================

    df["AvgMonthlySpend"] = (
        df["TotalCharges"] /
        (df["tenure"] + 1)
    )

    # ==========================================
    # LONG TERM FLAG
    # ==========================================

    df["IsLongTerm"] = np.where(
        df["tenure"] > 24,
        1,
        0
    )

    # ==========================================
    # HIGH MONTHLY CHARGES
    # ==========================================

    df["HighMonthlyCharges"] = np.where(
        df["MonthlyCharges"] > 70,
        1,
        0
    )

    # ==========================================
    # TENURE GROUP
    # ==========================================

    def tenure_group(x):

        if x <= 12:
            return "0-1 Year"

        elif x <= 24:
            return "1-2 Years"

        elif x <= 48:
            return "2-4 Years"

        else:
            return "4-6 Years"

    df["TenureGroup"] = df["tenure"].apply(
        tenure_group
    )

    # ==========================================
    # TOTAL SERVICES
    # ==========================================

    services = [

        "PhoneService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    existing_services = [
        s for s in services if s in df.columns
    ]

    df["TotalServices"] = (
        df[existing_services] == "Yes"
    ).sum(axis=1)

    return df