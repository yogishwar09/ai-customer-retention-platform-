"""Generate synthetic sample files for demos and tests.

Rows are sampled feature-by-feature from the reference data's *conditional*
structure (whole-row bootstrap + numeric jitter) and given synthetic IDs, so no
original customer identifier is published in the samples.

    python scripts/make_samples.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from retentioniq import config
from retentioniq.data.loader import load_training_data
from retentioniq.data.schema import FEATURE_COLUMNS


def synthesize(df: pd.DataFrame, n: int, seed: int, prefix: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df[FEATURE_COLUMNS].sample(n=n, replace=True, random_state=seed).reset_index(drop=True)
    out["tenure"] = np.clip(out["tenure"] + rng.integers(-2, 3, n), 0, 72).astype(int)
    out["MonthlyCharges"] = np.round(np.clip(out["MonthlyCharges"] * rng.uniform(0.95, 1.05, n), 18, 120), 2)
    out["TotalCharges"] = np.round(out["MonthlyCharges"] * np.maximum(out["tenure"], 0)
                                   * rng.uniform(0.95, 1.05, n), 2)
    out.loc[out["tenure"] == 0, "TotalCharges"] = np.nan
    out.insert(0, "customerID", [f"{prefix}-{i:05d}" for i in range(1, n + 1)])
    return out


def main() -> None:
    df, _ = load_training_data()
    config.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    clean = synthesize(df, 300, seed=7, prefix="SYN")
    clean.to_csv(config.SAMPLES_DIR / "batch_clean.csv", index=False)

    errors = synthesize(df, 60, seed=11, prefix="ERR")
    errors = errors.astype(object)
    errors.loc[2, "Contract"] = "Weekly"
    errors.loc[5, "MonthlyCharges"] = "abc"
    errors.loc[8, "tenure"] = -4
    errors.loc[11, "PaymentMethod"] = "Crypto"
    errors.loc[14, "customerID"] = errors.loc[13, "customerID"]
    errors.loc[17, "InternetService"] = "No"            # inconsistent with add-ons
    errors.loc[20, "gender"] = None
    errors.loc[23, "TotalCharges"] = 99999              # inconsistent -> warning only
    errors.loc[26, "PaymentMethod"] = "credit card"     # alias -> accepted
    errors = errors.rename(columns={"MonthlyCharges": "monthly_charges", "tenure": "Tenure"})
    errors["internal_notes"] = "ignored column"
    errors.to_csv(config.SAMPLES_DIR / "batch_with_errors.csv", index=False)

    drifted = synthesize(df, 400, seed=23, prefix="DRF")
    rng = np.random.default_rng(23)
    drifted["tenure"] = np.clip((drifted["tenure"] * 0.45).round(), 0, 72).astype(int)
    drifted["MonthlyCharges"] = np.round(np.clip(drifted["MonthlyCharges"] * 1.18, 18, 140), 2)
    flip = rng.random(len(drifted)) < 0.35
    drifted.loc[flip, "Contract"] = "Month-to-month"
    drifted["TotalCharges"] = np.round(drifted["MonthlyCharges"] * drifted["tenure"], 2)
    drifted.loc[drifted["tenure"] == 0, "TotalCharges"] = np.nan
    drifted.to_csv(config.SAMPLES_DIR / "batch_drifted.csv", index=False)

    template = clean.head(3)
    template.to_csv(config.SAMPLES_DIR / "template.csv", index=False)
    print("Wrote samples to", config.SAMPLES_DIR)


if __name__ == "__main__":
    main()
