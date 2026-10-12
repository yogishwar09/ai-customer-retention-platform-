import numpy as np
import pandas as pd
import pytest

from retentioniq.data.schema import FEATURE_COLUMNS, SchemaError, map_columns, validate_frame


def test_valid_row_is_accepted(frame):
    r = validate_frame(frame())
    assert r.summary["rows_accepted"] == 1 and r.rejected.empty
    assert list(r.valid.columns[:1]) == ["customerID"]
    assert r.valid["SeniorCitizen"].dtype.kind == "i"


def test_reference_dataset_passes_contract(raw_df):
    r = validate_frame(raw_df, require_ids=True)
    assert r.summary["rows_rejected"] == 0
    # 11 brand-new customers have blank TotalCharges -> accepted with a warning
    assert r.summary["rows_with_warnings"] >= 11


@pytest.mark.parametrize("override, message", [
    ({"Contract": "Weekly"}, "unknown value 'Weekly'"),
    ({"tenure": -1}, "below the minimum"),
    ({"MonthlyCharges": "abc"}, "must be numeric"),
    ({"gender": None}, "gender is missing"),
    ({"SeniorCitizen": 3}, "SeniorCitizen must be 0 or 1"),
    ({"InternetService": "No"}, "InternetService is 'No'"),
    ({"PhoneService": "No"}, "PhoneService is 'No'"),
    ({"OnlineSecurity": "No internet service"}, "has internet"),
])
def test_bad_values_are_rejected_with_reason(frame, override, message):
    r = validate_frame(frame(override))
    assert r.summary["rows_rejected"] == 1
    assert message in r.rejected["errors"].iloc[0]


def test_aliases_and_case_are_normalised(frame):
    r = validate_frame(frame({"PaymentMethod": "credit card", "Contract": "month-to-month",
                              "SeniorCitizen": "Yes"}))
    row = r.valid.iloc[0]
    assert row["PaymentMethod"] == "Credit card (automatic)"
    assert row["Contract"] == "Month-to-month"
    assert row["SeniorCitizen"] == 1


def test_blank_total_charges_is_a_warning_not_an_error(frame):
    r = validate_frame(frame({"TotalCharges": ""}))
    assert r.summary["rows_accepted"] == 1
    assert np.isnan(r.valid.iloc[0]["TotalCharges"])
    assert any("TotalCharges is blank" in w for w in r.row_warnings[0])


def test_inconsistent_totals_warn(frame):
    r = validate_frame(frame({"TotalCharges": 99999}))
    assert r.quality_status(0) == "warning"


def test_duplicate_ids_keep_first(frame):
    df = frame({}, {})
    df.loc[1, "customerID"] = df.loc[0, "customerID"]
    r = validate_frame(df)
    assert r.summary["rows_accepted"] == 1
    assert "duplicate customerID" in r.rejected["errors"].iloc[0]


def test_missing_columns_fail_the_whole_file(frame):
    with pytest.raises(SchemaError, match="Missing required column"):
        validate_frame(frame().drop(columns=["Contract"]))


def test_empty_frame_is_rejected():
    with pytest.raises(SchemaError):
        validate_frame(pd.DataFrame())


def test_column_mapping_is_case_and_punctuation_insensitive():
    m = map_columns(["Monthly_Charges", "TENURE", "customer id", "notes", *FEATURE_COLUMNS[:3]])
    assert m.mapped["Monthly_Charges"] == "MonthlyCharges"
    assert m.mapped["TENURE"] == "tenure"
    assert m.mapped["customer id"] == "customerID"
    assert "notes" in m.ignored


def test_target_leakage_column_is_not_a_feature(frame):
    df = frame()
    df["Churn"] = "Yes"
    r = validate_frame(df)
    assert "Churn" not in FEATURE_COLUMNS
    assert r.valid["Churn"].iloc[0] == 1
