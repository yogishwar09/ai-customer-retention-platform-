import numpy as np

from retentioniq.data.schema import FEATURE_COLUMNS, validate_frame
from retentioniq.monitoring.drift import build_reference_profile, drift_report, psi


def test_psi_zero_for_identical_distributions():
    assert psi([0.25] * 4, [0.25] * 4) == 0


def test_psi_large_for_shift():
    assert psi([0.5, 0.5], [0.95, 0.05]) > 0.25


def test_drift_report_flags_shifted_tenure(raw_df):
    df = validate_frame(raw_df).valid[FEATURE_COLUMNS]
    ref = build_reference_profile(df, np.full(len(df), 0.3))
    same = drift_report(ref, df.sample(2000, random_state=0))
    assert (same["status"] == "stable").all()
    shifted = df.sample(2000, random_state=1).copy()
    shifted["tenure"] = (shifted["tenure"] * 0.3).round()
    report = drift_report(ref, shifted).set_index("feature")
    assert report.loc["tenure", "status"] == "alert"
