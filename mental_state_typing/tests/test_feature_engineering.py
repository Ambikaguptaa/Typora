"""Unit tests for the Feature Engineering module and Feature Manifest."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data_engineering.cleaner import clean_dataset_with_report
from src.data_engineering.dataset_adapter import load_dataset
from src.data_engineering.feature_engineering import (
    build_feature_table,
    calculate_backspace_features,
    calculate_dwell_time,
    calculate_error_features,
    calculate_flight_time,
    calculate_pause_features,
    calculate_session_statistics,
    calculate_typing_speed,
    calculate_variability_features,
    calculate_word_features,
    get_available_features,
    validate_feature_table,
)
from src.data_engineering.feature_manifest import generate_feature_manifest

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "sample" / "sample_keystrokes.csv"


def test_calculate_dwell_time():
    """Verify dwell time calculation from raw press/release timestamps."""
    df = pd.DataFrame(
        {
            "press_time": [100.0, 300.0, 500.0],
            "release_time": [180.0, 420.0, 590.0],
        }
    )
    dwell = calculate_dwell_time(df)
    assert dwell is not None
    assert list(dwell) == [80.0, 120.0, 90.0]


def test_calculate_flight_time_first_event_nan():
    """Verify inter-key flight time calculation and first event handled as NaN."""
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "session_id": ["s1", "s1", "s1"],
            "press_time": [100.0, 250.0, 450.0],
            "release_time": [180.0, 350.0, 520.0],
        }
    )
    flight = calculate_flight_time(df, user_col="user_id", session_col="session_id")
    assert flight is not None
    assert np.isnan(flight.iloc[0])  # First event must be NaN
    assert flight.iloc[1] == 70.0    # 250 - 180
    assert flight.iloc[2] == 100.0   # 450 - 350


def test_calculate_flight_time_no_cross_session_bleed():
    """Verify flight times do not bleed across different sessions."""
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "session_id": ["sess_A", "sess_B"],
            "press_time": [100.0, 200.0],
            "release_time": [150.0, 250.0],
        }
    )
    flight = calculate_flight_time(df, user_col="user_id", session_col="session_id")
    assert flight is not None
    # Both events are the start of their respective sessions, so both must be NaN
    assert np.isnan(flight.iloc[0])
    assert np.isnan(flight.iloc[1])


def test_calculate_pause_features():
    """Verify pause metrics calculation with configurable threshold."""
    # Latencies in ms: 50, 100, 2200 (>2.0s pause), 4500 (>4.0s extended pause)
    flight_series = pd.Series([50.0, 100.0, 2200.0, 4500.0])
    pause_res = calculate_pause_features(flight_series, pause_threshold_ms=2000.0)

    assert pause_res["pause_rate"] == 0.5  # 2 pauses out of 4
    assert pause_res["long_pause_count"] == 1.0  # 1 pause >= 4000ms
    assert pause_res["max_pause_duration"] == 4500.0


def test_calculate_typing_speed():
    """Verify WPM throughput calculation."""
    df_sample = load_dataset(SAMPLE_PATH)
    speed_feats = calculate_typing_speed(df_sample)

    assert "mean_typing_speed_wpm" in speed_feats
    assert speed_feats["mean_typing_speed_wpm"] > 0.0
    assert speed_feats["max_typing_speed_wpm"] <= 250.0


def test_calculate_backspace_features():
    """Verify backspace rate calculation."""
    df = pd.DataFrame(
        {
            "key": ["a", "b", "Key.backspace", "c", "Key.backspace"],
        }
    )
    res = calculate_backspace_features(df)
    assert res["total_backspaces"] == 2.0
    assert res["backspace_rate"] == 0.4


def test_calculate_error_features_when_absent_does_not_fabricate():
    """Verify error features return empty when dataset lacks error annotations."""
    df_no_errors = pd.DataFrame({"dwell_time": [100.0, 110.0]})
    res = calculate_error_features(df_no_errors)
    # Must NOT fabricate error rate
    assert res == {}


def test_calculate_error_features_when_present():
    """Verify error features compute properly when explicit error indicators exist."""
    df_errors = pd.DataFrame(
        {
            "error_flag": [0, 1, 0, 1],
            "dwell_time": [100.0, 110.0, 90.0, 105.0],
        }
    )
    res = calculate_error_features(df_errors)
    assert res["total_errors"] == 2.0
    assert res["error_rate"] == 0.5


def test_calculate_word_features_unavailability():
    """Verify word features are cleanly marked unavailable if word timestamps are missing."""
    df = pd.DataFrame({"dwell_time": [100.0]})
    res = calculate_word_features(df)
    assert res == {}


def test_variability_features_safe_zero_division():
    """Verify CV handles near-zero and zero mean safely without Inf or NaN."""
    zero_series = pd.Series([0.0, 0.0, 0.0])
    res = calculate_variability_features(zero_series, "dwell")

    assert not np.isnan(res["cv_dwell"])
    assert not np.isinf(res["cv_dwell"])
    assert res["cv_dwell"] == 0.0


def test_variability_features_mad_calculation():
    """Verify Median Absolute Deviation (MAD) calculation."""
    series = pd.Series([10.0, 20.0, 30.0, 40.0, 100.0])
    res = calculate_variability_features(series, "metric")
    assert res["mad_metric"] > 0.0


def test_build_and_validate_feature_table():
    """Verify building and validating session-level feature matrix on sample dataset."""
    df_raw = load_dataset(SAMPLE_PATH)
    cleaned_df, clean_report = clean_dataset_with_report(df_raw)

    feature_df, summary = build_feature_table(
        cleaned_df,
        user_col="user_id",
        session_col="session_id",
        timestamp_col="timestamp",
        label_col="state",
    )

    assert len(feature_df) == 10  # 5 users * 2 sessions = 10 sessions
    assert "mean_dwell_time" in feature_df.columns
    assert "mean_flight_time" in feature_df.columns
    assert "cv_flight_time" in feature_df.columns
    assert "target_label" in feature_df.columns

    is_valid, violations = validate_feature_table(feature_df)
    assert is_valid is True, f"Feature table validation failed: {violations}"


def test_feature_manifest_availability_and_unavailability():
    """Verify manifest accurately distinguishes available vs unavailable features."""
    sample_cols = ["user_id", "session_id", "timestamp", "dwell_time", "flight_time", "state"]
    manifest = generate_feature_manifest(sample_cols)

    available_names = [f["feature_name"] for f in manifest["available_features"]]
    unavailable_names = [f["feature_name"] for f in manifest["unavailable_features"]]

    assert "mean_dwell_time" in available_names
    assert "mean_flight_time" in available_names
    # Word completion and explicit errors are not in sample_cols
    assert "mean_word_completion_time" in unavailable_names
    assert "total_errors" in unavailable_names

    # Check reason documented for unavailable feature
    word_meta = next(f for f in manifest["unavailable_features"] if f["feature_name"] == "mean_word_completion_time")
    assert "Missing required raw columns" in word_meta["reason_if_unavailable"]


def test_sensitive_text_quarantine_in_cleaning():
    """Verify sensitive text columns are quarantined and removed before feature extraction."""
    df_with_text = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "session_id": ["s1", "s1"],
            "timestamp": [1000, 1500],
            "dwell_time": [100.0, 110.0],
            "message": ["confidential typed text", "more sensitive sentences"],
            "typed_text": ["abc", "def"],
        }
    )
    cleaned_df, report = clean_dataset_with_report(df_with_text)

    assert "message" not in cleaned_df.columns
    assert "typed_text" not in cleaned_df.columns
    assert "message" in report["sensitive_text_columns_quarantined"]
    assert "typed_text" in report["sensitive_text_columns_quarantined"]
