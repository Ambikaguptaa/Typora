"""Unit tests for the Data Quality module."""

from pathlib import Path
import pandas as pd
import pytest

from src.data_engineering.data_quality import (
    check_class_imbalance,
    check_duplicates,
    check_invalid_timestamps,
    check_missing_values,
    check_negative_durations,
    check_numeric_anomalies,
    generate_quality_summary,
)
from src.data_engineering.dataset_adapter import load_dataset

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "sample" / "sample_keystrokes.csv"


def test_missing_values_report():
    """Verify missing values report structure and accuracy."""
    df = pd.DataFrame(
        {
            "a": [1.0, None, 3.0, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
        }
    )

    report_df = check_missing_values(df)
    assert isinstance(report_df, pd.DataFrame)
    row_a = report_df[report_df["column"] == "a"].iloc[0]
    assert row_a["missing_count"] == 1
    assert row_a["missing_percentage"] == 25.0


def test_duplicate_detection():
    """Verify duplicate row detection logic."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 2, 3],
            "val": ["x", "y", "y", "z"],
        }
    )

    dup_res = check_duplicates(df)
    assert dup_res["duplicate_count"] == 1
    assert dup_res["duplicate_percentage"] == 25.0
    assert dup_res["has_duplicates"] is True


def test_negative_durations_clean_sample():
    """Verify that sample dataset contains no negative timing durations."""
    df = load_dataset(SAMPLE_PATH)
    res = check_negative_durations(df)

    assert res["has_negative_durations"] is False
    assert res["total_negative_durations"] == 0


def test_negative_durations_flagging():
    """Verify detection of negative dwell and flight times."""
    dirty_df = pd.DataFrame(
        {
            "dwell_time": [100.0, -25.0, 80.0],
            "flight_time": [-10.0, 150.0, 90.0],
        }
    )

    res = check_negative_durations(dirty_df)
    assert res["has_negative_durations"] is True
    assert res["violations_by_column"]["dwell_time"] == 1
    assert res["violations_by_column"]["flight_time"] == 1
    assert res["total_negative_durations"] == 2


def test_invalid_timestamps_detection():
    """Verify detection of negative or null timestamps."""
    df_valid = pd.DataFrame({"timestamp": [1000, 1050, 1120]})
    res_valid = check_invalid_timestamps(df_valid, "timestamp")
    assert res_valid["is_valid"] is True

    df_invalid = pd.DataFrame({"timestamp": [1000, -5, None, 1200]})
    res_invalid = check_invalid_timestamps(df_invalid, "timestamp")
    assert res_invalid["is_valid"] is False
    assert res_invalid["negative_timestamps"] == 1
    assert res_invalid["null_timestamps"] == 1


def test_class_imbalance_metrics():
    """Verify class balance analysis and imbalance ratio."""
    df = pd.DataFrame(
        {
            "state": ["Calm"] * 10 + ["Stressed"] * 2,
        }
    )

    res = check_class_imbalance(df, "state")
    assert res["status"] == "evaluated"
    assert res["class_counts"]["Calm"] == 10
    assert res["class_counts"]["Stressed"] == 2
    assert res["imbalance_ratio"] == 5.0
    assert res["is_imbalanced"] is True


def test_quality_summary_scoring():
    """Verify composite quality score calculation on sample dataset."""
    df = load_dataset(SAMPLE_PATH)
    summary = generate_quality_summary(df)

    assert summary["quality_score"] >= 85.0
    assert summary["status"] == "EXCELLENT"
    assert summary["status_badge"] == "positive"
    assert summary["negative_durations"] == 0
