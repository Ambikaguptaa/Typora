"""Unit tests for generic real dataset validator module."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data_engineering.real_dataset_validator import validate_real_dataset


@pytest.fixture
def valid_research_df():
    """Create a synthetic fixture representing a valid research dataset with 4 participants."""
    rows = []
    conditions = ["neutral", "stressed"]
    for user_idx in range(1, 5):
        uid = f"participant_{user_idx:02d}"
        for sess_idx in range(1, 5):
            sid = f"sess_{sess_idx:02d}"
            cond = conditions[sess_idx % 2]
            for event_idx in range(25):
                t_press = float(event_idx * 200 + sess_idx * 10000)
                t_release = t_press + 90.0  # 90ms dwell time
                rows.append(
                    {
                        "participant_id": uid,
                        "session_id": sid,
                        "press_time": t_press,
                        "release_time": t_release,
                        "dwell_time": 90.0,
                        "flight_time": 110.0,
                        "condition": cond,
                    }
                )
    return pd.DataFrame(rows)


def test_validate_real_dataset_missing(tmp_path):
    """Verify validator handles missing or empty dataset sources gracefully."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    report = validate_real_dataset(source=empty_dir, save_reports=False)

    assert report["is_valid"] is False
    assert report["status"] == "missing"
    assert "REAL DATASET REQUIRED" in report["message"]


def test_validate_real_dataset_valid(valid_research_df):
    """Verify validator approves a fully compliant research dataset."""
    report = validate_real_dataset(source=valid_research_df, save_reports=False)

    assert report["is_valid"] is True
    assert report["status"] == "validated"
    assert report["total_participants"] == 4
    assert report["total_sessions"] == 16
    assert len(report["labels"]["unique_classes"]) == 2
    assert "neutral" in report["labels"]["unique_classes"]
    assert "stressed" in report["labels"]["unique_classes"]
    assert report["data_quality"]["duplicate_records"] == 0
    assert report["data_quality"]["timestamp_monotonic"] is True
    assert report["leakage_audit_readiness"]["passed"] is True


def test_validate_real_dataset_missing_labels(valid_research_df):
    """Verify validator rejects a dataset missing condition labels."""
    unlabeled_df = valid_research_df.drop(columns=["condition"])
    report = validate_real_dataset(source=unlabeled_df, save_reports=False)

    assert report["is_valid"] is False
    assert report["status"] == "blocked_missing_labels"
    assert report["labels"]["detected"] is False


def test_validate_real_dataset_single_class_rejected(valid_research_df):
    """Verify validator rejects a dataset where all rows have only a single condition class."""
    single_class_df = valid_research_df.copy()
    single_class_df["condition"] = "neutral"
    report = validate_real_dataset(source=single_class_df, save_reports=False)

    assert report["is_valid"] is False
    assert report["status"] == "blocked_missing_labels"
    assert len(report["labels"]["unique_classes"]) == 1


def test_validate_real_dataset_per_participant_breakdown(valid_research_df):
    """Verify participant breakdown captures session counts and class distribution per user."""
    report = validate_real_dataset(source=valid_research_df, save_reports=False)
    p_info = report["participant_validation"]["participant_breakdown"]

    assert len(p_info) == 4
    for uid, stats in p_info.items():
        assert stats["session_count"] == 4
        assert stats["record_count"] == 100
        assert "neutral" in stats["condition_distribution"]
        assert "stressed" in stats["condition_distribution"]


def test_validate_real_dataset_sensitive_text_quarantine(valid_research_df):
    """Verify validator flags and excludes sensitive text columns."""
    contaminated_df = valid_research_df.copy()
    contaminated_df["typed_text"] = "sample text"
    contaminated_df["password"] = "secret123"

    report = validate_real_dataset(source=contaminated_df, save_reports=False)
    assert report["privacy"]["zero_text_compliant"] is False
    assert "typed_text" in report["privacy"]["sensitive_fields_excluded"]
    assert "password" in report["privacy"]["sensitive_fields_excluded"]


def test_validate_real_dataset_timing_anomaly_detection(valid_research_df):
    """Verify validator counts physiologically implausible dwell times."""
    anomaly_df = valid_research_df.copy()
    anomaly_df.loc[0, "dwell_time"] = 9999.0  # Excessive hold time > 4000ms
    anomaly_df.loc[1, "dwell_time"] = 2.0     # Sub-physiological hold < 10ms

    report = validate_real_dataset(source=anomaly_df, save_reports=False)
    assert report["data_quality"]["invalid_timing_count"] >= 2


def test_validate_real_dataset_saves_reports(valid_research_df, tmp_path):
    """Verify validator persists JSON reports to processed directory."""
    report = validate_real_dataset(source=valid_research_df, save_reports=True, output_dir=tmp_path)
    report_file = tmp_path / "real_dataset_validation_report.json"
    label_file = tmp_path / "label_metadata.json"

    assert report_file.exists()
    assert label_file.exists()

    saved_data = json.loads(report_file.read_text(encoding="utf-8"))
    assert saved_data["is_valid"] is True


