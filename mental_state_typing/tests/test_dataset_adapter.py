"""Unit tests for the Dataset Adapter and Dataset Registry modules."""

from pathlib import Path
import pandas as pd
import pytest

from src.data_engineering.dataset_adapter import (
    analyze_class_distribution,
    analyze_duplicates,
    analyze_missing_values,
    detect_keystroke_columns,
    detect_label_column,
    detect_sensitive_text_columns,
    detect_session_column,
    detect_timestamp_column,
    detect_user_column,
    generate_dataset_report,
    load_dataset,
)
from src.data_engineering.dataset_registry import (
    DatasetConfig,
    get_dataset_config,
    list_registered_datasets,
    register_dataset,
)

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "sample" / "sample_keystrokes.csv"


def test_load_sample_dataset():
    """Verify loading the synthetic sample keystrokes dataset."""
    df = load_dataset(SAMPLE_PATH)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "user_id" in df.columns
    assert "state" in df.columns


def test_detect_columns_on_sample_dataset():
    """Verify heuristic detectors identify appropriate fields on sample dataset."""
    df = load_dataset(SAMPLE_PATH)

    user_col = detect_user_column(df)
    session_col = detect_session_column(df)
    timestamp_col = detect_timestamp_column(df)
    keystroke_cols = detect_keystroke_columns(df)
    label_col = detect_label_column(df)

    assert user_col == "user_id"
    assert session_col == "session_id"
    assert timestamp_col == "timestamp"
    assert "dwell_time" in keystroke_cols
    assert "flight_time" in keystroke_cols
    assert label_col == "state"


def test_detect_sensitive_text_columns():
    """Verify detection of sensitive message/text content columns."""
    sample_with_text = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "dwell_time": [100.0, 110.0],
            "message": ["private message text", "another secret"],
            "typed_text": ["hello", "world"],
        }
    )

    sensitive = detect_sensitive_text_columns(sample_with_text)
    assert "message" in sensitive
    assert "typed_text" in sensitive
    assert "dwell_time" not in sensitive


def test_analyze_missing_values_and_duplicates():
    """Verify missing values and duplicate analysis functions."""
    df = pd.DataFrame(
        {
            "col_a": [1.0, None, 3.0, 3.0],
            "col_b": ["x", "y", "z", "z"],
        }
    )

    missing = analyze_missing_values(df)
    assert missing.get("col_a") == 1
    assert "col_b" not in missing

    dups = analyze_duplicates(df)
    assert dups == 1


def test_analyze_class_distribution():
    """Verify class breakdown dictionary output."""
    df = pd.DataFrame({"state": ["Calm", "Calm", "Fatigued", "High_Workload"]})
    dist = analyze_class_distribution(df, "state")

    assert dist["Calm"] == 2
    assert dist["Fatigued"] == 1
    assert dist["High_Workload"] == 1


def test_generate_dataset_report():
    """Verify structured report dictionary matches expected keys and metrics."""
    df = load_dataset(SAMPLE_PATH)
    report = generate_dataset_report(df, dataset_name="Test Sample")

    assert report["dataset_name"] == "Test Sample"
    assert report["rows"] == len(df)
    assert report["columns"] == len(df.columns)
    assert report["user_column"] == "user_id"
    assert report["session_column"] == "session_id"
    assert report["label_column"] == "state"
    assert isinstance(report["class_distribution"], dict)
    assert report["missing_percentage"] == 0.0
    assert report["unique_users"] == 5


def test_dataset_registry():
    """Verify registering and retrieving dataset configs."""
    custom_cfg = DatasetConfig(
        name="academic_benchmark_2026",
        file_path=Path("data/raw/benchmark.csv"),
        user_column="subject",
        label_column="mental_effort",
        feature_columns=["iki", "dwell"],
    )

    register_dataset(custom_cfg)
    retrieved = get_dataset_config("academic_benchmark_2026")

    assert retrieved is not None
    assert retrieved.name == "academic_benchmark_2026"
    assert retrieved.user_column == "subject"
    assert "academic_benchmark_2026" in list_registered_datasets()
