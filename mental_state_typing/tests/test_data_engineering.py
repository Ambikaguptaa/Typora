"""Unit tests for the data engineering subsystem."""

import pandas as pd
import pytest

from src.data_engineering.cleaner import clean_keystroke_timing
from src.data_engineering.feature_engineering import extract_timing_features
from src.data_engineering.loader import load_keystroke_dataset


def test_data_engineering_modules_importable():
    """Verify that all data engineering modules import cleanly."""
    import src.data_engineering.loader as loader
    import src.data_engineering.cleaner as cleaner
    import src.data_engineering.feature_engineering as fe
    import src.data_engineering.baseline as baseline

    assert loader is not None
    assert cleaner is not None
    assert fe is not None
    assert baseline is not None


def test_clean_keystroke_timing():
    """Verify that invalid/out-of-bounds hold and flight times are filtered."""
    dirty_data = pd.DataFrame(
        {
            "hold_time": [5.0, 120.0, 2500.0, 150.0],  # 5 is < 10, 2500 is > 2000
            "flight_time": [-20.0, 100.0, 200.0, 7000.0],  # -20 is < 0, 7000 > 5000
        }
    )

    cleaned = clean_keystroke_timing(dirty_data)
    # Only row 1 (120.0, 100.0) should survive both filters
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["hold_time"] == 120.0
    assert cleaned.iloc[0]["flight_time"] == 100.0


def test_extract_timing_features():
    """Verify calculation of hold times, flight times, and pause rate."""
    sample_df = pd.DataFrame(
        {
            "press_time": [100.0, 250.0, 450.0, 1100.0],
            "release_time": [200.0, 350.0, 520.0, 1200.0],
        }
    )

    features = extract_timing_features(sample_df, pause_threshold_ms=500.0)

    assert "mean_hold_time_ms" in features
    assert "mean_flight_time_ms" in features
    assert "pause_rate" in features
    assert features["keystroke_count"] == 4.0

    # Hold times: 100, 100, 70, 100 -> mean = 92.5
    assert pytest.approx(features["mean_hold_time_ms"], rel=1e-2) == 92.5

    # Flight times between keys:
    # 250 - 200 = 50 ms
    # 450 - 350 = 100 ms
    # 1100 - 520 = 580 ms (> 500 ms pause)
    # 1 pause out of 3 transitions = 1/3
    assert pytest.approx(features["pause_rate"], rel=1e-2) == 1.0 / 3.0


def test_load_keystroke_dataset_nonexistent_file():
    """Verify FileNotFoundError when loading a nonexistent path."""
    with pytest.raises(FileNotFoundError):
        load_keystroke_dataset("data/raw/does_not_exist.csv")
