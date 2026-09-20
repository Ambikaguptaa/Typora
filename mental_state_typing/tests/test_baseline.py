"""Unit tests for the baseline heuristic strain estimation module."""

import pytest

from src.data_engineering.baseline import compute_baseline_strain


def test_baseline_module_importable():
    """Verify that baseline module and helper functions import."""
    import src.data_engineering.baseline as baseline

    assert hasattr(baseline, "compute_baseline_strain")


def test_compute_baseline_strain_low_data():
    """Verify handling when keystroke count is insufficient."""
    low_data_features = {
        "keystroke_count": 3,
        "mean_hold_time_ms": 100.0,
        "pause_rate": 0.1,
    }

    result = compute_baseline_strain(low_data_features)
    assert result["strain_score"] == 0.0
    assert result["strain_category"] == "Insufficient Data"
    assert result["is_medical_diagnosis"] is False


def test_compute_baseline_strain_normal_range():
    """Verify strain scoring output with realistic feature parameters."""
    features = {
        "keystroke_count": 25,
        "mean_hold_time_ms": 120.0,
        "std_flight_time_ms": 50.0,
        "pause_rate": 0.05,
    }

    result = compute_baseline_strain(features)
    assert 0.0 <= result["strain_score"] <= 100.0
    assert "strain_category" in result
    assert result["is_medical_diagnosis"] is False


def test_compute_baseline_strain_high_hesitation():
    """Verify that elevated pauses and high variance yield a higher strain score."""
    calm_features = {
        "keystroke_count": 30,
        "mean_hold_time_ms": 90.0,
        "std_flight_time_ms": 30.0,
        "pause_rate": 0.02,
    }
    strained_features = {
        "keystroke_count": 30,
        "mean_hold_time_ms": 220.0,
        "std_flight_time_ms": 280.0,
        "pause_rate": 0.35,
    }

    calm_result = compute_baseline_strain(calm_features)
    strained_result = compute_baseline_strain(strained_features)

    assert strained_result["strain_score"] > calm_result["strain_score"]
