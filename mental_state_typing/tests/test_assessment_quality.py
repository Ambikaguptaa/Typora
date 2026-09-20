"""Unit tests for assessment data quality gate and state validation (quality_checks.py)."""

import numpy as np
import pytest

from src.assessment.quality_checks import (
    STATE_BASELINE_AVAILABLE,
    STATE_INSUFFICIENT_DATA,
    STATE_MODEL_UNAVAILABLE,
    STATE_NOT_READY,
    STATE_PREDICTION_AVAILABLE,
    STATE_READY,
    validate_assessment_data,
)


def test_quality_gate_insufficient_keystrokes():
    """Verify that fewer than 5 keystrokes blocks assessment with STATE_INSUFFICIENT_DATA."""
    report = validate_assessment_data(keystroke_count=3, min_keystrokes=5)

    assert report["assessment_ready"] is False
    assert report["assessment_state"] == STATE_INSUFFICIENT_DATA
    assert any("Insufficient keystroke observations" in r for r in report["blocking_reasons"])


def test_quality_gate_nan_and_inf_detection():
    """Verify that sequence tensors containing NaNs or Infs are rejected with STATE_NOT_READY."""
    seq_nan = np.random.randn(20, 6)
    seq_nan[0, 0] = np.nan

    report_nan = validate_assessment_data(keystroke_count=50, sequence=seq_nan)
    assert report_nan["assessment_ready"] is False
    assert report_nan["assessment_state"] == STATE_NOT_READY
    assert any("NaN" in r for r in report_nan["blocking_reasons"])

    seq_inf = np.random.randn(20, 6)
    seq_inf[1, 1] = np.inf

    report_inf = validate_assessment_data(keystroke_count=50, sequence=seq_inf)
    assert report_inf["assessment_ready"] is False
    assert any("infinite" in r for r in report_inf["blocking_reasons"])


def test_quality_gate_dimension_and_shape_mismatches():
    """Verify detection of sequence length and feature count mismatches."""
    seq = np.random.randn(15, 4).astype(np.float32)

    report = validate_assessment_data(
        keystroke_count=30,
        sequence=seq,
        expected_sequence_length=30,  # actual is 15
        expected_features=["f1", "f2", "f3", "f4", "f5", "f6"],  # actual is 4
    )

    assert report["assessment_ready"] is False
    assert report["assessment_state"] == STATE_NOT_READY
    assert any("Sequence length mismatch" in r for r in report["blocking_reasons"])
    assert any("Feature count mismatch" in r for r in report["blocking_reasons"])


def test_quality_gate_missing_required_features():
    """Verify that missing feature names are caught."""
    incoming_feats = ["dwell_time", "flight_time"]
    expected_feats = ["dwell_time", "flight_time", "pause_duration"]

    report = validate_assessment_data(
        keystroke_count=20,
        feature_names=incoming_feats,
        expected_features=expected_feats,
    )

    assert report["assessment_ready"] is False
    assert any("pause_duration" in r for r in report["blocking_reasons"])


def test_quality_gate_state_ready():
    """Verify STATE_READY when both model and baseline are available."""
    report = validate_assessment_data(
        keystroke_count=50,
        model_available=True,
        baseline_status="available",
    )

    assert report["assessment_ready"] is True
    assert report["assessment_state"] == STATE_READY
    assert len(report["blocking_reasons"]) == 0


def test_quality_gate_state_prediction_available():
    """Verify STATE_PREDICTION_AVAILABLE when model is ready but baseline is cold-start."""
    report = validate_assessment_data(
        keystroke_count=50,
        model_available=True,
        baseline_status="insufficient_history",
    )

    assert report["assessment_ready"] is True
    assert report["assessment_state"] == STATE_PREDICTION_AVAILABLE
    assert len(report["warnings"]) > 0


def test_quality_gate_state_baseline_available():
    """Verify STATE_BASELINE_AVAILABLE when personal baseline is ready but model is not."""
    report = validate_assessment_data(
        keystroke_count=50,
        model_available=False,
        baseline_status="available",
    )

    assert report["assessment_ready"] is True
    assert report["assessment_state"] == STATE_BASELINE_AVAILABLE
    assert len(report["warnings"]) > 0


def test_quality_gate_state_model_unavailable():
    """Verify STATE_MODEL_UNAVAILABLE when neither model nor baseline is ready."""
    report = validate_assessment_data(
        keystroke_count=50,
        model_available=False,
        baseline_status="unavailable",
    )

    assert report["assessment_ready"] is False
    assert report["assessment_state"] == STATE_MODEL_UNAVAILABLE
