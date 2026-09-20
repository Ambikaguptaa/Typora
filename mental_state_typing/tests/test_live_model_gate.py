"""Unit tests for live typing pipeline model readiness and personal baseline gating."""

import pytest
from src.live_typing.live_pipeline import LiveTypingPipeline


def test_model_gate_reports_not_ready_without_real_dataset():
    """Verify model readiness gate strictly returns MODEL_NOT_READY and never fabricates predictions."""
    pipeline = LiveTypingPipeline(user_id="usr_gate_test")
    model_readiness = pipeline.get_model_readiness()

    assert model_readiness["status"] == "MODEL_NOT_READY"
    assert not model_readiness["ready"]
    assert "No verified production model trained on real research data" in model_readiness["reason"]
    assert "disclaimer" in model_readiness


def test_baseline_gate_insufficient_history():
    """Verify baseline readiness returns BASELINE_NOT_READY when historical sessions < 5."""
    pipeline = LiveTypingPipeline(user_id="usr_calib_test", min_calibration_sessions=5)
    gate = pipeline.get_baseline_readiness()

    assert gate["status"] == "BASELINE_NOT_READY"
    assert not gate["ready"]
    assert gate["completed_sessions"] == 0
    assert gate["required_sessions"] == 5
    assert "Minimum 5 calibration sessions required" in gate["reason"]


def test_baseline_gate_transitions_to_ready():
    """Verify baseline readiness transitions to BASELINE_READY once 5 sessions are accumulated."""
    pipeline = LiveTypingPipeline(user_id="usr_calib_test", min_calibration_sessions=5)

    # Mock 5 valid sessions into user_session_history
    for i in range(5):
        pipeline.user_session_history.append({
            "session_id": f"sess_{i:02d}",
            "user_id": "usr_calib_test",
            "mean_dwell_time": 90.0 + i,
            "mean_flight_time": 120.0 + i,
            "mean_typing_speed_wpm": 45.0,
            "backspace_rate": 0.05,
            "pause_rate": 0.1,
            "mean_pause_duration": 600.0,
            "cv_flight_time": 0.25,
            "cv_dwell_time": 0.2,
            "error_rate": 0.05,
        })

    gate = pipeline.get_baseline_readiness()
    assert gate["status"] == "BASELINE_READY"
    assert gate["ready"]
    assert gate["completed_sessions"] == 5
    assert "baseline_profile" in gate


def test_evaluate_live_session_gating_integrity():
    """Verify evaluate_live_session refuses to produce fake predictions when gates are locked."""
    pipeline = LiveTypingPipeline(user_id="usr_eval_test")

    # Empty session -> INSUFFICIENT_DATA
    res = pipeline.evaluate_live_session()
    assert res["status"] == "INSUFFICIENT_DATA"
    assert "disclaimer" in res
