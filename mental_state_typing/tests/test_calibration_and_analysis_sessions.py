"""Unit tests for calibration and analysis session flows and baseline accumulation."""

import pytest
from src.integration.behavioral_engine import BehavioralEngine
from src.integration.pipeline_state import SessionType


def _generate_valid_raw_events(count: int = 35) -> list[dict]:
    events = []
    t = 1000.0
    for i in range(count):
        dwell = 85.0 + (i % 4) * 5.0
        events.append({"event_type": "down", "timestamp_ms": t, "key_token": "k_alpha"})
        events.append({"event_type": "up", "timestamp_ms": t + dwell, "key_token": "k_alpha"})
        t += 180.0
    return events


def test_cold_start_flow_reports_not_ready_neutrally(tmp_path):
    """Verify session 1 with no baseline history reports BASELINE_NOT_READY without claiming abnormality."""
    engine = BehavioralEngine(
        user_id="cold_start_user",
        session_type=SessionType.ANALYSIS,
        models_dir=tmp_path / "models",
        baselines_dir=tmp_path / "baselines",
        assessments_dir=tmp_path / "assessments",
    )

    engine.start_session(SessionType.ANALYSIS)
    engine.session.start_time -= 5.0  # Satisfy duration threshold
    engine.ingest_raw_events(_generate_valid_raw_events(35))
    res = engine.stop_session()

    assert res.data_quality.is_valid
    assert res.baseline_result.status == "NOT_READY"
    assert res.baseline_result.session_count == 0
    assert "Minimum 5 calibration sessions required" in res.baseline_result.message
    # Assert language is neutral and does NOT label user as abnormal
    assert "abnormal" not in res.baseline_result.message.lower()


def test_calibration_session_accumulation_and_transition(tmp_path):
    """Verify 5 calibration sessions accumulate and transition baseline to READY."""
    engine = BehavioralEngine(
        user_id="calib_user_01",
        session_type=SessionType.CALIBRATION,
        min_calibration_sessions=5,
        models_dir=tmp_path / "models",
        baselines_dir=tmp_path / "baselines",
        assessments_dir=tmp_path / "assessments",
    )

    # Complete 4 calibration sessions
    for i in range(4):
        engine.start_session(SessionType.CALIBRATION)
        engine.session.start_time -= 5.0
        engine.ingest_raw_events(_generate_valid_raw_events(35))
        res = engine.stop_session()
        assert res.data_quality.is_valid
        assert res.baseline_result.status == "NOT_READY"
        assert res.baseline_result.session_count == i + 1

    # Complete 5th calibration session
    engine.start_session(SessionType.CALIBRATION)
    engine.session.start_time -= 5.0
    engine.ingest_raw_events(_generate_valid_raw_events(35))
    res5 = engine.stop_session()

    assert res5.baseline_result.status == "READY"
    assert res5.baseline_result.session_count == 5
    assert res5.baseline_result.typing_deviation_index is not None
    assert 0.0 <= res5.baseline_result.typing_deviation_index <= 100.0


def test_analysis_session_does_not_mutate_calibration_history(tmp_path):
    """Verify analysis sessions do not accidentally append to calibration history."""
    engine = BehavioralEngine(
        user_id="analysis_user",
        session_type=SessionType.ANALYSIS,
        models_dir=tmp_path / "models",
        baselines_dir=tmp_path / "baselines",
        assessments_dir=tmp_path / "assessments",
    )

    engine.start_session(SessionType.ANALYSIS)
    engine.session.start_time -= 5.0
    engine.ingest_raw_events(_generate_valid_raw_events(35))
    engine.stop_session()

    # History count should remain 0
    assert len(engine.user_calibration_history) == 0
