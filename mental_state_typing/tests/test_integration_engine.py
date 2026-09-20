"""Unit tests for the BehavioralEngine integration and pipeline state orchestration."""

import time
import pytest
from src.integration.behavioral_engine import BehavioralEngine
from src.integration.pipeline_state import PipelineState, SessionType
from src.integration.result_schema import BehavioralAssessmentResult


def test_engine_initial_state(tmp_path):
    """Verify engine initializes in IDLE state with default parameters."""
    engine = BehavioralEngine(
        user_id="test_user",
        session_type=SessionType.ANALYSIS,
        models_dir=tmp_path / "models",
        baselines_dir=tmp_path / "baselines",
        assessments_dir=tmp_path / "assessments",
    )
    assert engine.state == PipelineState.IDLE
    assert engine.session_type == SessionType.ANALYSIS
    assert engine.session.session_id.startswith("sess_")
    assert engine.feature_buffer.event_count == 0


def test_engine_state_transitions(tmp_path):
    """Verify state transitions: IDLE -> CAPTURING -> COLLECTING_DATA -> IDLE on reset."""
    engine = BehavioralEngine(
        user_id="test_user",
        models_dir=tmp_path / "models",
        baselines_dir=tmp_path / "baselines",
        assessments_dir=tmp_path / "assessments",
    )

    # Start
    engine.start_session(SessionType.CALIBRATION)
    assert engine.state == PipelineState.CAPTURING
    assert engine.session_type == SessionType.CALIBRATION

    # Ingest event batch
    batch = [
        {"event_type": "down", "timestamp_ms": 100.0, "key_token": "k_alpha"},
        {"event_type": "up", "timestamp_ms": 180.0, "key_token": "k_alpha"},
    ]
    added = engine.ingest_raw_events(batch)
    assert added == 1
    assert engine.state == PipelineState.COLLECTING_DATA

    # Reset
    engine.reset_session()
    assert engine.state == PipelineState.IDLE
    assert engine.feature_buffer.event_count == 0


def test_engine_quality_gate_failure(tmp_path):
    """Verify insufficient data transitions to QUALITY_CHECK_FAILED and halts inference."""
    engine = BehavioralEngine(
        user_id="test_user",
        models_dir=tmp_path / "models",
        baselines_dir=tmp_path / "baselines",
        assessments_dir=tmp_path / "assessments",
    )

    engine.start_session()
    # Ingest only 2 events (well below technical quality threshold of 15)
    batch = [
        {"event_type": "down", "timestamp_ms": 100.0, "key_token": "k_alpha"},
        {"event_type": "up", "timestamp_ms": 180.0, "key_token": "k_alpha"},
    ]
    engine.ingest_raw_events(batch)

    res = engine.stop_session()
    assert isinstance(res, BehavioralAssessmentResult)
    assert engine.state == PipelineState.QUALITY_CHECK_FAILED
    assert res.data_quality.verdict == "FAIL"
    assert not res.data_quality.is_valid
    assert len(res.data_quality.reasons) > 0
    # Model should report MODEL_NOT_READY and no prediction
    assert res.model_result.predicted_class is None
