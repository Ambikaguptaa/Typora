"""Tests for Dashboard backend data aggregation, training gate matrix, and session reset."""

import pytest

from src.integration.behavioral_engine import BehavioralEngine
from src.integration.pipeline_state import PipelineState, SessionType
from src.integration.system_status import (
    get_privacy_security_specs,
    get_system_overview_status,
    get_training_gate_matrix,
)


def test_system_overview_status_structure():
    """Verify that get_system_overview_status returns all required telemetry categories."""
    engine = BehavioralEngine()
    status = get_system_overview_status(engine)

    assert "dataset" in status
    assert "model" in status
    assert "baseline" in status
    assert "live_capture" in status
    assert "privacy" in status
    assert "leakage" in status

    # Verify model is NOT reported as ready when un-trained
    assert status["model"]["badge"] == "NOT READY"
    assert status["model"]["is_ready"] is False
    assert "MODEL_NOT_READY" in status["model"]["label"] or "MODEL NOT READY" in status["model"]["label"]


def test_training_gate_matrix_criteria():
    """Verify that get_training_gate_matrix reports exactly 13 empirical criteria."""
    matrix = get_training_gate_matrix()
    assert isinstance(matrix, list)
    assert len(matrix) == 13

    criteria_names = [m["criterion"] for m in matrix]
    assert "Dataset Available" in criteria_names
    assert "Schema Valid" in criteria_names
    assert "Labels Valid" in criteria_names
    assert "Participants Valid" in criteria_names
    assert "Feature Compatibility" in criteria_names
    assert "Privacy Audit" in criteria_names
    assert "Participant Leakage" in criteria_names
    assert "Session Leakage" in criteria_names
    assert "Window Leakage" in criteria_names
    assert "Scaler Readiness" in criteria_names
    assert "Training Split" in criteria_names
    assert "Validation Split" in criteria_names
    assert "Test Split" in criteria_names

    # Check that status is one of the approved states
    valid_statuses = {"PASS", "FAIL", "BLOCKED", "PENDING"}
    for m in matrix:
        assert m["status"] in valid_statuses


def test_privacy_security_specs_disclosures():
    """Verify that get_privacy_security_specs returns comprehensive disclosures."""
    specs = get_privacy_security_specs()
    assert "controls" in specs
    assert "what_is_collected" in specs
    assert "what_is_not_collected" in specs

    assert len(specs["controls"]) >= 7
    assert len(specs["what_is_collected"]) >= 5
    assert len(specs["what_is_not_collected"]) >= 5

    # Check for zero text invariant
    control_names = [c["name"] for c in specs["controls"]]
    assert "Zero Raw Text Invariant" in control_names
    assert "Zero Global Keyboard Hooks" in control_names


def test_session_reset_logic():
    """Verify that engine session reset properly clears state and feature buffers."""
    engine = BehavioralEngine()
    engine.start_session(SessionType.ANALYSIS)
    assert engine.state == PipelineState.CAPTURING

    # Ingest synthetic events
    sample_events = [
        {"timestamp": 100.0, "type": "down", "is_backspace": False, "is_enter": False, "is_space": False, "key_token": "k_alpha"},
        {"timestamp": 180.0, "type": "up", "is_backspace": False, "is_enter": False, "is_space": False, "key_token": "k_alpha"},
    ]
    engine.ingest_raw_events(sample_events)
    assert len(engine.session.get_paired_events()) == 1

    # Reset
    new_sid = engine.reset_session()
    assert engine.state == PipelineState.IDLE
    assert len(engine.session.get_paired_events()) == 0
    assert len(engine.feature_buffer.to_dataframe()) == 0
    assert engine.last_result is None
