"""Tests for Session History management, loading, auditing, and error handling."""

import json
from pathlib import Path
import pytest

from src.integration.result_schema import (
    BaselineResult,
    BehavioralAssessmentResult,
    DataQualityResult,
    FeatureSummary,
    ModelResult,
)
from src.integration.session_history import (
    delete_session_record,
    list_stored_sessions,
    load_session_detail,
)
from src.live_typing.privacy_filter import PrivacyViolationError


@pytest.fixture
def temp_assessments_dir(tmp_path: Path) -> Path:
    """Create a temporary assessments directory for test isolation."""
    d = tmp_path / "assessments"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_empty_session_history(temp_assessments_dir: Path):
    """Verify that an empty directory returns an empty list without error."""
    sessions = list_stored_sessions(assessments_dir=temp_assessments_dir)
    assert isinstance(sessions, list)
    assert len(sessions) == 0


def test_session_history_loading_valid_assessment(temp_assessments_dir: Path):
    """Verify that valid assessment records are properly parsed and summarized."""
    assessment = BehavioralAssessmentResult(
        session_id="sess_test_12345",
        session_type="ANALYSIS",
        timestamp="2026-09-20T12:00:00Z",
        pipeline_status="ASSESSMENT_COMPLETE",
        data_quality=DataQualityResult(
            is_valid=True,
            verdict="PASS",
            reasons=[],
            metrics={"active_duration_s": 12.5, "paired_events_count": 25},
        ),
        feature_summary=FeatureSummary(
            mean_dwell_ms=95.0,
            std_dwell_ms=12.0,
            mean_flight_ms=115.0,
            std_flight_ms=18.0,
            pause_count=2,
            pause_rate=0.08,
            estimated_wpm=52.0,
            backspace_count=1,
            sequence_windows_count=1,
        ),
        baseline_result=BaselineResult(
            status="READY",
            session_count=5,
            min_required_sessions=5,
            message="Baseline established.",
            typing_deviation_index=34.2,
        ),
        model_result=ModelResult(
            status="MODEL_NOT_READY",
            reason="Awaiting research dataset.",
        ),
    )

    file_path = temp_assessments_dir / "assessment_sess_test_12345.json"
    assessment.save_json(file_path)

    sessions = list_stored_sessions(assessments_dir=temp_assessments_dir)
    assert len(sessions) == 1
    s = sessions[0]
    assert s["session_id"] == "sess_test_12345"
    assert s["session_type"] == "ANALYSIS"
    assert s["duration_s"] == 12.5
    assert s["event_count"] == 25
    assert s["quality"] == "VALID"
    assert s["baseline_status"] == "READY"
    assert s["tdi"] == 34.2
    assert s["model_status"] == "MODEL_NOT_READY"
    assert not s["is_corrupted"]


def test_corrupted_report_handling(temp_assessments_dir: Path):
    """Verify that malformed or non-JSON files do not crash the session listing."""
    corrupted_file = temp_assessments_dir / "assessment_sess_corrupt.json"
    corrupted_file.write_text("NOT_JSON { malformed content [[[", encoding="utf-8")

    sessions = list_stored_sessions(assessments_dir=temp_assessments_dir)
    assert len(sessions) == 1
    s = sessions[0]
    assert s["session_id"] == "sess_corrupt"
    assert s["is_corrupted"] is True
    assert s["quality"] == "CORRUPTED"
    assert "Malformed JSON" in s["error_message"]


def test_privacy_violating_report_flagged(temp_assessments_dir: Path):
    """Verify that a report containing forbidden raw text keys is flagged and blocked."""
    leaky_file = temp_assessments_dir / "assessment_sess_leaky.json"
    leaky_payload = {
        "session_id": "sess_leaky",
        "typed_text": "Sensitive message password",
        "data_quality": {"is_valid": True},
    }
    leaky_file.write_text(json.dumps(leaky_payload), encoding="utf-8")

    sessions = list_stored_sessions(assessments_dir=temp_assessments_dir)
    assert len(sessions) == 1
    s = sessions[0]
    assert s["session_id"] == "sess_leaky"
    assert s["is_corrupted"] is True
    assert s["quality"] == "BLOCKED"
    assert "forbidden raw text" in s["error_message"]


def test_load_session_detail_and_deletion(temp_assessments_dir: Path):
    """Verify loading detail of a specific session and deleting the file."""
    assessment = BehavioralAssessmentResult(
        session_id="sess_delete_me",
        session_type="CALIBRATION",
        timestamp="2026-09-20T14:00:00Z",
        pipeline_status="ASSESSMENT_COMPLETE",
        data_quality=DataQualityResult(is_valid=True, verdict="PASS"),
        feature_summary=FeatureSummary(),
        baseline_result=BaselineResult(status="NOT_READY", session_count=1),
        model_result=ModelResult(status="MODEL_NOT_READY"),
    )
    assessment.save_json(temp_assessments_dir / "assessment_sess_delete_me.json")

    detail = load_session_detail("sess_delete_me", assessments_dir=temp_assessments_dir)
    assert detail is not None
    assert detail["session_id"] == "sess_delete_me"

    # Delete record
    deleted = delete_session_record("sess_delete_me", assessments_dir=temp_assessments_dir)
    assert deleted is True

    # Confirm it no longer exists
    assert load_session_detail("sess_delete_me", assessments_dir=temp_assessments_dir) is None
