"""Tests for Report Listing, Markdown/Text report formatting, and export integrity."""

import json
from pathlib import Path
import pytest

from src.assessment.report_generator import generate_markdown_report, generate_text_report
from src.integration.result_schema import (
    BaselineResult,
    BehavioralAssessmentResult,
    DataQualityResult,
    FeatureSummary,
    ModelResult,
)
from src.integration.session_history import list_stored_sessions, load_session_detail
from src.live_typing.privacy_filter import PrivacyViolationError


@pytest.fixture
def sample_assessment() -> BehavioralAssessmentResult:
    """Fixture providing a standard valid assessment result."""
    return BehavioralAssessmentResult(
        session_id="sess_report_alpha",
        session_type="ANALYSIS",
        timestamp="2026-09-20T15:30:00Z",
        pipeline_status="ASSESSMENT_COMPLETE",
        data_quality=DataQualityResult(
            is_valid=True,
            verdict="PASS",
            reasons=[],
            metrics={"active_duration_s": 14.2, "paired_events_count": 28},
        ),
        feature_summary=FeatureSummary(
            mean_dwell_ms=92.4,
            std_dwell_ms=11.2,
            mean_flight_ms=121.5,
            std_flight_ms=19.4,
            pause_count=1,
            pause_rate=0.04,
            estimated_wpm=48.2,
            backspace_count=1,
            sequence_windows_count=1,
        ),
        baseline_result=BaselineResult(
            status="READY",
            session_count=5,
            min_required_sessions=5,
            message="Individual baseline established.",
            typing_deviation_index=28.5,
        ),
        model_result=ModelResult(
            status="MODEL_NOT_READY",
            reason="Approved real research dataset required.",
        ),
    )


def test_report_listing_empty(tmp_path: Path):
    """Verify that an empty directory lists zero reports without raising errors."""
    sessions = list_stored_sessions(assessments_dir=tmp_path)
    assert len(sessions) == 0


def test_report_listing_and_markdown_generation(tmp_path: Path, sample_assessment: BehavioralAssessmentResult):
    """Verify listing a saved assessment and generating Markdown and plain-text reports."""
    file_path = tmp_path / "assessment_sess_report_alpha.json"
    sample_assessment.save_json(file_path)

    sessions = list_stored_sessions(assessments_dir=tmp_path)
    assert len(sessions) == 1

    detail = load_session_detail("sess_report_alpha", assessments_dir=tmp_path)
    assert detail is not None

    assessment_obj = BehavioralAssessmentResult.from_dict(detail)
    md_report = generate_markdown_report(assessment_obj)
    txt_report = generate_text_report(assessment_obj)

    assert "academic research instrument" in md_report.lower()
    assert "sess_report_alpha" in md_report
    assert "typing deviation index" in md_report.lower()
    assert "model_not_ready" in md_report.lower()

    assert "academic research instrument" in txt_report.lower()
    assert "sess_report_alpha" in txt_report


def test_report_blocked_on_sensitive_key_injection(tmp_path: Path):
    """Verify that attempting to load a report containing injected raw text raises PrivacyViolationError."""
    tainted_file = tmp_path / "assessment_sess_tainted.json"
    tainted_payload = {
        "session_id": "sess_tainted",
        "keystring": "SecretPassword123",
        "timestamp": "2026-09-20T16:00:00Z",
    }
    tainted_file.write_text(json.dumps(tainted_payload), encoding="utf-8")

    with pytest.raises(PrivacyViolationError):
        load_session_detail("sess_tainted", assessments_dir=tmp_path)
