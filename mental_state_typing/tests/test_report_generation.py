"""Unit tests for structured report generation and JSON persistence safety."""

import json
from pathlib import Path
import pytest

from src.assessment.report_generator import generate_markdown_report, generate_text_report
from src.integration.behavioral_engine import BehavioralEngine
from src.integration.pipeline_state import SessionType
from src.live_typing.privacy_filter import FORBIDDEN_PAYLOAD_FIELDS, audit_payload_for_sensitive_keys


def _generate_valid_raw_events(count: int = 35) -> list[dict]:
    events = []
    t = 1000.0
    for i in range(count):
        dwell = 85.0
        events.append({"event_type": "down", "timestamp_ms": t, "key_token": "k_alpha"})
        events.append({"event_type": "up", "timestamp_ms": t + dwell, "key_token": "k_alpha"})
        t += 180.0
    return events


def test_markdown_and_text_report_generation(tmp_path):
    """Verify structured report contains all required sections and adheres to Zero-Text policy."""
    engine = BehavioralEngine(
        user_id="report_test_user",
        session_type=SessionType.ANALYSIS,
        models_dir=tmp_path / "models",
        baselines_dir=tmp_path / "baselines",
        assessments_dir=tmp_path / "assessments",
    )

    engine.start_session()
    engine.session.start_time -= 5.0
    engine.ingest_raw_events(_generate_valid_raw_events(35))
    res = engine.stop_session()

    md_report = generate_markdown_report(res)
    text_report = generate_text_report(res)

    assert "Behavioral Session Intelligence Report" in md_report
    assert "Technical Data Quality" in md_report
    assert "Personal Typing Baseline" in md_report
    assert "Deep Learning Sequence Model" in md_report
    assert "MODEL_NOT_READY" in md_report
    assert "Fine-Motor Dynamics Telemetry" in md_report
    assert "Zero-Text Policy" in md_report

    # Plaintext report verification
    assert len(text_report) > 100
    assert "Technical Data Quality" in text_report


def test_json_report_persistence_and_privacy_safety(tmp_path):
    """Verify saved JSON assessment file is valid and contains zero forbidden textual fields."""
    engine = BehavioralEngine(
        user_id="json_report_user",
        session_type=SessionType.ANALYSIS,
        models_dir=tmp_path / "models",
        baselines_dir=tmp_path / "baselines",
        assessments_dir=tmp_path / "assessments",
    )

    engine.start_session()
    engine.session.start_time -= 5.0
    engine.ingest_raw_events(_generate_valid_raw_events(35))
    res = engine.stop_session()

    output_file = tmp_path / "assessments" / f"assessment_{res.session_id}.json"
    assert output_file.exists()

    with open(output_file, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)

    assert loaded_json["session_id"] == res.session_id
    assert loaded_json["pipeline_status"] == res.pipeline_status
    assert loaded_json["privacy_status"] == "ENFORCED"

    # Audit for forbidden keys
    violations = audit_payload_for_sensitive_keys(loaded_json)
    assert len(violations) == 0, f"Privacy violations found in persisted JSON: {violations}"
