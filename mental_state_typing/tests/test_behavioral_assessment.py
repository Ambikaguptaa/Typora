"""Integration tests for the complete behavioral assessment pipeline (behavioral_assessment.py)."""

import json
from pathlib import Path
import pytest

from src.assessment.behavioral_assessment import (
    AssessmentInput,
    generate_assessment_summary,
    generate_behavioral_assessment,
    save_assessment_report,
)
from src.assessment.quality_checks import (
    STATE_BASELINE_AVAILABLE,
    STATE_INSUFFICIENT_DATA,
    STATE_PREDICTION_AVAILABLE,
    STATE_READY,
)


def test_assessment_input_dataclass_serialization():
    """Verify AssessmentInput serialization and dictionary round-trip."""
    input_obj = AssessmentInput(
        user_id="usr_test_12345",
        session_id="sess_001",
        keystroke_count=60,
        predicted_class="Calm",
        class_probabilities={"Calm": 0.80, "Fatigued": 0.20},
        model_available=True,
        is_mock=True,
    )

    data_dict = input_obj.to_dict()
    assert data_dict["user_id"] == "usr_test_12345"
    assert data_dict["keystroke_count"] == 60
    assert data_dict["is_mock"] is True

    reconstituted = AssessmentInput.from_dict(data_dict)
    assert reconstituted.user_id == input_obj.user_id
    assert reconstituted.session_id == input_obj.session_id
    assert reconstituted.predicted_class == input_obj.predicted_class


def test_complete_assessment_pipeline_integration(tmp_path: Path):
    """Verify complete assessment generation chain with mock model output and baseline."""
    # 1. Mock inputs
    mock_baseline_result = {
        "user_id": "usr_9f83ac01",
        "baseline_status": "available",
        "typing_deviation_index": 64.5,
        "features": {
            "mean_dwell_time": {
                "z_score": 2.1,
                "direction": "higher",
                "absolute_deviation": 18.5,
                "percentage_deviation": 14.2,
                "unit": "ms",
                "interpretation": "average key hold duration",
            },
            "mean_flight_time": {
                "z_score": 1.7,
                "direction": "higher",
                "absolute_deviation": 25.0,
                "percentage_deviation": 11.5,
                "unit": "ms",
                "interpretation": "inter-key interval",
            },
            "backspace_rate": {
                "z_score": 0.3,
                "direction": "within_baseline",
                "absolute_deviation": 0.01,
                "percentage_deviation": 2.0,
                "unit": "ratio",
                "interpretation": "revision frequency",
            },
        },
    }

    mock_probs = {
        "Calm": 0.12,
        "Fatigued": 0.18,
        "High_Workload": 0.70,
    }

    assessment_input = AssessmentInput(
        user_id="usr_9f83ac01",
        session_id="session_wk_014",
        keystroke_count=75,
        predicted_class="High_Workload",
        class_probabilities=mock_probs,
        baseline_deviation_result=mock_baseline_result,
        model_available=True,
        model_version="lstm-v0.4.0",
        is_mock=True,
    )

    # 2. Generate Assessment
    assessment = generate_behavioral_assessment(assessment_input)

    # Verify structured output
    assert assessment["assessment_state"] == STATE_READY
    assert assessment["assessment_ready"] is True
    assert assessment["user_id"] == "usr_9f83ac01"
    assert assessment["is_mock"] is True

    # Verify Model Prediction Block
    m_pred = assessment["model_prediction"]
    assert m_pred["status"] == "interpreted"
    assert m_pred["predicted_class"] == "High_Workload"
    assert m_pred["top_probability"] == 0.70
    assert m_pred["reliability"] == "HIGH_SEPARATION"

    # Verify Personal Baseline Block (SEPARATE from model prediction)
    b_stat = assessment["personal_baseline"]
    assert b_stat["baseline_status"] == "available"
    assert b_stat["typing_deviation_index"] == 64.5
    assert b_stat["deviation_level"] == "higher_deviation"
    assert len(b_stat["largest_deviations"]) >= 2
    assert b_stat["largest_deviations"][0]["feature"] == "mean_dwell_time"

    # Verify Narrative and Limitations
    assert "High_Workload" in assessment["overall_interpretation"]
    assert "TDI: 64.5/100" in assessment["overall_interpretation"]
    assert len(assessment["limitations"]) >= 4

    # 3. Generate Human-Readable Summary
    summary_text = generate_assessment_summary(assessment)
    assert "MODEL OUTPUT" in summary_text
    assert "PERSONAL TYPING BASELINE" in summary_text
    assert "LARGEST OBSERVED DEVIATIONS" in summary_text
    assert "IMPORTANT LIMITATIONS" in summary_text
    assert "High_Workload" in summary_text

    # 4. Save Machine-Readable JSON Report
    saved_path = save_assessment_report(assessment, output_dir=tmp_path)
    assert Path(saved_path).exists()

    with open(saved_path, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)

    assert loaded_json["user_id"] == "usr_9f83ac01"
    assert loaded_json["privacy"]["zero_text_guaranteed"] is True
    assert loaded_json["privacy"]["raw_text_stored"] is False

    # Audit for zero-text policy in serialized artifact
    json_str = json.dumps(loaded_json).lower()
    for sensitive_keyword in ["password", "typed_text", "raw_keystrokes", "chat_message"]:
        assert sensitive_keyword not in json_str


def test_assessment_when_model_is_unavailable():
    """Verify assessment generation when model is not trained (baseline only)."""
    mock_baseline = {
        "baseline_status": "available",
        "typing_deviation_index": 22.0,
        "features": {},
    }

    input_obj = AssessmentInput(
        user_id="usr_002",
        session_id="sess_002",
        keystroke_count=40,
        baseline_deviation_result=mock_baseline,
        model_available=False,
    )

    assessment = generate_behavioral_assessment(input_obj)

    assert assessment["assessment_state"] == STATE_BASELINE_AVAILABLE
    assert assessment["model_prediction"]["status"] == "model_unavailable"
    assert "trained model has not been loaded" in assessment["overall_interpretation"]


def test_assessment_when_baseline_is_unavailable():
    """Verify assessment generation when personal baseline is cold-start (model prediction only)."""
    input_obj = AssessmentInput(
        user_id="usr_new",
        session_id="sess_new",
        keystroke_count=35,
        predicted_class="Calm",
        class_probabilities={"Calm": 0.85, "Fatigued": 0.15},
        baseline_deviation_result={"baseline_status": "insufficient_history", "message": "Only 1 session."},
        model_available=True,
    )

    assessment = generate_behavioral_assessment(input_obj)

    assert assessment["assessment_state"] == STATE_PREDICTION_AVAILABLE
    assert assessment["personal_baseline"]["baseline_status"] == "insufficient_history"
    assert "Personal baseline is not yet established" in assessment["overall_interpretation"]
    assert assessment["model_prediction"]["predicted_class"] == "Calm"


def test_assessment_when_insufficient_keystrokes():
    """Verify assessment blocking when keystrokes are below minimum threshold."""
    input_obj = AssessmentInput(
        user_id="usr_003",
        session_id="sess_short",
        keystroke_count=2,  # < 5 keystrokes
        model_available=True,
    )

    assessment = generate_behavioral_assessment(input_obj)

    assert assessment["assessment_state"] == STATE_INSUFFICIENT_DATA
    assert assessment["assessment_ready"] is False
    assert "insufficient keystroke observations" in assessment["overall_interpretation"].lower()
