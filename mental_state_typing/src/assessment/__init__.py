"""Behavioral assessment and model interpretation package."""

from src.assessment.behavioral_assessment import (
    AssessmentInput,
    generate_assessment_summary,
    generate_behavioral_assessment,
    save_assessment_report,
)
from src.assessment.report_generator import (
    generate_markdown_report,
    generate_text_report,
)
from src.assessment.interpretation import (
    ASSESSMENT_DISCLAIMER,
    DEVIATION_LEVEL_EXPECTED,
    DEVIATION_LEVEL_HIGHER,
    DEVIATION_LEVEL_MODERATE,
    RELIABILITY_HIGH_SEPARATION,
    RELIABILITY_LOW_SEPARATION,
    RELIABILITY_MODERATE_SEPARATION,
    interpret_baseline_deviation,
    interpret_model_prediction,
    rank_feature_deviations,
)
from src.assessment.quality_checks import (
    STATE_BASELINE_AVAILABLE,
    STATE_BASELINE_UNAVAILABLE,
    STATE_INSUFFICIENT_DATA,
    STATE_MODEL_UNAVAILABLE,
    STATE_NOT_READY,
    STATE_PREDICTION_AVAILABLE,
    STATE_READY,
    VALID_ASSESSMENT_STATES,
    validate_assessment_data,
)

__all__ = [
    "AssessmentInput",
    "generate_behavioral_assessment",
    "generate_assessment_summary",
    "generate_markdown_report",
    "generate_text_report",
    "save_assessment_report",
    "interpret_model_prediction",
    "interpret_baseline_deviation",
    "rank_feature_deviations",
    "validate_assessment_data",
    "RELIABILITY_HIGH_SEPARATION",
    "RELIABILITY_MODERATE_SEPARATION",
    "RELIABILITY_LOW_SEPARATION",
    "DEVIATION_LEVEL_EXPECTED",
    "DEVIATION_LEVEL_MODERATE",
    "DEVIATION_LEVEL_HIGHER",
    "STATE_READY",
    "STATE_PREDICTION_AVAILABLE",
    "STATE_BASELINE_AVAILABLE",
    "STATE_MODEL_UNAVAILABLE",
    "STATE_BASELINE_UNAVAILABLE",
    "STATE_INSUFFICIENT_DATA",
    "STATE_NOT_READY",
    "VALID_ASSESSMENT_STATES",
    "ASSESSMENT_DISCLAIMER",
]
