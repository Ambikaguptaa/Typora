"""Test-only mock model isolation and end-to-end inference verification.

CRITICAL ARCHITECTURAL SAFETY GUARANTEE:
This mock engine is strictly marked `is_mock = True` and exists exclusively for
unit testing the downstream assessment and uncertainty pipeline. It is NEVER
loaded or referenced by the production Streamlit application.
"""

from typing import Any, Dict
import numpy as np
import pytest

from src.integration.behavioral_engine import BehavioralEngine
from src.integration.pipeline_state import SessionType


class MockTestModelEngine:
    """Isolated, test-only mock sequence predictor."""

    is_mock: bool = True

    def __init__(self, target_class: str = "cadence_regular", high_separation: bool = True):
        self.target_class = target_class
        self.high_separation = high_separation

    def predict_sequences(self, X: np.ndarray) -> Dict[str, Any]:
        """Produce deterministic mock probabilities for unit testing pipeline orchestration."""
        if self.high_separation:
            probs = {"cadence_regular": 0.82, "cadence_hesitant": 0.12, "cadence_variable": 0.06}
            reliability = "HIGH_SEPARATION"
            margin = 0.70
            entropy = 0.28
        else:
            probs = {"cadence_regular": 0.38, "cadence_hesitant": 0.35, "cadence_variable": 0.27}
            reliability = "LOW_SEPARATION"
            margin = 0.03
            entropy = 0.95

        return {
            "predicted_class": self.target_class,
            "class_probabilities": probs,
            "top_probability": max(probs.values()),
            "second_probability": sorted(probs.values())[-2],
            "probability_margin": margin,
            "normalized_entropy": entropy,
            "reliability": reliability,
            "is_mock": True,
        }


def _generate_valid_raw_events(count: int = 35) -> list[dict]:
    events = []
    t = 1000.0
    for i in range(count):
        dwell = 85.0
        events.append({"event_type": "down", "timestamp_ms": t, "key_token": "k_alpha"})
        events.append({"event_type": "up", "timestamp_ms": t + dwell, "key_token": "k_alpha"})
        t += 180.0
    return events


def test_mock_model_produces_assessment_when_active(tmp_path):
    """Verify that when a mock engine is explicitly injected, full assessment is produced."""
    mock_engine = MockTestModelEngine(target_class="cadence_regular", high_separation=True)
    assert mock_engine.is_mock is True

    engine = BehavioralEngine(
        user_id="mock_test_user",
        session_type=SessionType.ANALYSIS,
        models_dir=tmp_path / "models",
        baselines_dir=tmp_path / "baselines",
        assessments_dir=tmp_path / "assessments",
        mock_model_engine=mock_engine,
    )

    engine.start_session()
    engine.session.start_time -= 5.0
    engine.ingest_raw_events(_generate_valid_raw_events(35))
    res = engine.stop_session()

    assert res.data_quality.is_valid
    assert res.model_result.status == "MODEL_READY"
    assert res.model_result.is_mock is True
    assert res.model_result.predicted_class == "cadence_regular"
    assert res.model_result.reliability == "HIGH_SEPARATION"
    assert res.model_result.probability_margin == 0.70

    # Verify behavioral assessment received and interpreted model signal
    assert res.assessment_result is not None
    assert "cadence_regular" in res.assessment_result["overall_interpretation"]


def test_production_engine_defaults_to_model_not_ready(tmp_path):
    """Verify default engine without mock NEVER activates mock predictions and reports MODEL_NOT_READY."""
    default_engine = BehavioralEngine(
        user_id="prod_user",
        models_dir=tmp_path / "models",
        baselines_dir=tmp_path / "baselines",
        assessments_dir=tmp_path / "assessments",
    )
    assert default_engine.mock_model_engine is None

    default_engine.start_session()
    default_engine.session.start_time -= 5.0
    default_engine.ingest_raw_events(_generate_valid_raw_events(35))
    res = default_engine.stop_session()

    assert res.model_result.status == "MODEL_NOT_READY"
    assert res.model_result.is_mock is False
    assert res.model_result.predicted_class is None
    assert res.model_result.class_probabilities is None
