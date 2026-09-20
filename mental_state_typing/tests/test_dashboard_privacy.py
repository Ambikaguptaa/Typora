"""Tests for Dashboard Privacy enforcement, zero raw text auditing, and chart data safety."""

import pandas as pd
import pytest

from src.integration.behavioral_engine import BehavioralEngine
from src.integration.result_schema import FORBIDDEN_PAYLOAD_FIELDS
from src.live_typing.privacy_filter import PrivacyViolationError, audit_payload_for_sensitive_keys
from src.visualization.charts import validate_chart_privacy


def test_chart_data_privacy_validation_passes_on_numerical_features():
    """Verify that validate_chart_privacy cleanly passes for valid numerical telemetry."""
    valid_payload = {
        "mean_dwell_ms": 95.4,
        "mean_flight_ms": 112.8,
        "pause_rate": 0.05,
        "estimated_wpm": 46.2,
        "backspace_count": 2,
    }
    # Should not raise any exception
    validate_chart_privacy(valid_payload)

    # List of numbers
    validate_chart_privacy([10.5, 20.2, 30.1])

    # DataFrame with valid columns
    df = pd.DataFrame({"dwell_time": [100.0, 110.0], "flight_time": [120.0, 130.0]})
    validate_chart_privacy(df)


@pytest.mark.parametrize("forbidden_key", list(FORBIDDEN_PAYLOAD_FIELDS))
def test_chart_data_privacy_validation_rejects_forbidden_keys(forbidden_key: str):
    """Verify that validate_chart_privacy raises PrivacyViolationError for every forbidden key."""
    tainted_payload = {
        forbidden_key: "Sensitive content",
        "dwell_time": 100.0,
    }
    with pytest.raises(PrivacyViolationError):
        validate_chart_privacy(tainted_payload)


def test_engine_telemetry_contains_no_forbidden_keys():
    """Verify that BehavioralEngine telemetry extraction has zero forbidden keys."""
    engine = BehavioralEngine()
    engine.start_session()

    # Ingest synthetic events
    sample_events = [
        {"timestamp": 0.0, "type": "down", "is_backspace": False, "is_enter": False, "is_space": False, "key_token": "k_alpha"},
        {"timestamp": 85.0, "type": "up", "is_backspace": False, "is_enter": False, "is_space": False, "key_token": "k_alpha"},
        {"timestamp": 160.0, "type": "down", "is_backspace": False, "is_enter": False, "is_space": False, "key_token": "k_alpha"},
        {"timestamp": 240.0, "type": "up", "is_backspace": False, "is_enter": False, "is_space": False, "key_token": "k_alpha"},
    ]
    engine.ingest_raw_events(sample_events)

    telemetry = engine.feature_buffer.extract_telemetry()
    audit_payload_for_sensitive_keys(telemetry)

    # Check keys against forbidden set
    for k in telemetry.keys():
        assert k.lower() not in FORBIDDEN_PAYLOAD_FIELDS


def test_dataframe_with_forbidden_columns_rejected():
    """Verify that a DataFrame containing raw text columns is rejected."""
    tainted_df = pd.DataFrame(
        {
            "dwell_time": [100.0, 110.0],
            "typed_text": ["a", "b"],
        }
    )
    with pytest.raises(PrivacyViolationError):
        validate_chart_privacy(tainted_df)
