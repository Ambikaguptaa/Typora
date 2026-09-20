"""Deterministic headless unit and integration tests for the Live Behavioral Session.

Tests all 30 required conditions:
1. READY initial state
2. START transition (READY -> CAPTURING)
3. Actual event ingestion
4. Event ordering preserved
5. Dwell-time calculation
6. Flight-time calculation
7. Pause detection
8. Backspace classification
9. Feature extraction
10. Feature shape
11. 30-row model window creation
12. Insufficient-event handling
13. PAUSE transition
14. Paused events ignored
15. RESUME transition
16. STOP transition
17. Quality validation
18. Calibration persistence
19. Analysis with baseline
20. Analysis without baseline
21. LSTM unavailable
22. LSTM available (mock)
23. Malformed event payload
24. Duplicate events
25. Empty event stream
26. NaN/inf protection
27. RESET
28. Session persistence
29. Privacy / zero-text enforcement
30. No raw text persistence
"""

import time
from pathlib import Path
import numpy as np
import pytest

from database.database import get_assessment_by_id, init_db
from src.integration.behavioral_engine import BehavioralEngine
from src.integration.pipeline_state import PipelineState, SessionType
from src.integration.result_schema import BehavioralAssessmentResult
from src.live_typing.event_normalizer import LiveEventNormalizer
from src.live_typing.event_types import RawBrowserEvent, TypingEvent
from src.live_typing.feature_buffer import CANONICAL_SEQUENCE_FEATURES, LiveFeatureBuffer
from src.live_typing.privacy_filter import PrivacyViolationError, validate_browser_event
from src.live_typing.session import LiveTypingSession, SessionState, SessionStatus
from src.live_typing.validation import validate_session_quality


@pytest.fixture
def clean_engine(tmp_path):
    """Create clean BehavioralEngine with temporary directories."""
    db_file = tmp_path / "test_live_session.db"
    init_db(db_file)
    engine = BehavioralEngine(
        user_id="test_user_live",
        session_type=SessionType.ANALYSIS,
        models_dir=tmp_path / "models",
        baselines_dir=tmp_path / "baselines",
        assessments_dir=tmp_path / "assessments",
    )
    return engine


def generate_synthetic_keystroke_batch(num_keystrokes: int = 20, start_time_ms: float = 1000.0, dwell_ms: float = 80.0, flight_ms: float = 120.0):
    """Generate a realistic synthetic event sequence (keydown, keyup pairs)."""
    events = []
    current_time = start_time_ms
    for i in range(num_keystrokes):
        # Key down
        events.append({
            "event_id": f"test_down_{i}_{current_time}",
            "event_type": "keydown",
            "timestamp": current_time,
            "key_token": "k_alpha" if i % 5 != 0 else "k_backspace",
            "is_backspace": (i % 5 == 0),
        })
        # Key up
        current_time += dwell_ms
        events.append({
            "event_id": f"test_up_{i}_{current_time}",
            "event_type": "keyup",
            "timestamp": current_time,
            "key_token": "k_alpha" if i % 5 != 0 else "k_backspace",
            "is_backspace": (i % 5 == 0),
        })
        current_time += flight_ms
    return events


# ==============================================================================
# 1-4: STATE & INGESTION
# ==============================================================================

def test_1_ready_initial_state(clean_engine):
    """Verify initial state is strictly READY."""
    assert clean_engine.session.state == SessionState.READY
    assert clean_engine.session.status == SessionState.READY
    assert clean_engine.state == PipelineState.READY
    assert clean_engine.session.event_count == 0
    assert len(clean_engine.session.get_paired_events()) == 0


def test_2_start_transition(clean_engine):
    """Verify READY -> START -> CAPTURING transition."""
    sid = clean_engine.start_session()
    assert clean_engine.session.state == SessionState.CAPTURING
    assert clean_engine.session.status == SessionState.CAPTURING
    assert clean_engine.state == PipelineState.CAPTURING
    assert clean_engine.session.start_time is not None
    assert sid.startswith("sess_")

    # Disallow starting while already CAPTURING
    with pytest.raises(ValueError, match="Cannot START"):
        clean_engine.session.start()


def test_3_actual_event_ingestion(clean_engine):
    """Verify raw browser events flow into actual capture pipeline."""
    clean_engine.start_session()
    events = generate_synthetic_keystroke_batch(num_keystrokes=5)
    added = clean_engine.ingest_raw_events(events)

    assert added == 5
    assert clean_engine.session.event_count == 10
    assert len(clean_engine.session.get_paired_events()) == 5


def test_4_event_ordering(clean_engine):
    """Verify event ordering is preserved even if emitted slightly out-of-order."""
    clean_engine.start_session()
    # Scrambled order
    batch = [
        {"event_id": "e2", "event_type": "keyup", "timestamp": 1100.0, "key_token": "k_alpha"},
        {"event_id": "e1", "event_type": "keydown", "timestamp": 1000.0, "key_token": "k_alpha"},
        {"event_id": "e4", "event_type": "keyup", "timestamp": 1300.0, "key_token": "k_alpha"},
        {"event_id": "e3", "event_type": "keydown", "timestamp": 1200.0, "key_token": "k_alpha"},
    ]
    added = clean_engine.ingest_raw_events(batch)
    assert added == 2
    paired = clean_engine.session.get_paired_events()
    assert paired[0].press_timestamp == 1000.0
    assert paired[1].press_timestamp == 1200.0


# ==============================================================================
# 5-11: FEATURES & 30-EVENT WINDOW
# ==============================================================================

def test_5_dwell_time_calculation(clean_engine):
    """Verify dwell time is calculated accurately from keydown and keyup timestamps."""
    clean_engine.start_session()
    batch = [
        {"event_id": "d1", "event_type": "keydown", "timestamp": 1000.0, "key_token": "k_alpha"},
        {"event_id": "u1", "event_type": "keyup", "timestamp": 1085.0, "key_token": "k_alpha"},
    ]
    clean_engine.ingest_raw_events(batch)
    paired = clean_engine.session.get_paired_events()
    assert len(paired) == 1
    assert paired[0].dwell_time == 85.0


def test_6_flight_time_calculation(clean_engine):
    """Verify flight time is calculated from previous release to current press."""
    clean_engine.start_session()
    batch = [
        {"event_id": "d1", "event_type": "keydown", "timestamp": 1000.0, "key_token": "k_alpha"},
        {"event_id": "u1", "event_type": "keyup", "timestamp": 1080.0, "key_token": "k_alpha"},
        {"event_id": "d2", "event_type": "keydown", "timestamp": 1200.0, "key_token": "k_alpha"},
        {"event_id": "u2", "event_type": "keyup", "timestamp": 1270.0, "key_token": "k_alpha"},
    ]
    clean_engine.ingest_raw_events(batch)
    paired = clean_engine.session.get_paired_events()
    assert len(paired) == 2
    assert paired[0].flight_time is None  # First key has no preceding release
    assert paired[1].flight_time == 120.0  # 1200 - 1080


def test_7_pause_detection(clean_engine):
    """Verify pause durations > 500ms are identified in feature dataframe."""
    clean_engine.start_session()
    batch = [
        {"event_id": "d1", "event_type": "keydown", "timestamp": 1000.0, "key_token": "k_alpha"},
        {"event_id": "u1", "event_type": "keyup", "timestamp": 1050.0, "key_token": "k_alpha"},
        # 600ms flight = pause
        {"event_id": "d2", "event_type": "keydown", "timestamp": 1650.0, "key_token": "k_alpha"},
        {"event_id": "u2", "event_type": "keyup", "timestamp": 1700.0, "key_token": "k_alpha"},
    ]
    clean_engine.ingest_raw_events(batch)
    df = clean_engine.feature_buffer.to_dataframe()
    assert len(df) == 2
    assert df.iloc[1]["pause_duration"] == 600.0


def test_8_backspace_classification(clean_engine):
    """Verify backspaces are correctly classified and counted."""
    clean_engine.start_session()
    batch = [
        {"event_id": "d1", "event_type": "keydown", "timestamp": 1000.0, "key_token": "k_alpha"},
        {"event_id": "u1", "event_type": "keyup", "timestamp": 1080.0, "key_token": "k_alpha"},
        {"event_id": "d2", "event_type": "keydown", "timestamp": 1200.0, "key_token": "k_backspace", "is_backspace": True},
        {"event_id": "u2", "event_type": "keyup", "timestamp": 1280.0, "key_token": "k_backspace", "is_backspace": True},
    ]
    clean_engine.ingest_raw_events(batch)
    paired = clean_engine.session.get_paired_events()
    assert paired[1].is_backspace is True
    telemetry = clean_engine.feature_buffer.extract_telemetry()
    assert telemetry["backspace_count"] == 1


def test_9_feature_extraction(clean_engine):
    """Verify canonical feature DataFrame extraction matches expected columns."""
    clean_engine.start_session()
    events = generate_synthetic_keystroke_batch(num_keystrokes=10)
    clean_engine.ingest_raw_events(events)

    df = clean_engine.feature_buffer.to_dataframe()
    assert not df.empty
    assert len(df) == 10
    for col in CANONICAL_SEQUENCE_FEATURES:
        assert col in df.columns


def test_10_feature_shape(clean_engine):
    """Verify 3D temporal sequence arrays match shape (num_windows, 30, 6)."""
    clean_engine.start_session()
    # Ingest 35 keystrokes -> enough for 1 window of 30
    events = generate_synthetic_keystroke_batch(num_keystrokes=35)
    clean_engine.ingest_raw_events(events)

    X, meta = clean_engine.feature_buffer.generate_sequence_windows(sequence_length=30, sequence_stride=10)
    assert X.shape == (1, 30, 6)
    assert len(meta) == 1
    assert meta[0]["start_event_idx"] == 0
    assert meta[0]["end_event_idx"] == 30


def test_11_30_row_model_window_creation(clean_engine):
    """Verify exact 30 valid feature rows produce one model window."""
    clean_engine.start_session()
    events = generate_synthetic_keystroke_batch(num_keystrokes=30)
    clean_engine.ingest_raw_events(events)

    X, _ = clean_engine.feature_buffer.generate_sequence_windows(sequence_length=30)
    assert len(X) == 1
    assert X.shape[1] == 30
    assert X.shape[2] == 6


def test_12_insufficient_event_handling(clean_engine):
    """Verify fewer than 30 events produces 0 model sequence windows."""
    clean_engine.start_session()
    events = generate_synthetic_keystroke_batch(num_keystrokes=25)
    clean_engine.ingest_raw_events(events)

    X, meta = clean_engine.feature_buffer.generate_sequence_windows(sequence_length=30)
    assert len(X) == 0
    assert len(meta) == 0


# ==============================================================================
# 13-16: PAUSE, RESUME, STOP TRANSITIONS
# ==============================================================================

def test_13_pause_transition(clean_engine):
    """Verify CAPTURING -> PAUSE -> PAUSED."""
    clean_engine.start_session()
    clean_engine.pause_session()
    assert clean_engine.session.state == SessionState.PAUSED
    assert clean_engine.state == PipelineState.PAUSED

    # Invalid transition: cannot pause while already paused
    with pytest.raises(ValueError, match="Cannot PAUSE"):
        clean_engine.session.pause()


def test_14_paused_events_ignored(clean_engine):
    """Verify keystrokes typed while paused are strictly ignored."""
    clean_engine.start_session()
    events1 = generate_synthetic_keystroke_batch(num_keystrokes=5)
    clean_engine.ingest_raw_events(events1)
    assert len(clean_engine.session.get_paired_events()) == 5

    clean_engine.pause_session()
    # Attempt to ingest while paused
    events_while_paused = generate_synthetic_keystroke_batch(num_keystrokes=5, start_time_ms=5000.0)
    added = clean_engine.ingest_raw_events(events_while_paused)
    assert added == 0
    # Buffer unchanged
    assert len(clean_engine.session.get_paired_events()) == 5


def test_15_resume_transition_and_flight_boundary(clean_engine):
    """Verify PAUSED -> RESUME -> CAPTURING and no artificial flight time spike is injected."""
    clean_engine.start_session()
    batch1 = [
        {"event_id": "d1", "event_type": "keydown", "timestamp": 1000.0, "key_token": "k_alpha"},
        {"event_id": "u1", "event_type": "keyup", "timestamp": 1080.0, "key_token": "k_alpha"},
    ]
    clean_engine.ingest_raw_events(batch1)

    clean_engine.pause_session()
    time.sleep(0.05)  # Simulate pause delay
    clean_engine.resume_session()
    assert clean_engine.session.state == SessionState.CAPTURING

    # First event after resume occurs 2000ms later
    batch2 = [
        {"event_id": "d2", "event_type": "keydown", "timestamp": 3080.0, "key_token": "k_alpha"},
        {"event_id": "u2", "event_type": "keyup", "timestamp": 3160.0, "key_token": "k_alpha"},
    ]
    clean_engine.ingest_raw_events(batch2)
    paired = clean_engine.session.get_paired_events()
    assert len(paired) == 2
    # Flight time for first key after resume must NOT be 2000ms
    assert paired[1].flight_time is None or paired[1].flight_time == 0.0


def test_16_stop_transition(clean_engine):
    """Verify CAPTURING -> STOP -> COMPLETED."""
    clean_engine.start_session()
    clean_engine.stop_session()
    assert clean_engine.session.state == SessionState.COMPLETED
    assert clean_engine.state == PipelineState.COMPLETED
    assert clean_engine.session.end_time is not None

    # Disallow stopping when already completed
    with pytest.raises(ValueError, match="Cannot STOP"):
        clean_engine.session.stop()


# ==============================================================================
# 17-22: VALIDATION, CALIBRATION, ANALYSIS, LSTM
# ==============================================================================

def test_17_validation_with_insufficient_data(clean_engine):
    """Verify quality validation rejects sessions below 15 events."""
    clean_engine.start_session()
    events = generate_synthetic_keystroke_batch(num_keystrokes=4)
    clean_engine.ingest_raw_events(events)

    res = clean_engine.stop_session()
    assert isinstance(res, BehavioralAssessmentResult)
    assert not res.data_quality.is_valid
    assert res.data_quality.verdict == "FAIL"
    assert len(res.data_quality.reasons) > 0


def test_18_calibration_session_persistence(clean_engine):
    """Verify calibration session updates personal baseline history without model prediction."""
    clean_engine.set_session_type(SessionType.CALIBRATION)
    clean_engine.start_session()

    events = generate_synthetic_keystroke_batch(num_keystrokes=20, flight_ms=100.0)
    clean_engine.ingest_raw_events(events)

    res = clean_engine.stop_session()
    assert res.session_type == "CALIBRATION"
    assert res.data_quality.is_valid
    assert len(clean_engine.user_calibration_history) == 1
    # Baseline not ready yet (needs 5)
    assert res.baseline_result.status == "NOT_READY"
    assert res.baseline_result.session_count == 1
    # Model should not have run
    assert res.model_result.predicted_class is None


def test_19_analysis_with_established_baseline(clean_engine):
    """Verify analysis session calculates TDI when 5 calibration sessions exist."""
    # Seed 5 calibration sessions
    for i in range(5):
        clean_engine.user_calibration_history.append({
            "session_id": f"calib_{i}",
            "mean_hold_time_ms": 80.0 + i * 2,
            "mean_flight_time_ms": 120.0,
            "pause_rate": 0.05,
            "typing_speed": 45.0,
            "backspace_rate": 0.02,
        })
    clean_engine._save_user_history()

    clean_engine.set_session_type(SessionType.ANALYSIS)
    clean_engine.start_session()
    events = generate_synthetic_keystroke_batch(num_keystrokes=20)
    clean_engine.ingest_raw_events(events)

    res = clean_engine.stop_session()
    assert res.baseline_result.status == "READY"
    assert res.baseline_result.typing_deviation_index is not None
    assert 0.0 <= res.baseline_result.typing_deviation_index <= 100.0


def test_20_analysis_without_baseline(clean_engine):
    """Verify analysis session reports BASELINE NOT READY when < 5 calibration sessions exist."""
    clean_engine.set_session_type(SessionType.ANALYSIS)
    clean_engine.start_session()
    events = generate_synthetic_keystroke_batch(num_keystrokes=20)
    clean_engine.ingest_raw_events(events)

    res = clean_engine.stop_session()
    assert res.baseline_result.status == "NOT_READY"
    assert "Minimum 5 calibration sessions required" in res.baseline_result.message


def test_21_lstm_unavailable_does_not_fabricate(clean_engine):
    """Verify absence of model files returns MODEL_NOT_READY without fake predictions."""
    clean_engine.start_session()
    events = generate_synthetic_keystroke_batch(num_keystrokes=35)
    clean_engine.ingest_raw_events(events)

    res = clean_engine.stop_session()
    assert res.model_result.status == "MODEL_NOT_READY"
    assert res.model_result.predicted_class is None
    assert "No verified production LSTM model file exists" in res.model_result.reason


def test_22_lstm_available_with_mock_engine(clean_engine):
    """Verify model inference path when approved engine is supplied."""
    class MockModel:
        is_mock = True
        def predict_sequences(self, X):
            return {
                "predicted_class": "Class_Normal",
                "class_probabilities": {"Class_Normal": 0.85, "Class_Strain": 0.15},
                "top_probability": 0.85,
                "second_probability": 0.15,
                "probability_margin": 0.70,
                "normalized_entropy": 0.35,
                "reliability": "HIGH",
            }

    clean_engine.mock_model_engine = MockModel()
    clean_engine.start_session()
    events = generate_synthetic_keystroke_batch(num_keystrokes=35)
    clean_engine.ingest_raw_events(events)

    res = clean_engine.stop_session()
    assert res.model_result.status == "MODEL_READY"
    assert res.model_result.predicted_class == "Class_Normal"
    assert res.model_result.reliability == "HIGH"


# ==============================================================================
# 23-30: CONTRACT, SECURITY, DEDUP, RESET, PERSISTENCE
# ==============================================================================

def test_23_malformed_event_payload_rejected(clean_engine):
    """Verify malformed payloads (non-dict, invalid types) are rejected safely."""
    clean_engine.start_session()
    malformed = [
        "not a dict",
        {"event_type": "invalid_action", "timestamp": 1000.0},
        {"event_type": "keydown", "timestamp": "not-a-number"},
        None,
    ]
    added = clean_engine.ingest_raw_events(malformed, strict_privacy=False)
    assert added == 0


def test_24_duplicate_events_ignored(clean_engine):
    """Verify duplicate event IDs or duplicate signatures are dropped."""
    clean_engine.start_session()
    batch1 = [
        {"event_id": "ev_unique_1", "event_type": "keydown", "timestamp": 1000.0, "key_token": "k_alpha"},
        {"event_id": "ev_unique_2", "event_type": "keyup", "timestamp": 1080.0, "key_token": "k_alpha"},
    ]
    added1 = clean_engine.ingest_raw_events(batch1)
    assert added1 == 1

    # Re-ingest the exact same batch
    added2 = clean_engine.ingest_raw_events(batch1)
    assert added2 == 0
    assert len(clean_engine.session.get_paired_events()) == 1


def test_25_empty_event_stream_handled(clean_engine):
    """Verify empty event stream is handled gracefully without crashing."""
    clean_engine.start_session()
    added = clean_engine.ingest_raw_events([])
    assert added == 0
    res = clean_engine.stop_session()
    assert res.data_quality.verdict == "FAIL"


def test_26_nan_and_inf_protection():
    """Verify NaN and inf timestamps are strictly rejected by privacy filter and validation."""
    ev_nan = {"event_type": "keydown", "timestamp": float("nan"), "key_token": "k_alpha"}
    assert validate_browser_event(ev_nan) is None

    ev_inf = {"event_type": "keydown", "timestamp": float("inf"), "key_token": "k_alpha"}
    assert validate_browser_event(ev_inf) is None


def test_27_reset_restores_ready_state(clean_engine):
    """Verify RESET clears in-memory buffers and restores READY state."""
    clean_engine.start_session()
    events = generate_synthetic_keystroke_batch(num_keystrokes=5)
    clean_engine.ingest_raw_events(events)
    assert clean_engine.session.event_count > 0

    old_sid = clean_engine.session.session_id
    clean_engine.reset_session()

    assert clean_engine.session.state == SessionState.READY
    assert clean_engine.state == PipelineState.READY
    assert clean_engine.session.session_id != old_sid
    assert clean_engine.session.event_count == 0
    assert len(clean_engine.session.get_paired_events()) == 0
    assert clean_engine.feature_buffer.event_count == 0


def test_28_session_persistence(clean_engine):
    """Verify completed session assessment is persisted and readable from database."""
    clean_engine.start_session()
    events = generate_synthetic_keystroke_batch(num_keystrokes=20)
    clean_engine.ingest_raw_events(events)

    res = clean_engine.stop_session()
    sid = res.session_id

    # Verify retrieval from persistent storage
    loaded = get_assessment_by_id(sid)
    assert loaded is not None
    assert loaded["session_id"] == sid


def test_29_privacy_zero_text_enforcement(clean_engine):
    """Verify forbidden text keys cause immediate PrivacyViolationError."""
    clean_engine.start_session()
    dirty_batch = [
        {"event_type": "keydown", "timestamp": 1000.0, "key": "a"},
    ]
    with pytest.raises(PrivacyViolationError, match="Forbidden keys detected"):
        clean_engine.ingest_raw_events(dirty_batch, strict_privacy=True)


def test_30_no_raw_text_persisted(clean_engine):
    """Verify saved assessment payload contains zero forbidden text fields."""
    clean_engine.start_session()
    events = generate_synthetic_keystroke_batch(num_keystrokes=20)
    clean_engine.ingest_raw_events(events)
    res = clean_engine.stop_session()

    d = res.to_dict()
    forbidden = ["key", "text", "character", "password", "word", "sentence", "typed_text"]
    for k in forbidden:
        assert k not in d
