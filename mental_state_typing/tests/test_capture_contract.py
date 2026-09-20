"""Capture Component Contract Tests.

Tests the component/Python boundary contract:
- Valid payloads with event_id, event_type ("keydown", "keyup", "down", "up"), timestamp
- Malformed payloads rejected safely
- Duplicate payloads ignored
- Raw text is strictly prohibited and never persisted
- Event timestamps are processed accurately
- Chronological ordering is preserved
"""

import pytest
from src.live_typing.event_normalizer import LiveEventNormalizer
from src.live_typing.event_types import KeyEventType, RawBrowserEvent, TypingEvent
from src.live_typing.privacy_filter import (
    FORBIDDEN_EVENT_KEYS,
    PrivacyViolationError,
    validate_browser_event,
)
from src.live_typing.session import LiveTypingSession, SessionState


def test_contract_valid_payload_acceptance():
    """Verify that both 'keydown'/'keyup' and 'down'/'up' payloads are accepted with event_id."""
    down_payload = {
        "event_id": "ev_contract_down_01",
        "event_type": "keydown",
        "timestamp": 1000.0,
        "key_token": "k_alpha",
    }
    up_payload = {
        "event_id": "ev_contract_up_01",
        "event_type": "keyup",
        "timestamp": 1090.0,
        "key_token": "k_alpha",
    }

    down_ev = validate_browser_event(down_payload)
    up_ev = validate_browser_event(up_payload)

    assert down_ev is not None
    assert down_ev.event_id == "ev_contract_down_01"
    assert down_ev.event_type == KeyEventType.KEY_DOWN.value
    assert down_ev.timestamp_ms == 1000.0
    assert down_ev.key_token == "k_alpha"

    assert up_ev is not None
    assert up_ev.event_id == "ev_contract_up_01"
    assert up_ev.event_type == KeyEventType.KEY_UP.value
    assert up_ev.timestamp_ms == 1090.0

    # Also test 'down' and 'up' variants with 'timestamp_ms'
    alt_payload = {
        "event_id": "ev_contract_down_02",
        "type": "down",
        "timestamp_ms": 1200.0,
        "key_token": "k_space",
        "is_space": True,
    }
    alt_ev = validate_browser_event(alt_payload)
    assert alt_ev is not None
    assert alt_ev.event_type == "down"
    assert alt_ev.is_space is True


def test_contract_malformed_payload_rejected_safely():
    """Verify that malformed payloads do not raise unhandled exceptions and return None."""
    malformed_cases = [
        {},  # empty
        {"event_type": "click"},  # invalid type
        {"event_type": "keydown"},  # missing timestamp
        {"timestamp": 1000.0},  # missing event_type
        {"event_type": "keydown", "timestamp": "invalid_number"},
        {"event_type": "keydown", "timestamp": -50.0},  # negative timestamp
        {"event_type": "keydown", "timestamp": float("nan")},  # NaN
        {"event_type": "keydown", "timestamp": float("inf")},  # inf
        None,  # non-dict
        "string payload",  # non-dict
        [1, 2, 3],  # non-dict
    ]

    for item in malformed_cases:
        res = validate_browser_event(item, strict_raise=False)
        assert res is None, f"Expected None for malformed case: {item}"


def test_contract_duplicate_payload_ignored():
    """Verify that duplicate payloads are ignored by the session ingest engine."""
    session = LiveTypingSession()
    session.start()

    payload_down = {
        "event_id": "ev_dup_1",
        "event_type": "keydown",
        "timestamp": 2000.0,
        "key_token": "k_alpha",
    }
    payload_up = {
        "event_id": "ev_dup_2",
        "event_type": "keyup",
        "timestamp": 2080.0,
        "key_token": "k_alpha",
    }

    # First ingestion
    added1 = session.ingest_browser_batch([payload_down, payload_up])
    assert added1 == 1
    assert session.event_count == 2
    assert len(session.get_paired_events()) == 1

    # Exact duplicate delivery (e.g. network retry or Streamlit rerun)
    added2 = session.ingest_browser_batch([payload_down, payload_up])
    assert added2 == 0
    # Counts remain unchanged
    assert session.event_count == 2
    assert len(session.get_paired_events()) == 1


def test_contract_raw_text_never_persisted_or_accepted():
    """Verify that any payload containing character identities triggers PrivacyViolationError."""
    forbidden_payloads = [
        {"event_id": "e1", "event_type": "keydown", "timestamp": 1000.0, "key": "a"},
        {"event_id": "e2", "event_type": "keydown", "timestamp": 1000.0, "char": "z"},
        {"event_id": "e3", "event_type": "keydown", "timestamp": 1000.0, "character": "x"},
        {"event_id": "e4", "event_type": "keydown", "timestamp": 1000.0, "text": "hello"},
        {"event_id": "e5", "event_type": "keydown", "timestamp": 1000.0, "typed_text": "secret"},
        {"event_id": "e6", "event_type": "keydown", "timestamp": 1000.0, "password": "pass"},
        {"event_id": "e7", "event_type": "keydown", "timestamp": 1000.0, "word": "test"},
        {"event_id": "e8", "event_type": "keydown", "timestamp": 1000.0, "key_token": "a"},  # raw single char
    ]

    for p in forbidden_payloads:
        with pytest.raises(PrivacyViolationError):
            validate_browser_event(p, strict_raise=True)


def test_contract_event_timestamps_and_dwell_calculation():
    """Verify exact calculation of dwell and flight times from component events."""
    normalizer = LiveEventNormalizer()

    ev_down = RawBrowserEvent(
        event_id="e1",
        event_type="down",
        timestamp_ms=5000.0,
        key_token="k_alpha",
    )
    ev_up = RawBrowserEvent(
        event_id="e2",
        event_type="up",
        timestamp_ms=5075.0,
        key_token="k_alpha",
    )

    p1 = normalizer.process_event(ev_down)
    assert p1 is None  # Pending keyup

    p2 = normalizer.process_event(ev_up)
    assert isinstance(p2, TypingEvent)
    assert p2.dwell_time == 75.0
    assert p2.flight_time is None  # Initial event


def test_contract_event_ordering_preserved():
    """Verify chronological ordering is maintained even when events arrive interleaved."""
    normalizer = LiveEventNormalizer()

    events = [
        RawBrowserEvent(event_id="e2", event_type="up", timestamp_ms=6080.0, key_token="k_alpha"),
        RawBrowserEvent(event_id="e1", event_type="down", timestamp_ms=6000.0, key_token="k_alpha"),
    ]

    # Process batch sorts by timestamp
    paired = normalizer.process_batch(events)
    assert len(paired) == 1
    assert paired[0].press_timestamp == 6000.0
    assert paired[0].release_timestamp == 6080.0
    assert paired[0].dwell_time == 80.0
