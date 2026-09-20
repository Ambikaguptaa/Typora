"""Unit tests for live typing event capture interface and bridge safety."""

import pytest
from src.live_typing.event_types import KeyEventType, RawBrowserEvent
from src.live_typing.privacy_filter import (
    FORBIDDEN_PAYLOAD_FIELDS,
    audit_payload_for_sensitive_keys,
    validate_browser_event,
)


def test_raw_browser_event_structure():
    """Verify RawBrowserEvent creates sanitized, strictly temporal event records."""
    event = RawBrowserEvent(
        event_type="down",
        timestamp_ms=1234.56,
        key_token="k_alpha",
        is_backspace=False,
        is_enter=False,
        is_space=False,
    )
    d = event.to_dict()

    assert d["event_type"] == "down"
    assert d["timestamp_ms"] == 1234.56
    assert d["key_token"] == "k_alpha"
    assert not d["is_backspace"]
    # Ensure no character string or text field exists
    assert "key" not in d
    assert "char" not in d
    assert "text" not in d
    assert "value" not in d


def test_zero_global_hooks_compliance():
    """Verify absence of prohibited system-level keylogging libraries."""
    import sys

    # Assert pynput or keyboard are not imported or required
    assert "pynput" not in sys.modules
    assert "keyboard" not in sys.modules


def test_abstract_key_tokens_sanitization():
    """Verify that only abstract categories are accepted and raw characters are rejected."""
    # Valid abstract tokens
    valid_alpha = {"event_type": "down", "timestamp_ms": 100.0, "key_token": "k_alpha"}
    assert validate_browser_event(valid_alpha) is not None

    valid_bk = {"event_type": "down", "timestamp_ms": 150.0, "key_token": "k_backspace", "is_backspace": True}
    assert validate_browser_event(valid_bk) is not None

    # Invalid: single raw character leaked without k_ prefix
    raw_char = {"event_type": "down", "timestamp_ms": 200.0, "key_token": "a"}
    assert validate_browser_event(raw_char) is None
