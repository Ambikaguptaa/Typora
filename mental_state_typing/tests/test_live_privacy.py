"""Unit tests for live typing privacy filtration and zero-raw-text invariant enforcement."""

import pytest
from src.live_typing.privacy_filter import (
    FORBIDDEN_PAYLOAD_FIELDS,
    PrivacyViolationError,
    audit_payload_for_sensitive_keys,
    sanitize_event_batch,
    validate_browser_event,
)


def test_audit_detects_forbidden_keys():
    """Verify audit_payload_for_sensitive_keys flags prohibited fields."""
    clean_payload = {"event_type": "down", "timestamp_ms": 100.0, "key_token": "k_alpha"}
    assert len(audit_payload_for_sensitive_keys(clean_payload)) == 0

    dirty_payload = {
        "event_type": "down",
        "timestamp_ms": 100.0,
        "key_token": "k_alpha",
        "key": "A",
        "text": "Hello world",
        "password": "secret",
    }
    violations = audit_payload_for_sensitive_keys(dirty_payload)
    assert "key" in violations
    assert "text" in violations
    assert "password" in violations


def test_sanitize_event_batch_rejects_malicious_payload():
    """Verify sanitize_event_batch filters out entries with forbidden fields."""
    batch = [
        {"event_type": "down", "timestamp_ms": 100.0, "key_token": "k_alpha"},
        {"event_type": "down", "timestamp_ms": 150.0, "key_token": "k_alpha", "char": "x"},  # Dirty
        {"event_type": "up", "timestamp_ms": 180.0, "key_token": "k_alpha"},
    ]

    sanitized, rejected_count = sanitize_event_batch(batch, strict_raise=False)
    assert len(sanitized) == 2
    assert rejected_count == 1
    # Verify no sanitized event contains sensitive fields
    for ev in sanitized:
        assert not hasattr(ev, "char")
        assert not hasattr(ev, "key")
        assert not hasattr(ev, "text")


def test_strict_mode_raises_privacy_violation():
    """Verify strict mode raises PrivacyViolationError immediately upon violation."""
    malicious_event = {
        "event_type": "down",
        "timestamp_ms": 100.0,
        "key_token": "k_alpha",
        "typed_text": "Sensitive message",
    }

    with pytest.raises(PrivacyViolationError) as exc_info:
        validate_browser_event(malicious_event, strict_raise=True)
    assert "Forbidden keys detected" in str(exc_info.value)


def test_forbidden_keys_catalog_comprehensiveness():
    """Verify FORBIDDEN_PAYLOAD_FIELDS covers all standard sensitive keystroke attributes."""
    expected = ["key", "char", "character", "text", "word", "sentence", "message", "value", "password"]
    for field in expected:
        assert field in FORBIDDEN_PAYLOAD_FIELDS
