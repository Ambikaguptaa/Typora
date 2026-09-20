"""Unit tests for live typing session lifecycle and pseudonymous duration tracking."""

import time
import pytest
from src.live_typing.session import LiveTypingSession, SessionStatus
from src.live_typing.validation import validate_session_quality


def test_session_lifecycle_transitions():
    """Verify state transitions: IDLE -> ACTIVE -> PAUSED -> ACTIVE -> COMPLETED -> IDLE."""
    session = LiveTypingSession()
    assert session.status == SessionStatus.IDLE
    assert session.session_id.startswith("sess_")

    # Start
    session.start()
    assert session.status == SessionStatus.ACTIVE
    assert session.start_time is not None

    # Pause
    session.pause()
    assert session.status == SessionStatus.PAUSED

    # Resume
    session.resume()
    assert session.status == SessionStatus.ACTIVE

    # Stop
    session.stop()
    assert session.status == SessionStatus.COMPLETED
    assert session.end_time is not None

    # Reset
    old_id = session.session_id
    session.reset()
    assert session.status == SessionStatus.IDLE
    assert session.session_id != old_id  # Generates fresh session ID
    assert len(session.get_paired_events()) == 0


def test_session_active_duration_accounting():
    """Verify session duration computes active typing time excluding paused periods."""
    session = LiveTypingSession()
    session.start()
    time.sleep(0.05)  # 50ms active

    session.pause()
    time.sleep(0.05)  # 50ms paused (should be deducted)

    session.resume()
    time.sleep(0.05)  # 50ms active

    session.stop()
    dur = session.duration_seconds
    assert dur >= 0.08  # At least the active portion
    assert dur < 0.25   # Not over counting paused time


def test_session_quality_validation():
    """Verify validation flags sessions with insufficient events or duration."""
    session = LiveTypingSession()
    session.start()

    # Empty session
    is_valid, verdict, details = validate_session_quality(session)
    assert not is_valid
    assert verdict == "INSUFFICIENT_DATA"
    assert len(details["reasons"]) >= 2
