"""Data quality validation for live typing capture sessions.

Applies configurable technical quality thresholds to ensure that only sessions with
sufficient micro-timing observations are submitted to downstream feature buffers
or sequence models.

NON-DIAGNOSTIC RULE:
If a session contains insufficient events, its status is strictly INSUFFICIENT_DATA.
It is never classified as low strain, normal, or any psychological state.
"""

from typing import Any, Dict, List, Optional, Tuple

from src.live_typing.session import LiveTypingSession, SessionStatus

DEFAULT_MIN_EVENTS: int = 15
DEFAULT_MIN_DURATION_SECONDS: float = 3.0
DEFAULT_MIN_VALID_DWELL_EVENTS: int = 5
DEFAULT_MIN_VALID_FLIGHT_EVENTS: int = 4


def validate_session_quality(
    session: LiveTypingSession,
    min_events: int = DEFAULT_MIN_EVENTS,
    min_duration_seconds: float = DEFAULT_MIN_DURATION_SECONDS,
    min_valid_dwell_events: int = DEFAULT_MIN_VALID_DWELL_EVENTS,
    min_valid_flight_events: int = DEFAULT_MIN_VALID_FLIGHT_EVENTS,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Evaluate whether a live typing session satisfies technical data quality thresholds.

    Args:
        session: Active or completed LiveTypingSession.
        min_events: Minimum total paired keystroke events required.
        min_duration_seconds: Minimum active session duration in seconds.
        min_valid_dwell_events: Minimum events with positive dwell times.
        min_valid_flight_events: Minimum events with valid non-null flight times.

    Returns:
        Tuple[bool, str, Dict[str, Any]]:
            - is_valid (bool): True if all quality criteria are met.
            - verdict (str): 'SESSION_VALID' or 'INSUFFICIENT_DATA'.
            - details (Dict[str, Any]): Breakdown of actual measurements and thresholds.
    """
    paired_events = session.get_paired_events()
    total_events = len(paired_events)
    duration = session.duration_seconds

    valid_dwell_count = sum(1 for ev in paired_events if ev.dwell_time > 0.0)
    valid_flight_count = sum(1 for ev in paired_events if ev.flight_time is not None and ev.flight_time >= 0.0)

    reasons: List[str] = []

    if total_events < min_events:
        reasons.append(
            f"Insufficient events: {total_events} recorded (minimum {min_events} required)."
        )

    if duration < min_duration_seconds:
        reasons.append(
            f"Duration too short: {duration:.1f}s elapsed (minimum {min_duration_seconds:.1f}s required)."
        )

    if valid_dwell_count < min_valid_dwell_events:
        reasons.append(
            f"Insufficient valid dwell times: {valid_dwell_count} (minimum {min_valid_dwell_events} required)."
        )

    if valid_flight_count < min_valid_flight_events:
        reasons.append(
            f"Insufficient valid flight times: {valid_flight_count} (minimum {min_valid_flight_events} required)."
        )

    is_valid = len(reasons) == 0
    verdict = "SESSION_VALID" if is_valid else "INSUFFICIENT_DATA"

    details = {
        "is_valid": is_valid,
        "verdict": verdict,
        "session_id": session.session_id,
        "metrics": {
            "total_paired_events": total_events,
            "duration_seconds": round(duration, 2),
            "valid_dwell_count": valid_dwell_count,
            "valid_flight_count": valid_flight_count,
        },
        "thresholds": {
            "min_events": min_events,
            "min_duration_seconds": min_duration_seconds,
            "min_valid_dwell_events": min_valid_dwell_events,
            "min_valid_flight_events": min_valid_flight_events,
        },
        "reasons": reasons,
    }

    return is_valid, verdict, details
