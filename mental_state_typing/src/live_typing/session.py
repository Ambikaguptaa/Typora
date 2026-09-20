"""Session lifecycle management for live typing behavior capture.

Manages session transitions: START, PAUSE, RESUME, STOP, and RESET.
Tracks pseudonymous session identifiers, session duration, and buffered events in memory.
"""

from enum import Enum
import time
from typing import Any, Dict, List, Optional
import uuid

from src.live_typing.event_normalizer import LiveEventNormalizer
from src.live_typing.event_types import RawBrowserEvent, TypingEvent
from src.live_typing.privacy_filter import sanitize_event_batch


class SessionStatus(str, Enum):
    """Lifecycle states for a live typing test session."""

    IDLE = "IDLE"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class LiveTypingSession:
    """Manages an active or completed in-memory typing capture session."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id: str = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        self.status: SessionStatus = SessionStatus.IDLE
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self._pause_time: Optional[float] = None
        self._total_paused_duration: float = 0.0

        self.normalizer = LiveEventNormalizer()
        self._raw_event_count: int = 0
        self._paired_events: List[TypingEvent] = []

    def start(self) -> None:
        """Start or restart a typing session."""
        self.status = SessionStatus.ACTIVE
        self.start_time = time.time()
        self.end_time = None
        self._pause_time = None
        self._total_paused_duration = 0.0
        self._raw_event_count = 0
        self._paired_events.clear()
        self.normalizer.reset()

    def pause(self) -> None:
        """Pause an active session."""
        if self.status == SessionStatus.ACTIVE:
            self.status = SessionStatus.PAUSED
            self._pause_time = time.time()

    def resume(self) -> None:
        """Resume a paused session."""
        if self.status == SessionStatus.PAUSED:
            if self._pause_time is not None:
                self._total_paused_duration += time.time() - self._pause_time
                self._pause_time = None
            self.status = SessionStatus.ACTIVE

    def stop(self) -> None:
        """Stop and finalize the current session."""
        if self.status in (SessionStatus.ACTIVE, SessionStatus.PAUSED):
            self.end_time = time.time()
            self.status = SessionStatus.COMPLETED

    def reset(self) -> None:
        """Completely reset the session buffer and generate a clean session ID."""
        self.session_id = f"sess_{uuid.uuid4().hex[:12]}"
        self.status = SessionStatus.IDLE
        self.start_time = None
        self.end_time = None
        self._pause_time = None
        self._total_paused_duration = 0.0
        self._raw_event_count = 0
        self._paired_events.clear()
        self.normalizer.reset()

    def ingest_browser_batch(
        self,
        raw_events: List[Dict[str, Any]],
        strict_privacy: bool = False,
    ) -> int:
        """Ingest, filter, and normalize a batch of raw browser events.

        Args:
            raw_events: List of raw dictionaries from the browser bridge.
            strict_privacy: If True, raises PrivacyViolationError if forbidden fields exist.

        Returns:
            int: Number of new paired TypingEvents appended.
        """
        if self.status != SessionStatus.ACTIVE:
            # Events received while idle/paused/stopped are ignored
            return 0

        # Step 1: Privacy filter sanitization
        clean_events, rejected_count = sanitize_event_batch(raw_events, strict_raise=strict_privacy)
        self._raw_event_count += len(clean_events)

        # Step 2: Normalization & pairing
        paired = self.normalizer.process_batch(clean_events)
        self._paired_events.extend(paired)

        return len(paired)

    @property
    def duration_seconds(self) -> float:
        """Total active elapsed duration in seconds excluding pauses."""
        if self.start_time is None:
            return 0.0

        if self.end_time is not None:
            total_elapsed = self.end_time - self.start_time
        elif self.status == SessionStatus.PAUSED and self._pause_time is not None:
            total_elapsed = self._pause_time - self.start_time
        else:
            total_elapsed = time.time() - self.start_time

        return max(0.0, total_elapsed - self._total_paused_duration)

    @property
    def event_count(self) -> int:
        """Total raw browser events ingested."""
        return self._raw_event_count

    @property
    def valid_pair_count(self) -> int:
        """Count of successfully paired TypingEvents."""
        return len(self._paired_events)

    def get_paired_events(self) -> List[TypingEvent]:
        """Return a copy of currently buffered paired events."""
        return list(self._paired_events)

    def get_active_duration_seconds(self) -> float:
        """Return total active elapsed duration in seconds."""
        return self.duration_seconds

    def get_summary(self) -> Dict[str, Any]:
        """Return a structured summary of the session."""
        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "duration_seconds": round(self.duration_seconds, 2),
            "raw_event_count": self._raw_event_count,
            "valid_pair_count": len(self._paired_events),
            "is_active": self.status == SessionStatus.ACTIVE,
        }
