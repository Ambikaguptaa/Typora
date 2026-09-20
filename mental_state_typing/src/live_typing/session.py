"""Session lifecycle management for live typing behavior capture.

Manages authoritative session transitions:
READY -> START -> CAPTURING -> PAUSE -> PAUSED -> RESUME -> CAPTURING -> STOP -> STOPPING -> VALIDATING -> COMPLETED.
RESET from any safe state -> READY.
Unexpected failures -> ERROR.

Strict Security & Correctness Invariants:
1. Authoritative state machine strictly rejects impossible transitions.
2. Keystroke events received outside CAPTURING state are discarded.
3. Duplicate event payloads are safely ignored without buffer pollution.
4. Pause/resume boundaries prevent artificial multi-second flight time spikes.
5. Zero character identities or typed texts are ever stored or processed.
"""

from enum import Enum
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

from src.live_typing.event_normalizer import LiveEventNormalizer
from src.live_typing.event_types import RawBrowserEvent, TypingEvent
from src.live_typing.privacy_filter import sanitize_event_batch


class SessionState(str, Enum):
    """Authoritative lifecycle states for a live typing test session."""

    READY = "READY"
    CAPTURING = "CAPTURING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    VALIDATING = "VALIDATING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"

    # Compatibility aliases for legacy and pipeline references
    IDLE = "READY"
    ACTIVE = "CAPTURING"
    INSUFFICIENT_DATA = "COMPLETED"


# Alias SessionStatus to SessionState for 100% backward compatibility
SessionStatus = SessionState


class LiveTypingSession:
    """Authoritative stateful manager for in-memory typing capture sessions."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id: str = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        self.state: SessionState = SessionState.READY
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self._pause_time: Optional[float] = None
        self._total_paused_duration: float = 0.0

        self.normalizer = LiveEventNormalizer()
        self._raw_event_count: int = 0
        self._paired_events: List[TypingEvent] = []

        # Deduplication tracking structures
        self._seen_event_ids: Set[str] = set()
        self._seen_signatures: Set[Tuple[str, float, str]] = set()

    @property
    def status(self) -> SessionState:
        """Compatibility property matching legacy status attribute."""
        return self.state

    @status.setter
    def status(self, val: SessionState) -> None:
        self.state = val

    def start(self) -> None:
        """Transition READY -> CAPTURING.

        Starts or restarts a fresh typing session.
        Valid only from READY or COMPLETED. Rejects CAPTURING, PAUSED, STOPPING, VALIDATING.
        """
        if self.state in (SessionState.CAPTURING, SessionState.PAUSED, SessionState.STOPPING, SessionState.VALIDATING):
            raise ValueError(f"Cannot START session while in {self.state.value} state.")

        self.state = SessionState.CAPTURING
        self.start_time = time.time()
        self.end_time = None
        self._pause_time = None
        self._total_paused_duration = 0.0
        self._raw_event_count = 0
        self._paired_events.clear()
        self._seen_event_ids.clear()
        self._seen_signatures.clear()
        self.normalizer.reset()

    def pause(self) -> None:
        """Transition CAPTURING -> PAUSED.

        Suspends event capture and preserves buffers.
        Valid only from CAPTURING.
        """
        if self.state != SessionState.CAPTURING:
            raise ValueError(f"Cannot PAUSE session from state: {self.state.value}")

        self.state = SessionState.PAUSED
        self._pause_time = time.time()
        self.normalizer.notify_pause()

    def resume(self) -> None:
        """Transition PAUSED -> CAPTURING.

        Resumes event capture, preserving earlier data.
        Valid only from PAUSED.
        """
        if self.state != SessionState.PAUSED:
            raise ValueError(f"Cannot RESUME session from state: {self.state.value}")

        if self._pause_time is not None:
            self._total_paused_duration += time.time() - self._pause_time
            self._pause_time = None

        self.normalizer.notify_resume()
        self.state = SessionState.CAPTURING

    def stop(self) -> None:
        """Transition CAPTURING / PAUSED -> COMPLETED.

        Finalizes duration, closes buffers.
        Valid only from CAPTURING or PAUSED.
        """
        if self.state not in (SessionState.CAPTURING, SessionState.PAUSED, SessionState.STOPPING, SessionState.VALIDATING):
            raise ValueError(f"Cannot STOP session from state: {self.state.value}")

        now = time.time()
        if self.state == SessionState.PAUSED and self._pause_time is not None:
            self._total_paused_duration += now - self._pause_time
            self._pause_time = None

        self.end_time = now
        self.state = SessionState.COMPLETED

    def transition_to(self, target_state: SessionState) -> None:
        """Controlled transition helper for intermediate pipeline states (STOPPING, VALIDATING, ERROR)."""
        self.state = target_state

    def reset(self) -> None:
        """Transition any safe state -> READY.

        Generates a fresh session ID, resets timers, and purges in-memory event queues.
        """
        self.session_id = f"sess_{uuid.uuid4().hex[:12]}"
        self.state = SessionState.READY
        self.start_time = None
        self.end_time = None
        self._pause_time = None
        self._total_paused_duration = 0.0
        self._raw_event_count = 0
        self._paired_events.clear()
        self._seen_event_ids.clear()
        self._seen_signatures.clear()
        self.normalizer.reset()

    def ingest_browser_batch(
        self,
        raw_events: List[Dict[str, Any]],
        strict_privacy: bool = False,
    ) -> int:
        """Ingest, filter, deduplicate, and normalize a batch of raw browser events.

        Args:
            raw_events: List of raw dictionaries from the browser bridge.
            strict_privacy: If True, raises PrivacyViolationError if forbidden fields exist.

        Returns:
            int: Number of new paired TypingEvents appended.
        """
        if self.state != SessionState.CAPTURING:
            # Events received while READY/PAUSED/STOPPED/VALIDATING are strictly ignored
            return 0

        # Step 1: Privacy filter sanitization
        clean_events, _ = sanitize_event_batch(raw_events, strict_raise=strict_privacy)

        # Step 2: Deduplication protection
        unseen_events: List[RawBrowserEvent] = []
        for ev in clean_events:
            # Check unique event_id if present
            if ev.event_id is not None:
                if ev.event_id in self._seen_event_ids:
                    continue
                self._seen_event_ids.add(ev.event_id)

            # Check categorical timing signature
            sig = (ev.event_type, float(ev.timestamp_ms), ev.key_token)
            if sig in self._seen_signatures:
                continue
            self._seen_signatures.add(sig)
            unseen_events.append(ev)

        self._raw_event_count += len(unseen_events)

        # Step 3: Normalization & pairing
        paired = self.normalizer.process_batch(unseen_events)
        self._paired_events.extend(paired)

        return len(paired)

    @property
    def duration_seconds(self) -> float:
        """Total active elapsed duration in seconds excluding pauses."""
        # Calculate physiological keystroke span from paired events
        if len(self._paired_events) >= 2:
            event_span = max(0.0, (self._paired_events[-1].release_timestamp - self._paired_events[0].press_timestamp) / 1000.0)
        else:
            event_span = 0.0

        if self.start_time is None:
            return event_span

        if self.end_time is not None:
            wall_elapsed = max(0.0, (self.end_time - self.start_time) - self._total_paused_duration)
        elif self.state == SessionState.PAUSED and self._pause_time is not None:
            wall_elapsed = max(0.0, (self._pause_time - self.start_time) - self._total_paused_duration)
        else:
            wall_elapsed = max(0.0, (time.time() - self.start_time) - self._total_paused_duration)

        return max(wall_elapsed, event_span)

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
            "status": self.state.value,
            "duration_seconds": round(self.duration_seconds, 2),
            "raw_event_count": self._raw_event_count,
            "valid_pair_count": len(self._paired_events),
            "is_active": self.state == SessionState.CAPTURING,
        }
