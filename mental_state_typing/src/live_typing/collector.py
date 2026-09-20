"""Live typing event collector interface.

Defines the structure for capturing relative timing events without recording
the actual key identities or written text.
"""

from dataclasses import dataclass
from typing import List, Optional
import time


@dataclass(frozen=True)
class KeystrokeTimingEvent:
    """Represents an anonymized keystroke timing event.

    Strictly records timestamps, omitting any character or key name information.
    """

    press_time_ms: float
    release_time_ms: float

    @property
    def hold_time_ms(self) -> float:
        """Hold duration (dwell time) in milliseconds."""
        return max(0.0, self.release_time_ms - self.press_time_ms)


class TimingCollector:
    """In-memory collector buffer for anonymous timing events."""

    def __init__(self, buffer_size: int = 200):
        self.buffer_size = buffer_size
        self._events: List[KeystrokeTimingEvent] = []

    def record_event(self, press_time_ms: float, release_time_ms: float) -> None:
        """Add an event to the buffer, maintaining maximum capacity."""
        event = KeystrokeTimingEvent(
            press_time_ms=press_time_ms,
            release_time_ms=release_time_ms,
        )
        self._events.append(event)
        if len(self._events) > self.buffer_size:
            self._events.pop(0)

    def get_events(self) -> List[KeystrokeTimingEvent]:
        """Return a copy of currently buffered events."""
        return list(self._events)

    def clear(self) -> None:
        """Flush the buffer."""
        self._events.clear()

    def count(self) -> int:
        """Number of events currently held."""
        return len(self._events)
