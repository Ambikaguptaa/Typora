"""Event type definitions and models for the live typing capture engine.

ZERO-RAW-TEXT INVARIANT:
No character identity, typed string, word, sentence, or input content is ever
represented in these data models. All events are purely temporal and categorical.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional
import numpy as np


class KeyEventType(str, Enum):
    """Categorical event types emitted during typing."""

    KEY_DOWN = "down"
    KEY_UP = "up"


@dataclass(frozen=True)
class RawBrowserEvent:
    """Sanitized raw event emitted from browser keystroke listeners.

    Contains exclusively timing and safe categorical flags. Zero character strings.
    """

    event_type: str  # 'down' or 'up'
    timestamp_ms: float  # High-resolution millisecond timestamp
    key_token: str  # Abstract token (e.g. 'k_alpha', 'k_backspace', 'k_space', 'k_enter', 'k_other')
    is_backspace: bool = False
    is_enter: bool = False
    is_space: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        return {
            "event_type": self.event_type,
            "timestamp_ms": self.timestamp_ms,
            "key_token": self.key_token,
            "is_backspace": self.is_backspace,
            "is_enter": self.is_enter,
            "is_space": self.is_space,
        }


@dataclass(frozen=True)
class TypingEvent:
    """Normalized keystroke event with paired down/up timestamps and micro-timings.

    Zero text content is preserved.
    """

    sequence_index: int
    press_timestamp: float
    release_timestamp: float
    dwell_time: float
    flight_time: Optional[float]  # None / NaN for the very first event in a session
    key_token: str  # Abstract token (e.g. 'k_alpha', 'k_backspace')
    is_backspace: bool = False
    is_enter: bool = False
    is_space: bool = False
    is_correction: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert normalized event to dictionary representation."""
        return {
            "sequence_index": self.sequence_index,
            "press_timestamp": self.press_timestamp,
            "release_timestamp": self.release_timestamp,
            "dwell_time": self.dwell_time,
            "flight_time": self.flight_time if self.flight_time is not None else np.nan,
            "key_token": self.key_token,
            "is_backspace": self.is_backspace,
            "is_enter": self.is_enter,
            "is_space": self.is_space,
            "is_correction": self.is_correction,
        }
