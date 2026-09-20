"""Live typing event normalizer and down/up event pairing engine.

Pairs sequential KEY_DOWN and KEY_UP events, computes physiological micro-timings
(dwell time, flight time), and detects editing/correction patterns without storing text.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from src.live_typing.event_types import KeyEventType, RawBrowserEvent, TypingEvent


class LiveEventNormalizer:
    """Stateful normalizer that pairs browser down/up keystroke events into TypingEvents."""

    def __init__(self, max_dwell_ms: float = 4000.0, min_dwell_ms: float = 10.0):
        self.max_dwell_ms = max_dwell_ms
        self.min_dwell_ms = min_dwell_ms

        # Pending keydowns: key_token -> (press_timestamp, RawBrowserEvent)
        self._pending_downs: Dict[str, Tuple[float, RawBrowserEvent]] = {}
        self._previous_release_timestamp: Optional[float] = None
        self._sequence_index: int = 0
        self._last_backspace_time: Optional[float] = None

    def reset(self) -> None:
        """Clear normalizer internal tracking state."""
        self._pending_downs.clear()
        self._previous_release_timestamp = None
        self._sequence_index = 0
        self._last_backspace_time = None

    def process_event(self, raw_event: RawBrowserEvent) -> Optional[TypingEvent]:
        """Ingest a single sanitized browser event and return a paired TypingEvent if complete.

        Args:
            raw_event: Sanitized RawBrowserEvent.

        Returns:
            Optional[TypingEvent]: Complete paired event, or None if pending keyup or invalid.
        """
        token = raw_event.key_token
        t = raw_event.timestamp_ms

        if raw_event.event_type == KeyEventType.KEY_DOWN.value:
            # If keydown already pending for this token (e.g. OS auto-repeat or missed keyup),
            # replace or update pending timestamp
            self._pending_downs[token] = (t, raw_event)
            return None

        elif raw_event.event_type == KeyEventType.KEY_UP.value:
            if token not in self._pending_downs:
                # Keyup without observed keydown (e.g. key pressed before focus was gained)
                return None

            press_time, down_event = self._pending_downs.pop(token)
            release_time = t

            dwell = release_time - press_time
            # Reject negative dwell (browser clock jitter) or implausibly extended dwell
            if dwell < 0.0 or dwell > self.max_dwell_ms:
                return None

            # Calculate flight time from previous keystroke's release
            flight: Optional[float] = None
            if self._previous_release_timestamp is not None:
                raw_flight = press_time - self._previous_release_timestamp
                # In rollover typing, press can occur slightly before previous release; clamp to 0.0
                flight = max(0.0, float(raw_flight))

            # Correction detection: rapid backspace burst (< 400ms) or backspace sequence
            is_correction = False
            if raw_event.is_backspace:
                if self._last_backspace_time is not None and (press_time - self._last_backspace_time) < 400.0:
                    is_correction = True
                self._last_backspace_time = press_time

            paired_event = TypingEvent(
                sequence_index=self._sequence_index,
                press_timestamp=press_time,
                release_timestamp=release_time,
                dwell_time=dwell,
                flight_time=flight,
                key_token=token,
                is_backspace=raw_event.is_backspace,
                is_enter=raw_event.is_enter,
                is_space=raw_event.is_space,
                is_correction=is_correction,
            )

            # Advance tracking
            self._previous_release_timestamp = release_time
            self._sequence_index += 1

            return paired_event

        return None

    def process_batch(self, events: List[RawBrowserEvent]) -> List[TypingEvent]:
        """Process a batch of sanitized browser events in chronological order.

        Args:
            events: List of RawBrowserEvents.

        Returns:
            List[TypingEvent]: Normalized paired events.
        """
        # Sort batch chronologically to handle any asynchronous network jitter
        sorted_events = sorted(events, key=lambda ev: ev.timestamp_ms)
        paired_list: List[TypingEvent] = []

        for ev in sorted_events:
            paired = self.process_event(ev)
            if paired is not None:
                paired_list.append(paired)

        return paired_list
