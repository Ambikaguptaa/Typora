"""Unit tests for live typing feature extraction, telemetry, and buffer management."""

import pytest
import numpy as np
from src.live_typing.event_types import TypingEvent
from src.live_typing.feature_buffer import CANONICAL_SEQUENCE_FEATURES, LiveFeatureBuffer


def _create_sample_events(count: int = 20) -> list[TypingEvent]:
    """Helper to generate a sequence of valid normalized TypingEvents."""
    events = []
    current_time = 1000.0
    for i in range(count):
        dwell = 80.0 + (i % 5) * 5.0  # 80 to 100ms
        flight = 120.0 + (i % 4) * 10.0 if i > 0 else None
        press = current_time
        release = press + dwell
        is_bk = (i == 10)  # 1 backspace
        is_sp = (i % 6 == 0)
        events.append(
            TypingEvent(
                sequence_index=i,
                press_timestamp=press,
                release_timestamp=release,
                dwell_time=dwell,
                flight_time=flight,
                key_token="k_backspace" if is_bk else ("k_space" if is_sp else "k_alpha"),
                is_backspace=is_bk,
                is_space=is_sp,
                is_correction=False,
            )
        )
        current_time = release + (120.0 if flight is not None else 0.0)
    return events


def test_feature_buffer_telemetry_calculation():
    """Verify live telemetry correctly aggregates dwell, flight, WPM, and error counts."""
    buffer = LiveFeatureBuffer()
    events = _create_sample_events(25)
    buffer.add_events(events)

    telemetry = buffer.extract_telemetry()
    assert telemetry["event_count"] == 25
    assert 80.0 <= telemetry["mean_dwell_ms"] <= 100.0
    assert 120.0 <= telemetry["mean_flight_ms"] <= 150.0
    assert telemetry["backspace_count"] == 1
    assert telemetry["estimated_wpm"] > 0.0
    assert telemetry["duration_seconds"] > 0.0


def test_feature_dataframe_matches_canonical_schema():
    """Verify DataFrame conversion contains required canonical sequence feature columns."""
    buffer = LiveFeatureBuffer()
    events = _create_sample_events(15)
    buffer.add_events(events)

    df = buffer.to_dataframe(user_id="usr_test", session_id="sess_test")
    assert len(df) == 15
    for col in CANONICAL_SEQUENCE_FEATURES:
        assert col in df.columns
    assert df["user_id"].iloc[0] == "usr_test"
    assert df["session_id"].iloc[0] == "sess_test"


def test_buffer_capacity_eviction():
    """Verify FIFO eviction when buffer capacity is reached."""
    buffer = LiveFeatureBuffer(max_buffer_size=10)
    events = _create_sample_events(15)
    buffer.add_events(events)

    assert buffer.event_count == 10
    buffered = buffer.get_events()
    # Oldest 5 events should be evicted; sequence_index of first remaining is 5
    assert buffered[0].sequence_index == 5
