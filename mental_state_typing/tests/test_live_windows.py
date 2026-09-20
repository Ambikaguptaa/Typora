"""Unit tests for live temporal sequence windowing and tensor preparation."""

import pytest
import numpy as np
from src.live_typing.event_types import TypingEvent
from src.live_typing.feature_buffer import LiveFeatureBuffer


def _build_synthetic_events(n: int) -> list[TypingEvent]:
    events = []
    t = 1000.0
    for i in range(n):
        dwell = 90.0
        flight = 110.0 if i > 0 else None
        p = t
        r = p + dwell
        events.append(
            TypingEvent(
                sequence_index=i,
                press_timestamp=p,
                release_timestamp=r,
                dwell_time=dwell,
                flight_time=flight,
                key_token="k_alpha",
            )
        )
        t = r + 110.0
    return events


def test_windowing_insufficient_events():
    """Verify window generator returns empty array when events < sequence_length."""
    buffer = LiveFeatureBuffer(sequence_length=30)
    buffer.add_events(_build_synthetic_events(20))  # 20 < 30

    X, meta = buffer.generate_sequence_windows(sequence_length=30)
    assert X.shape == (0, 30, 6)
    assert len(meta) == 0


def test_windowing_exact_and_multiple_windows():
    """Verify window generator produces correct 3D array shape (N, 30, 6) with stride."""
    buffer = LiveFeatureBuffer(sequence_length=30, sequence_stride=10)
    # 50 events with length 30 and stride 10 -> start indices 0, 10, 20 -> 3 windows
    buffer.add_events(_build_synthetic_events(50))

    X, meta = buffer.generate_sequence_windows(sequence_length=30, sequence_stride=10)
    assert isinstance(X, np.ndarray)
    assert X.shape == (3, 30, 6)
    assert len(meta) == 3
    assert meta[0]["start_event_idx"] == 0
    assert meta[0]["end_event_idx"] == 30
    assert meta[1]["start_event_idx"] == 10
    assert meta[2]["start_event_idx"] == 20
