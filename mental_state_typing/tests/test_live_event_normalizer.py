"""Unit tests for live typing event normalizer and pairing engine."""

import pytest
from src.live_typing.event_normalizer import LiveEventNormalizer
from src.live_typing.event_types import RawBrowserEvent, TypingEvent


def test_normalizer_pairs_down_up_events():
    """Verify normalizer accurately pairs KEY_DOWN and KEY_UP into a TypingEvent."""
    normalizer = LiveEventNormalizer()

    down = RawBrowserEvent(event_type="down", timestamp_ms=100.0, key_token="k_alpha")
    up = RawBrowserEvent(event_type="up", timestamp_ms=180.0, key_token="k_alpha")

    res_down = normalizer.process_event(down)
    assert res_down is None  # Pending keyup

    res_up = normalizer.process_event(up)
    assert res_up is not None
    assert isinstance(res_up, TypingEvent)
    assert res_up.dwell_time == 80.0
    assert res_up.flight_time is None  # First event in sequence has no flight time
    assert res_up.sequence_index == 0


def test_normalizer_computes_flight_time_across_keys():
    """Verify flight time is computed as consecutive key press minus prior key release."""
    normalizer = LiveEventNormalizer()

    # Key 1: down at 100, up at 180 (dwell = 80)
    normalizer.process_event(RawBrowserEvent(event_type="down", timestamp_ms=100.0, key_token="k_alpha"))
    ev1 = normalizer.process_event(RawBrowserEvent(event_type="up", timestamp_ms=180.0, key_token="k_alpha"))

    # Key 2: down at 250, up at 330 (flight = 250 - 180 = 70, dwell = 80)
    normalizer.process_event(RawBrowserEvent(event_type="down", timestamp_ms=250.0, key_token="k_space", is_space=True))
    ev2 = normalizer.process_event(RawBrowserEvent(event_type="up", timestamp_ms=330.0, key_token="k_space", is_space=True))

    assert ev1 is not None
    assert ev2 is not None
    assert ev2.dwell_time == 80.0
    assert ev2.flight_time == 70.0
    assert ev2.sequence_index == 1
    assert ev2.is_space


def test_normalizer_discards_negative_and_outlier_dwells():
    """Verify normalizer rejects negative dwell times and implausibly high values."""
    normalizer = LiveEventNormalizer(max_dwell_ms=2000.0)

    # Negative dwell (release before press)
    normalizer.process_event(RawBrowserEvent(event_type="down", timestamp_ms=500.0, key_token="k_alpha"))
    ev_neg = normalizer.process_event(RawBrowserEvent(event_type="up", timestamp_ms=450.0, key_token="k_alpha"))
    assert ev_neg is None

    # Outlier dwell (5000ms > max 2000ms)
    normalizer.process_event(RawBrowserEvent(event_type="down", timestamp_ms=1000.0, key_token="k_alpha"))
    ev_out = normalizer.process_event(RawBrowserEvent(event_type="up", timestamp_ms=6500.0, key_token="k_alpha"))
    assert ev_out is None


def test_normalizer_detects_correction_bursts():
    """Verify rapid consecutive backspaces are flagged as correction bursts."""
    normalizer = LiveEventNormalizer()

    # First backspace
    normalizer.process_event(RawBrowserEvent(event_type="down", timestamp_ms=100.0, key_token="k_backspace", is_backspace=True))
    bk1 = normalizer.process_event(RawBrowserEvent(event_type="up", timestamp_ms=180.0, key_token="k_backspace", is_backspace=True))
    assert bk1.is_backspace
    assert not bk1.is_correction

    # Second backspace rapidly after (press at 250ms -> delta 150ms < 400ms)
    normalizer.process_event(RawBrowserEvent(event_type="down", timestamp_ms=250.0, key_token="k_backspace", is_backspace=True))
    bk2 = normalizer.process_event(RawBrowserEvent(event_type="up", timestamp_ms=320.0, key_token="k_backspace", is_backspace=True))
    assert bk2.is_backspace
    assert bk2.is_correction
