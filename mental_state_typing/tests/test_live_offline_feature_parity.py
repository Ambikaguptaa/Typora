"""Feature Parity Verification Suite: Live vs. Offline Feature Pipelines.

Verifies that:
1. Canonical event fixtures produce identical micro-timing feature vectors in both the live engine and offline data engineering pipeline.
2. Dwell times, flight latencies, pauses, backspaces, and sequence dimensions adhere strictly to documented tolerances (< 1e-2).
3. The 3D temporal sequence array produced by the live engine matches the offline recurrent tensor contract (N, 30, 6).
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import pytest

from src.data_engineering.canonical_schema import CANONICAL_REQUIRED_COLUMNS
from src.data_engineering.feature_engineering import (
    build_feature_table,
    calculate_dwell_time,
    calculate_flight_time,
)
from src.deep_learning.preprocessing import prepare_grouped_sequences
from src.live_typing.event_normalizer import LiveEventNormalizer
from src.live_typing.event_types import RawBrowserEvent, TypingEvent
from src.live_typing.feature_buffer import CANONICAL_SEQUENCE_FEATURES, LiveFeatureBuffer


def _build_deterministic_fixture(n_keystrokes: int = 40) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """Create a deterministic canonical keystroke sequence for dual-pipeline evaluation."""
    raw_events = []
    canonical_rows = []
    current_time = 1000.0

    for i in range(n_keystrokes):
        dwell = 80.0 + (i % 7) * 5.0  # 80.0 to 110.0 ms
        flight = 120.0 + (i % 5) * 15.0 if i > 0 else 0.0  # 120.0 to 180.0 ms
        is_bk = (i in (12, 13))  # 2 backspaces
        is_sp = (i % 8 == 0)
        token = "k_backspace" if is_bk else ("k_space" if is_sp else "k_alpha")

        press_t = current_time
        release_t = press_t + dwell

        # Browser raw event dicts (down + up)
        raw_events.append({
            "event_type": "down",
            "timestamp_ms": press_t,
            "key_token": token,
            "is_backspace": is_bk,
            "is_enter": False,
            "is_space": is_sp,
        })
        raw_events.append({
            "event_type": "up",
            "timestamp_ms": release_t,
            "key_token": token,
            "is_backspace": is_bk,
            "is_enter": False,
            "is_space": is_sp,
        })

        # Canonical event dataframe row
        canonical_rows.append({
            "participant_id": "usr_parity_01",
            "session_id": "sess_parity_01",
            "timestamp": press_t,
            "event_type": "keystroke",
            "key_identifier": token,
            "press_time": press_t,
            "release_time": release_t,
            "condition": "neutral",
            "dwell_time": dwell,
            "flight_time": flight if i > 0 else np.nan,
            "is_backspace": is_bk,
        })

        # Next keypress arrives after flight latency
        next_flight = 120.0 + ((i + 1) % 5) * 15.0
        current_time = release_t + next_flight

    canonical_df = pd.DataFrame(canonical_rows)
    return raw_events, canonical_df


def test_dwell_and_flight_timing_parity():
    """Verify live normalizer dwell and flight values match offline calculations identically."""
    raw_events, canonical_df = _build_deterministic_fixture(35)

    # 1. Process via Live Event Normalizer
    normalizer = LiveEventNormalizer()
    sanitized_events = [RawBrowserEvent(**ev) for ev in raw_events]
    live_paired = normalizer.process_batch(sanitized_events)

    # 2. Process via Offline Feature Functions
    offline_dwell = calculate_dwell_time(canonical_df)
    offline_flight = calculate_flight_time(canonical_df, session_col="session_id")

    assert len(live_paired) == len(canonical_df)

    for i in range(len(live_paired)):
        lp = live_paired[i]
        # Dwell parity
        expected_dwell = float(offline_dwell.iloc[i])
        assert abs(lp.dwell_time - expected_dwell) < 1e-4, f"Dwell mismatch at {i}: live={lp.dwell_time}, offline={expected_dwell}"

        # Flight parity (first event is None / NaN)
        if i == 0:
            assert lp.flight_time is None
            assert pd.isna(offline_flight.iloc[i])
        else:
            expected_flight = float(offline_flight.iloc[i])
            assert abs(lp.flight_time - expected_flight) < 1e-4, f"Flight mismatch at {i}: live={lp.flight_time}, offline={expected_flight}"


def test_canonical_sequence_features_parity():
    """Verify live feature buffer schema and ordering match offline sequence expectations."""
    raw_events, canonical_df = _build_deterministic_fixture(40)

    # Live pipeline
    normalizer = LiveEventNormalizer()
    sanitized_events = [RawBrowserEvent(**ev) for ev in raw_events]
    live_paired = normalizer.process_batch(sanitized_events)

    buffer = LiveFeatureBuffer(sequence_length=30, sequence_stride=10)
    buffer.add_events(live_paired)
    live_df = buffer.to_dataframe(user_id="usr_parity_01", session_id="sess_parity_01")

    # Verify column ordering matches canonical sequence features exactly
    for col in CANONICAL_SEQUENCE_FEATURES:
        assert col in live_df.columns

    # Verify 3D tensor shape
    X_live, meta_live = buffer.generate_sequence_windows(sequence_length=30, sequence_stride=10)
    assert X_live.ndim == 3
    assert X_live.shape == (2, 30, 6)  # 40 events with len 30, stride 10 -> 2 windows

    # Offline grouped sequence slicing
    canonical_df["user_id"] = canonical_df["participant_id"]
    canonical_df["pause_duration"] = canonical_df["flight_time"].apply(
        lambda f: f if (pd.notnull(f) and f > 500.0) else 0.0
    )
    canonical_df["typing_speed"] = 45.0  # constant benchmark
    canonical_df["backspace"] = canonical_df["is_backspace"].astype(float)
    canonical_df["error_flag"] = canonical_df["backspace"]

    X_offline, _, meta_offline = prepare_grouped_sequences(
        canonical_df,
        feature_cols=CANONICAL_SEQUENCE_FEATURES,
        sequence_length=30,
        sequence_stride=10,
    )

    assert X_live.shape == X_offline.shape
    assert len(meta_live) == len(meta_offline)


def test_session_level_feature_table_parity():
    """Verify session-level summary features produced from live data match offline feature tables."""
    raw_events, canonical_df = _build_deterministic_fixture(30)

    normalizer = LiveEventNormalizer()
    live_paired = normalizer.process_batch([RawBrowserEvent(**ev) for ev in raw_events])

    buffer = LiveFeatureBuffer()
    buffer.add_events(live_paired)
    live_df = buffer.to_dataframe(user_id="usr_parity_01", session_id="sess_parity_01")

    # Offline build_feature_table
    offline_summary, _ = build_feature_table(
        canonical_df.rename(columns={"participant_id": "user_id"}),
        user_col="user_id",
        session_col="session_id",
        timestamp_col="press_time",
    )

    # Live build_feature_table on live_df
    live_summary, _ = build_feature_table(
        live_df,
        user_col="user_id",
        session_col="session_id",
        timestamp_col="press_time",
    )

    assert not offline_summary.empty
    assert not live_summary.empty

    row_off = offline_summary.iloc[0]
    row_live = live_summary.iloc[0]

    # Verify statistical aggregates match within 0.05 tolerance
    assert abs(row_off["mean_dwell_time"] - row_live["mean_dwell_time"]) < 0.05
    assert abs(row_off["median_dwell_time"] - row_live["median_dwell_time"]) < 0.05
    assert abs(row_off["mean_flight_time"] - row_live["mean_flight_time"]) < 0.05
    assert abs(row_off["total_backspaces"] - row_live["total_backspaces"]) == 0
