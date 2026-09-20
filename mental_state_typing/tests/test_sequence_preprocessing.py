"""Unit tests for Temporal Sequence Preparation."""

import numpy as np
import pandas as pd
import pytest

from src.deep_learning.preprocessing import (
    prepare_grouped_sequences,
    prepare_sequences,
)


def test_prepare_sequences_legacy():
    """Verify legacy 2D to 3D array sliding window function."""
    matrix = np.arange(100).reshape(20, 5)
    seqs = prepare_sequences(matrix, sequence_length=10, step_size=5)

    assert seqs.shape == (3, 10, 5)


def test_grouped_sequences_preserves_temporal_order():
    """Verify sequences maintain chronological timestamp order."""
    # Create 35 chronological records for a single session
    records = []
    for i in range(35):
        records.append(
            {
                "user_id": "u1",
                "session_id": "s1",
                "timestamp": 1000 + i * 100,
                "dwell": float(80 + i),
                "flight": float(50 + i),
                "state": "Calm",
            }
        )
    df = pd.DataFrame(records)

    X, y, meta = prepare_grouped_sequences(
        df,
        feature_cols=["dwell", "flight"],
        group_cols=["user_id", "session_id"],
        timestamp_col="timestamp",
        label_col="state",
        sequence_length=10,
        sequence_stride=5,
    )

    # Number of windows: (35 - 10) / 5 + 1 = 6
    assert X.shape == (6, 10, 2)
    assert len(y) == 6
    assert all(label == "Calm" for label in y)

    # Verify first sequence first timestep corresponds to timestamp 1000
    assert meta.iloc[0]["start_timestamp"] == 1000
    assert meta.iloc[0]["end_timestamp"] == 1900


def test_grouped_sequences_never_mixes_users_or_sessions():
    """CRITICAL TEST: Verify sequences never span across different users or sessions."""
    # User A has 15 records, User B has 15 records
    user_a_records = [
        {"user_id": "user_A", "session_id": "sess_1", "timestamp": 1000 + i, "feat1": 1.0}
        for i in range(15)
    ]
    user_b_records = [
        {"user_id": "user_B", "session_id": "sess_2", "timestamp": 2000 + i, "feat1": 2.0}
        for i in range(15)
    ]
    df = pd.DataFrame(user_a_records + user_b_records)

    # Request sequence length of 10
    X, y, meta = prepare_grouped_sequences(
        df,
        feature_cols=["feat1"],
        group_cols=["user_id", "session_id"],
        sequence_length=10,
        sequence_stride=5,
    )

    # For user A: (15 - 10) / 5 + 1 = 2 sequences
    # For user B: (15 - 10) / 5 + 1 = 2 sequences
    # Total = 4 sequences
    assert X.shape == (4, 10, 1)
    assert len(meta) == 4

    # Check each window's metadata to confirm pure user attribution
    assert meta.iloc[0]["user_id"] == "user_A"
    assert meta.iloc[1]["user_id"] == "user_A"
    assert meta.iloc[2]["user_id"] == "user_B"
    assert meta.iloc[3]["user_id"] == "user_B"

    # Verify user A sequences only contain feat1 == 1.0
    assert np.all(X[0, :, 0] == 1.0)
    assert np.all(X[1, :, 0] == 1.0)

    # Verify user B sequences only contain feat1 == 2.0
    assert np.all(X[2, :, 0] == 2.0)
    assert np.all(X[3, :, 0] == 2.0)


def test_grouped_sequences_skips_too_short_sessions():
    """Verify sessions with fewer observations than sequence_length are skipped without errors."""
    short_session = [
        {"user_id": "u1", "session_id": "s_short", "timestamp": 1000 + i, "val": 1.0}
        for i in range(5)  # only 5 events
    ]
    df = pd.DataFrame(short_session)

    X, y, meta = prepare_grouped_sequences(
        df,
        feature_cols=["val"],
        group_cols=["user_id", "session_id"],
        sequence_length=20,  # requires 20 events
    )

    assert X.shape == (0, 20, 1)
    assert len(meta) == 0
