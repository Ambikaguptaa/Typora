"""Unit tests verifying zero sliding-window leakage across sessions and partitions."""

import numpy as np
import pandas as pd
import pytest

from src.deep_learning.data_split import split_sequences_by_group
from src.deep_learning.preprocessing import prepare_grouped_sequences


def test_window_generation_never_crosses_session_boundaries():
    """Verify that sliding windows are generated strictly within individual sessions."""
    # Create dataset with 2 sessions of 35 events each
    # (sequence_length=30, stride=5 -> 2 windows per session)
    records = []
    for s_idx in [1, 2]:
        for e_idx in range(35):
            records.append({
                "user_id": "usr_01",
                "session_id": f"sess_{s_idx}",
                "timestamp": float(s_idx * 10000 + e_idx * 100),
                "dwell_time": 100.0 + float(s_idx * 10),
                "flight_time": 150.0,
                "typing_speed": 45.0,
                "state": "neutral" if s_idx == 1 else "stressed",
            })

    df = pd.DataFrame(records)
    feature_cols = ["dwell_time", "flight_time", "typing_speed"]

    X, y, meta = prepare_grouped_sequences(
        df=df,
        feature_cols=feature_cols,
        group_cols=["user_id", "session_id"],
        sequence_length=30,
        sequence_stride=5,
        label_col="state",
    )

    # 35 events with window=30, stride=5 gives (35 - 30)//5 + 1 = 2 windows per session -> 4 windows total
    assert len(X) == 4
    assert len(meta) == 4

    # First 2 windows must belong strictly to sess_1
    assert meta["session_id"].iloc[0] == "sess_1"
    assert meta["session_id"].iloc[1] == "sess_1"

    # Next 2 windows must belong strictly to sess_2
    assert meta["session_id"].iloc[2] == "sess_2"
    assert meta["session_id"].iloc[3] == "sess_2"

    # Verify that values inside X for sess_1 windows contain ONLY sess_1 dwell times (110.0)
    assert np.allclose(X[0, :, 0], 110.0)
    assert np.allclose(X[1, :, 0], 110.0)

    # Verify that values inside X for sess_2 windows contain ONLY sess_2 dwell times (120.0)
    assert np.allclose(X[2, :, 0], 120.0)
    assert np.allclose(X[3, :, 0], 120.0)


def test_split_sequences_by_group_guarantees_disjoint_partitions():
    """Verify that grouped splitting never places sequences from the same participant in multiple partitions."""
    # 6 users, 10 sequences per user (60 sequences total)
    n_sequences = 60
    X = np.random.randn(n_sequences, 20, 4).astype(np.float32)
    y = np.array(["neutral" if i % 2 == 0 else "stressed" for i in range(n_sequences)])
    meta = pd.DataFrame({
        "user_id": [f"user_{i // 10:02d}" for i in range(n_sequences)],
        "session_id": [f"sess_{i // 5:02d}" for i in range(n_sequences)],
    })

    (
        (X_train, y_train, meta_train),
        (X_val, y_val, meta_val),
        (X_test, y_test, meta_test),
        summary,
    ) = split_sequences_by_group(
        X=X,
        y=y,
        metadata_df=meta,
        group_col="user_id",
        val_size=0.17,
        test_size=0.17,
        random_state=42,
    )

    train_users = set(meta_train["user_id"].unique())
    val_users = set(meta_val["user_id"].unique())
    test_users = set(meta_test["user_id"].unique())

    # Verify mutual exclusivity
    assert len(train_users.intersection(val_users)) == 0, "Train and Val users must not overlap"
    assert len(train_users.intersection(test_users)) == 0, "Train and Test users must not overlap"
    assert len(val_users.intersection(test_users)) == 0, "Val and Test users must not overlap"

    # All sequences must be accounted for
    assert len(X_train) + len(X_val) + len(X_test) == n_sequences
    assert summary["zero_leakage_guaranteed"] is True
