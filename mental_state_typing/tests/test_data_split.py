"""Unit tests for Leakage-Safe Data Splitting, Scaling, and Label Encoding."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.deep_learning.data_split import (
    create_grouped_train_test_split,
    fit_feature_scaler,
    load_scaler,
    save_scaler,
    transform_features,
)
from src.deep_learning.label_encoder import (
    decode_labels,
    encode_labels,
    inspect_labels,
    load_label_mapping,
    save_label_mapping,
)


def test_grouped_train_test_split_zero_leakage():
    """Verify train and test partitions contain mutually exclusive groups."""
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4", "u5", "u5"],
            "feat_a": np.random.randn(10),
            "state": ["Calm", "Calm", "Fatigued", "Fatigued", "High_Workload"] * 2,
        }
    )

    train_df, test_df, summary = create_grouped_train_test_split(
        df, group_col="user_id", test_size=0.4, random_state=42
    )

    train_users = set(train_df["user_id"].unique())
    test_users = set(test_df["user_id"].unique())

    # Assert ZERO overlap
    assert len(train_users.intersection(test_users)) == 0
    assert summary["zero_leakage_guaranteed"] is True
    assert len(train_df) + len(test_df) == len(df)


def test_isolated_feature_scaler():
    """Verify scaler is fit strictly on train split and transforms test split without leak."""
    train_df = pd.DataFrame({"val": [10.0, 20.0, 30.0, 40.0]})
    test_df = pd.DataFrame({"val": [15.0, 25.0]})

    scaler, scaled_train = fit_feature_scaler(train_df, feature_cols=["val"], scaler_type="standard")
    scaled_test = transform_features(scaler, test_df, feature_cols=["val"])

    # Training mean should be 0 and std 1
    assert pytest.approx(float(scaled_train["val"].mean()), abs=1e-5) == 0.0
    assert pytest.approx(float(scaled_train["val"].std(ddof=0)), abs=1e-5) == 1.0

    # Test values scaled using TRAIN statistics (train mean=25.0, std=11.18)
    # (15 - 25) / 11.18 ~= -0.8944
    assert scaled_test["val"].iloc[0] < 0.0

    # Ensure no NaN or infinite values
    assert not scaled_train["val"].isnull().any()
    assert not scaled_test["val"].isnull().any()


def test_scaler_save_and_load(tmp_path):
    """Verify serialization and deserialization of fitted scaler."""
    train_df = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [10.0, 20.0, 30.0]})
    scaler, _ = fit_feature_scaler(train_df, feature_cols=["f1", "f2"])

    scaler_file = tmp_path / "test_scaler.pkl"
    save_scaler(scaler, ["f1", "f2"], scaler_file)

    loaded_scaler, loaded_cols = load_scaler(scaler_file)
    assert loaded_cols == ["f1", "f2"]

    test_point = pd.DataFrame({"f1": [2.0], "f2": [20.0]})
    transformed = transform_features(loaded_scaler, test_point, feature_cols=["f1", "f2"])
    assert pytest.approx(transformed["f1"].iloc[0], abs=1e-4) == 0.0
    assert pytest.approx(transformed["f2"].iloc[0], abs=1e-4) == 0.0


def test_label_encoding_and_serialization(tmp_path):
    """Verify categorical label encoding, decoding, and JSON persistence."""
    labels = pd.Series(["Low_Strain", "High_Strain", "Low_Strain", "Moderate_Strain"])

    inspection = inspect_labels(labels)
    assert inspection["num_classes"] == 3
    assert "Low_Strain" in inspection["unique_labels"]

    encoded, mapping = encode_labels(labels)
    assert len(encoded) == 4
    assert set(mapping.keys()) == {"High_Strain", "Low_Strain", "Moderate_Strain"}

    decoded = decode_labels(encoded, mapping)
    assert decoded == list(labels)

    map_path = tmp_path / "label_map.json"
    save_label_mapping(mapping, map_path)
    loaded_map = load_label_mapping(map_path)
    assert loaded_map == mapping


def test_grouped_train_val_test_split_zero_leakage():
    """Verify 3-way partition (train, val, test) contains zero group overlap."""
    from src.deep_learning.data_split import create_grouped_train_val_test_split

    df = pd.DataFrame({
        "user_id": [f"user_{i:02d}" for i in range(10) for _ in range(4)],
        "val": np.random.randn(40),
    })

    train_df, val_df, test_df, summary = create_grouped_train_val_test_split(
        df, group_col="user_id", val_size=0.2, test_size=0.2, random_state=42
    )

    train_users = set(train_df["user_id"].unique())
    val_users = set(val_df["user_id"].unique())
    test_users = set(test_df["user_id"].unique())

    # Guarantees zero group overlap across all three partitions
    assert len(train_users.intersection(val_users)) == 0
    assert len(train_users.intersection(test_users)) == 0
    assert len(val_users.intersection(test_users)) == 0
    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert summary["zero_leakage_guaranteed"] is True


def test_split_sequences_by_group():
    """Verify splitting 3D sequence array preserves alignment and prevents group bleed."""
    from src.deep_learning.data_split import split_sequences_by_group

    X = np.arange(30 * 5 * 3).reshape(30, 5, 3).astype(np.float32)
    y = np.array([0, 1, 2] * 10)
    meta = pd.DataFrame({
        "user_id": [f"u{i % 6}" for i in range(30)],
        "session_id": [f"s{i % 10}" for i in range(30)],
    })

    (
        (X_train, y_train, meta_train),
        (X_val, y_val, meta_val),
        (X_test, y_test, meta_test),
        summary,
    ) = split_sequences_by_group(X, y, meta, group_col="user_id", val_size=0.2, test_size=0.2)

    train_u = set(meta_train["user_id"])
    val_u = set(meta_val["user_id"])
    test_u = set(meta_test["user_id"])

    assert len(train_u.intersection(val_u)) == 0
    assert len(train_u.intersection(test_u)) == 0
    assert len(val_u.intersection(test_u)) == 0
    assert len(X_train) + len(X_val) + len(X_test) == 30
    assert len(y_train) == len(X_train)
    assert len(meta_train) == len(X_train)


def test_fit_and_transform_sequence_scaler():
    """Verify 3D sequence scaling fits only on training sequences and transforms correctly."""
    from src.deep_learning.data_split import fit_sequence_scaler, transform_sequence

    # X_train with mean 50 and non-zero std
    X_train = np.ones((10, 20, 4), dtype=np.float32) * 50.0
    X_train[0, 0, 0] = 100.0  # add variance

    scaler, X_train_scaled = fit_sequence_scaler(X_train)
    assert X_train_scaled.shape == (10, 20, 4)

    # Test sequence transformed with train scaler
    X_test = np.ones((5, 20, 4), dtype=np.float32) * 50.0
    X_test_scaled = transform_sequence(scaler, X_test)
    assert X_test_scaled.shape == (5, 20, 4)

    # Single 2D sequence transform (timesteps, features)
    single_seq = np.ones((20, 4), dtype=np.float32) * 50.0
    single_scaled = transform_sequence(scaler, single_seq)
    assert single_scaled.shape == (20, 4)

