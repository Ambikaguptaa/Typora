"""Unit tests for training readiness validation (training_validation.py)."""

import numpy as np
import pandas as pd
import pytest

from src.deep_learning.training_validation import validate_training_data


def test_validate_training_data_valid():
    """Verify that a well-formed 3D sequence tensor and labels pass validation."""
    X = np.random.randn(20, 15, 6).astype(np.float32)
    y = np.array(["Calm"] * 10 + ["Fatigued"] * 10)
    meta = pd.DataFrame({
        "user_id": [f"user_{i % 4:02d}" for i in range(20)],
        "session_id": [f"sess_{i % 5:02d}" for i in range(20)],
    })

    report = validate_training_data(
        X=X,
        y=y,
        metadata_df=meta,
        expected_timesteps=15,
        expected_features=6,
    )

    assert report["ready"] is True
    assert report["samples"] == 20
    assert report["timesteps"] == 15
    assert report["features"] == 6
    assert len(report["classes"]) == 2
    assert len(report["blocking_errors"]) == 0
    assert report["users"] == 4


def test_validate_training_data_none_or_non_array_x():
    """Verify that None or non-array X is rejected."""
    report_none = validate_training_data(X=None, y=np.array(["A", "B"]))
    assert report_none["ready"] is False
    assert any("is None" in err for err in report_none["blocking_errors"])

    report_list = validate_training_data(X=[[1, 2], [3, 4]], y=np.array(["A", "B"]))
    assert report_list["ready"] is False
    assert any("numpy.ndarray" in err for err in report_list["blocking_errors"])


def test_validate_training_data_wrong_dimension():
    """Verify that 2D array is rejected because sequences must be 3D."""
    X_2d = np.random.randn(20, 6).astype(np.float32)
    y = np.array(["A"] * 10 + ["B"] * 10)

    report = validate_training_data(X=X_2d, y=y)
    assert report["ready"] is False
    assert any("3-dimensional" in err for err in report["blocking_errors"])


def test_validate_training_data_nan_and_inf():
    """Verify that NaNs and infinite values trigger blocking errors."""
    X = np.random.randn(10, 5, 4).astype(np.float32)
    X[0, 0, 0] = np.nan
    X[1, 1, 1] = np.inf
    y = np.array(["A"] * 5 + ["B"] * 5)

    report = validate_training_data(X=X, y=y)
    assert report["ready"] is False
    assert any("NaN" in err for err in report["blocking_errors"])
    assert any("infinite" in err for err in report["blocking_errors"])


def test_validate_training_data_sample_label_mismatch():
    """Verify that mismatch between X sample count and y label count is caught."""
    X = np.random.randn(15, 10, 4).astype(np.float32)
    y = np.array(["A"] * 10)  # 10 labels for 15 sequences

    report = validate_training_data(X=X, y=y)
    assert report["ready"] is False
    assert any("Sample count mismatch" in err for err in report["blocking_errors"])


def test_validate_training_data_single_class():
    """Verify that target with only one class is blocked."""
    X = np.random.randn(20, 10, 4).astype(np.float32)
    y = np.array(["Calm"] * 20)

    report = validate_training_data(X=X, y=y)
    assert report["ready"] is False
    assert any("At least two distinct classes" in err for err in report["blocking_errors"])


def test_validate_training_data_insufficient_samples_per_class():
    """Verify that class with fewer observations than threshold is blocked."""
    X = np.random.randn(20, 10, 4).astype(np.float32)
    y = np.array(["Calm"] * 19 + ["RareClass"] * 1)

    report = validate_training_data(X=X, y=y, min_samples_per_class=3)
    assert report["ready"] is False
    assert any("RareClass" in err for err in report["blocking_errors"])


def test_validate_training_data_dimensional_consistency():
    """Verify that expected sequence length and feature count mismatches are flagged."""
    X = np.random.randn(10, 20, 5).astype(np.float32)
    y = np.array(["A"] * 5 + ["B"] * 5)

    report = validate_training_data(
        X=X,
        y=y,
        expected_timesteps=30,  # Mismatch: actual is 20
        expected_features=8,   # Mismatch: actual is 5
    )
    assert report["ready"] is False
    assert any("Sequence length mismatch" in err for err in report["blocking_errors"])
    assert any("Feature count mismatch" in err for err in report["blocking_errors"])
