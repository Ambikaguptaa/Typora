"""Unit tests verifying zero scaler leakage across partitions."""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.data_engineering.leakage_audit import audit_data_leakage
from src.deep_learning.data_split import fit_sequence_scaler, transform_sequence


def test_scaler_fit_strictly_on_training_data():
    """Verify that scaler parameters are derived solely from training sequences."""
    np.random.seed(42)
    # Training data drawn from N(10, 2)
    X_train = np.random.normal(loc=10.0, scale=2.0, size=(20, 10, 3)).astype(np.float32)
    # Test data drawn from distinct distribution N(50, 5)
    X_test = np.random.normal(loc=50.0, scale=5.0, size=(10, 10, 3)).astype(np.float32)

    scaler, X_train_scaled = fit_sequence_scaler(X_train, scaler_type="standard")

    # Scaler mean should match training mean (~10.0), NOT test mean (~50.0)
    for mean_val in scaler.mean_:
        assert 9.0 < mean_val < 11.0, f"Scaler mean {mean_val} should be close to training distribution 10.0"

    # Transform test sequences without refitting
    X_test_scaled = transform_sequence(scaler, X_test)

    # Scaler mean must NOT have changed after transforming test data
    for mean_val in scaler.mean_:
        assert 9.0 < mean_val < 11.0, "Scaler mean should remain unchanged after transforming test data"


def test_scaler_leakage_audit_flag():
    """Verify that audit_data_leakage detects and rejects scaler fitting on test/val."""
    # When scaler is fitted on train only: PASS
    pass_report = audit_data_leakage(
        train_groups=["u1", "u2"],
        val_groups=["u3"],
        test_groups=["u4"],
        scaler_fitted_on_train_only=True,
        save_reports=False,
    )
    assert pass_report["passed"] is True
    assert pass_report["scaler_leakage"]["status"] == "PASS"

    # When scaler is fitted outside training partition: FAIL
    fail_report = audit_data_leakage(
        train_groups=["u1", "u2"],
        val_groups=["u3"],
        test_groups=["u4"],
        scaler_fitted_on_train_only=False,
        save_reports=False,
    )
    assert fail_report["passed"] is False
    assert fail_report["scaler_leakage"]["status"] == "FAIL"
    assert any("Scaler leakage" in reason for reason in fail_report["reasons"])


def test_test_partition_refit_causes_distributional_leakage():
    """Demonstrate why fitting on test data alters normalization and creates leakage."""
    np.random.seed(42)
    X_train = np.random.normal(loc=5.0, scale=1.0, size=(30, 5, 2)).astype(np.float32)
    X_test = np.random.normal(loc=25.0, scale=1.0, size=(15, 5, 2)).astype(np.float32)

    # Legitimate approach: fit train, transform test
    scaler_legit, _ = fit_sequence_scaler(X_train)
    scaled_test_legit = transform_sequence(scaler_legit, X_test)

    # Leaked approach: refit on test
    scaler_leaked, scaled_test_leaked = fit_sequence_scaler(X_test)

    # Scaled test values must be dramatically different
    # Under legitimate scaling, test values (mean ~25) scaled by train (mean ~5) will have values ~20
    # Under leaked scaling, test values are centered to ~0
    assert not np.allclose(scaled_test_legit, scaled_test_leaked)
    assert np.mean(scaled_test_leaked) < 0.1  # Leaked: centers around 0
    assert np.mean(scaled_test_legit) > 15.0  # Legitimate: preserves true distribution shift
