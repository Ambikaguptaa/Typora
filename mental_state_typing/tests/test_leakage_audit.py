"""Unit tests for pre-training data leakage audit module."""

import pytest

from src.data_engineering.leakage_audit import (
    assert_no_data_leakage,
    audit_data_leakage,
)


def test_leakage_audit_clean_pass():
    """Verify clean disjoint participant and session sets pass the audit."""
    train_users = ["usr_01", "usr_02", "usr_03"]
    val_users = ["usr_04"]
    test_users = ["usr_05"]

    train_sess = ["s1", "s2", "s3"]
    val_sess = ["s4"]
    test_sess = ["s5"]

    features = ["dwell_time", "flight_time", "typing_speed"]

    res = audit_data_leakage(
        train_groups=train_users,
        val_groups=val_users,
        test_groups=test_users,
        train_sessions=train_sess,
        val_sessions=val_sess,
        test_sessions=test_sess,
        feature_names=features,
        target_name="condition",
    )

    assert res["passed"] is True
    assert len(res["user_overlap"]) == 0
    assert len(res["session_overlap"]) == 0
    assert res["target_leakage_detected"] is False


def test_leakage_audit_user_overlap_train_val():
    """Verify participant overlap between train and validation partitions is detected."""
    train_users = ["usr_01", "usr_02", "usr_03"]
    val_users = ["usr_02", "usr_04"]  # usr_02 leaks into validation

    res = audit_data_leakage(
        train_groups=train_users,
        val_groups=val_users,
    )

    assert res["passed"] is False
    assert "usr_02" in res["user_overlap"]
    assert any("Participant ID leakage" in r for r in res["reasons"])


def test_leakage_audit_user_overlap_train_test():
    """Verify participant overlap between train and test partitions is detected."""
    train_users = ["usr_01", "usr_02"]
    val_users = ["usr_03"]
    test_users = ["usr_01"]  # usr_01 leaks into test

    res = audit_data_leakage(
        train_groups=train_users,
        val_groups=val_users,
        test_groups=test_users,
    )

    assert res["passed"] is False
    assert "usr_01" in res["user_overlap"]


def test_leakage_audit_session_overlap():
    """Verify session ID overlap across partitions is detected."""
    res = audit_data_leakage(
        train_groups=["usr_01"],
        val_groups=["usr_02"],
        train_sessions=["sess_common_01"],
        val_sessions=["sess_common_01"],
    )

    assert res["passed"] is False
    assert "sess_common_01" in res["session_overlap"]


def test_leakage_audit_target_feature_leakage():
    """Verify target condition or label name accidentally included in X feature tensor is detected."""
    res = audit_data_leakage(
        train_groups=["usr_01"],
        val_groups=["usr_02"],
        feature_names=["dwell_time", "flight_time", "condition"],  # condition in X
        target_name="condition",
    )

    assert res["passed"] is False
    assert res["target_leakage_detected"] is True
    assert "condition" in res["leaked_features"]


def test_assert_no_data_leakage():
    """Verify assert_no_data_leakage raises ValueError when leakage occurs."""
    # Should pass without exception
    assert_no_data_leakage(
        train_groups=["usr_01"],
        val_groups=["usr_02"],
        feature_names=["dwell_time"],
    )

    # Should raise ValueError
    with pytest.raises(ValueError, match="DATA LEAKAGE DETECTED"):
        assert_no_data_leakage(
            train_groups=["usr_01"],
            val_groups=["usr_01"],
            feature_names=["dwell_time"],
        )
