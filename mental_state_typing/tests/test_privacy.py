"""Unit and integration tests for Data Security & Privacy Subsystem.

Covers data classification, minimization, HMAC-SHA256 pseudonymization, Fernet encryption,
storage policies, access control, Laplace differential privacy, retention, and audit logging.
"""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import pytest

from src.privacy import (
    AccessController,
    DataSensitivityLevel,
    DecryptionError,
    FORBIDDEN_TEXT_COLUMNS,
    Permission,
    PermissionDeniedError,
    Role,
    SecurityEventType,
    StoragePolicy,
    assert_zero_raw_text,
    audit_zero_raw_text,
    calculate_expiration,
    classify_field,
    classify_schema,
    cleanup_expired_artifacts,
    compute_privacy_budget,
    decrypt_bytes,
    decrypt_file,
    decrypt_json,
    encrypt_bytes,
    encrypt_file,
    encrypt_json,
    enforce_data_minimization,
    generate_encryption_key,
    get_storage_policy,
    identify_expired_artifacts,
    is_expired,
    is_safe_metadata_only,
    is_valid_pseudonym,
    log_security_event,
    minimize_keystroke_dataframe,
    private_count,
    private_mean,
    pseudonymize_series,
    pseudonymize_user_id,
    purge_participant_data,
    read_recent_audit_events,
    strip_character_data,
    validate_artifact_storage_compliance,
)


def test_privacy_modules_importable():
    """Verify that all privacy modules and exports are accessible."""
    import src.privacy as p
    assert p is not None
    assert hasattr(p, "pseudonymize_user_id")
    assert hasattr(p, "encrypt_bytes")
    assert hasattr(p, "private_mean")
    assert hasattr(p, "AccessController")


def test_pseudonymization_deterministic():
    """Verify pseudonymization generates consistent hashes for the same user & secret."""
    user = "student_experiment_001"
    secret = "fixed_test_secret_key"

    hash_1 = pseudonymize_user_id(user, secret=secret)
    hash_2 = pseudonymize_user_id(user, secret=secret)

    assert hash_1 == hash_2
    assert hash_1.startswith("usr_")
    assert user not in hash_1
    assert is_valid_pseudonym(hash_1) is True


def test_pseudonymization_secret_difference():
    """Verify different secrets generate different pseudonyms for the same user."""
    user = "student_experiment_001"

    hash_a = pseudonymize_user_id(user, secret="secret_alpha")
    hash_b = pseudonymize_user_id(user, secret="secret_beta")

    assert hash_a != hash_b
    assert is_valid_pseudonym(hash_a)
    assert is_valid_pseudonym(hash_b)


def test_pseudonymization_invalid_inputs():
    """Verify that empty inputs raise ValueError."""
    with pytest.raises(ValueError, match="empty or blank"):
        pseudonymize_user_id("")

    with pytest.raises(ValueError, match="empty or blank"):
        pseudonymize_user_id("   ")

    with pytest.raises(ValueError, match="cannot be empty"):
        pseudonymize_user_id("valid_user", secret="")


def test_pseudonymize_series():
    """Verify pandas Series pseudonymization."""
    series = pd.Series(["participant_1", "participant_2", "participant_1"])
    result = pseudonymize_series(series, secret="test_secret")

    assert len(result) == 3
    assert result.iloc[0] == result.iloc[2]
    assert result.iloc[0] != result.iloc[1]
    assert all(is_valid_pseudonym(x) for x in result)


def test_data_classification():
    """Verify sensitivity classification logic across fields."""
    assert classify_field("raw_user_id") == DataSensitivityLevel.HIGHLY_SENSITIVE
    assert classify_field("password") == DataSensitivityLevel.HIGHLY_SENSITIVE
    assert classify_field("key") == DataSensitivityLevel.HIGHLY_SENSITIVE
    assert classify_field("text") == DataSensitivityLevel.HIGHLY_SENSITIVE
    assert classify_field("dwell_time") == DataSensitivityLevel.SENSITIVE
    assert classify_field("flight_time") == DataSensitivityLevel.SENSITIVE
    assert classify_field("typing_speed") == DataSensitivityLevel.SENSITIVE
    assert classify_field("session_id") == DataSensitivityLevel.INTERNAL
    assert classify_field("record_count") == DataSensitivityLevel.INTERNAL

    schema = classify_schema(["user_id", "dwell_time", "session_id"])
    assert schema["user_id"] == DataSensitivityLevel.HIGHLY_SENSITIVE
    assert schema["dwell_time"] == DataSensitivityLevel.SENSITIVE
    assert schema["session_id"] == DataSensitivityLevel.INTERNAL


def test_data_minimization_drops_forbidden_text():
    """Verify that forbidden text/character columns are dropped and reported."""
    unsafe_df = pd.DataFrame(
        {
            "press_time": [100.0, 200.0],
            "release_time": [180.0, 280.0],
            "key": ["a", "b"],
            "text": ["hello", "world"],
            "character": ["a", "b"],
            "password": ["p1", "p2"],
        }
    )

    clean_df, report = minimize_keystroke_dataframe(unsafe_df)

    assert "press_time" in clean_df.columns
    assert "release_time" in clean_df.columns
    assert "key" not in clean_df.columns
    assert "text" not in clean_df.columns
    assert "password" not in clean_df.columns
    assert report["zero_text_compliant"] is True
    assert set(report["dropped_columns"]) == {"key", "text", "character", "password"}


def test_data_minimization_pseudonymizes_user_column():
    """Verify that participant ID columns are converted to pseudonyms."""
    raw_df = pd.DataFrame(
        {
            "user_id": ["alice@uni.edu", "bob@uni.edu"],
            "dwell_time": [120.0, 140.0],
        }
    )

    clean_df, report = minimize_keystroke_dataframe(raw_df, secret="uni_secret")

    assert "user_id" in report["pseudonymized_columns"]
    for val in clean_df["user_id"]:
        assert is_valid_pseudonym(val)
        assert "@uni.edu" not in val

    # Verify enforce_data_minimization convenience wrapper
    enforced = enforce_data_minimization(raw_df)
    assert len(enforced) == 2


def test_encryption_roundtrip_bytes():
    """Verify Fernet authenticated encryption roundtrip for raw bytes."""
    key = generate_encryption_key()
    data = b"Microsecond timing event stream: [120, 150, 90, 210]"

    ciphertext = encrypt_bytes(data, key=key)
    assert ciphertext != data
    assert len(ciphertext) > len(data)

    decrypted = decrypt_bytes(ciphertext, key=key)
    assert decrypted == data


def test_encryption_roundtrip_json():
    """Verify Fernet authenticated encryption roundtrip for Python dictionaries."""
    key = generate_encryption_key()
    payload = {
        "user_pseudonym": "usr_1234567890abcdef",
        "baseline_dwell_mean": 115.4,
        "baseline_flight_mean": 182.1,
        "session_count": 8,
    }

    ciphertext = encrypt_json(payload, key=key)
    assert isinstance(ciphertext, bytes)

    restored = decrypt_json(ciphertext, key=key)
    assert restored == payload


def test_encryption_file_roundtrip(tmp_path):
    """Verify encrypt_file and decrypt_file on local filesystem."""
    key = generate_encryption_key()
    src_file = tmp_path / "baseline_report.json"
    enc_file = tmp_path / "baseline_report.enc"
    dec_file = tmp_path / "baseline_report.restored.json"

    original_content = json.dumps({"status": "assessment_ready", "score": 42.5})
    src_file.write_text(original_content, encoding="utf-8")

    encrypt_file(src_file, enc_file, key=key)
    assert enc_file.exists()
    assert enc_file.read_bytes() != original_content.encode("utf-8")

    decrypt_file(enc_file, dec_file, key=key)
    assert dec_file.exists()
    assert dec_file.read_text(encoding="utf-8") == original_content


def test_encryption_tamper_detection():
    """Verify that tampering with ciphertext triggers DecryptionError."""
    key = generate_encryption_key()
    data = b"Sensitive behavioral payload"
    ciphertext = bytearray(encrypt_bytes(data, key=key))

    # Tamper with the last byte
    ciphertext[-1] ^= 0xFF

    with pytest.raises(DecryptionError, match="invalid encryption key or corrupted/tampered"):
        decrypt_bytes(bytes(ciphertext), key=key)


def test_encryption_wrong_key_fails():
    """Verify that decrypting with an incorrect key raises DecryptionError."""
    key_a = generate_encryption_key()
    key_b = generate_encryption_key()

    data = b"Confidential timing sequence"
    ciphertext = encrypt_bytes(data, key=key_a)

    with pytest.raises(DecryptionError):
        decrypt_bytes(ciphertext, key=key_b)


def test_storage_policy_compliance():
    """Verify storage policy retrieval and compliance checks."""
    policy = get_storage_policy("personal_baselines")
    assert policy.encryption_required is True
    assert policy.pseudonymization_required is True
    assert policy.allow_raw_text is False

    # Fully compliant
    valid, violations = validate_artifact_storage_compliance(
        artifact_type="personal_baselines",
        has_raw_text=False,
        is_pseudonymized=True,
        is_encrypted=True,
    )
    assert valid is True
    assert len(violations) == 0

    # Non-compliant: contains raw text and unencrypted
    valid_bad, violations_bad = validate_artifact_storage_compliance(
        artifact_type="personal_baselines",
        has_raw_text=True,
        is_pseudonymized=False,
        is_encrypted=False,
    )
    assert valid_bad is False
    assert len(violations_bad) == 3


def test_access_control_rbac():
    """Verify role-based access control rules."""
    # User can view own data but not train models
    assert AccessController.has_permission(Role.USER, Permission.VIEW_OWN_TYPING_DEVIATION, is_own_data=True) is True
    assert AccessController.has_permission(Role.USER, Permission.VIEW_OWN_TYPING_DEVIATION, is_own_data=False) is False
    assert AccessController.has_permission(Role.USER, Permission.TRAIN_RESEARCH_MODELS) is False

    # Researcher can train models and view aggregate metrics
    assert AccessController.has_permission(Role.RESEARCHER, Permission.TRAIN_RESEARCH_MODELS) is True
    assert AccessController.has_permission(Role.RESEARCHER, Permission.VIEW_AGGREGATE_METRICS) is True
    assert AccessController.has_permission(Role.RESEARCHER, Permission.MANAGE_RETENTION_POLICIES) is False

    # Admin can manage retention and view audit logs
    assert AccessController.has_permission(Role.ADMIN, Permission.MANAGE_RETENTION_POLICIES) is True
    assert AccessController.has_permission(Role.ADMIN, Permission.VIEW_SECURITY_AUDIT_LOGS) is True

    # Zero-Raw-Text invariant: prohibited across all roles
    for role in [Role.USER, Role.RESEARCHER, Role.ADMIN]:
        assert AccessController.has_permission(role, Permission.VIEW_RAW_KEYSTROKE_TEXT) is False

    # Enforcement raises PermissionDeniedError
    with pytest.raises(PermissionDeniedError):
        AccessController.enforce_permission(Role.USER, Permission.TRAIN_RESEARCH_MODELS)


def test_zero_raw_text_audit():
    """Verify audit_zero_raw_text and assert_zero_raw_text."""
    safe_df = pd.DataFrame({"press_time": [10.0], "release_time": [20.0]})
    unsafe_df = pd.DataFrame({"press_time": [10.0], "key": ["a"]})

    audit_safe = audit_zero_raw_text(safe_df)
    assert audit_safe["compliant"] is True
    assert len(audit_safe["violations"]) == 0

    audit_unsafe = audit_zero_raw_text(unsafe_df)
    assert audit_unsafe["compliant"] is False
    assert len(audit_unsafe["violations"]) == 1

    # Dictionary audit
    safe_dict = {"dwell_mean": 120, "flight_mean": 150}
    unsafe_dict = {"dwell_mean": 120, "text": "hello"}
    assert audit_zero_raw_text(safe_dict)["compliant"] is True
    assert audit_zero_raw_text(unsafe_dict)["compliant"] is False

    # Assert raises ValueError
    assert_zero_raw_text(safe_df)  # Should not raise
    with pytest.raises(ValueError, match="Zero-Text Policy violation"):
        assert_zero_raw_text(unsafe_df)


def test_differential_privacy_laplace_mean():
    """Verify Laplace differential privacy mechanism on sample feature mean."""
    speeds = [45.0, 52.0, 48.0, 60.0, 55.0, 49.0, 53.0, 51.0]
    bounds = (30.0, 80.0)

    res = private_mean(speeds, bounds=bounds, epsilon=1.0, seed=42)

    assert res["is_differentially_private"] is True
    assert res["mechanism"] == "Laplace"
    assert res["epsilon"] == 1.0
    assert res["sample_size"] == 8
    assert isinstance(res["private_value"], float)
    assert isinstance(res["noise_added"], float)

    # Deterministic seed test
    res_repeat = private_mean(speeds, bounds=bounds, epsilon=1.0, seed=42)
    assert res["private_value"] == res_repeat["private_value"]

    # Invalid epsilon raises ValueError
    with pytest.raises(ValueError, match="must be positive"):
        private_mean(speeds, bounds=bounds, epsilon=0.0)


def test_differential_privacy_count():
    """Verify Laplace differential privacy mechanism for record count."""
    items = ["session_1", "session_2", "session_3", "session_4"]
    res = private_count(items, epsilon=1.0, seed=123)

    assert res["is_differentially_private"] is True
    assert res["true_value"] == 4
    assert res["private_value"] >= 0.0


def test_differential_privacy_budget_composition():
    """Verify cumulative privacy budget composition."""
    q1 = {"epsilon": 0.5, "query": "mean_dwell"}
    q2 = {"epsilon": 0.5, "query": "mean_flight"}
    q3 = {"epsilon": 1.0, "query": "session_count"}

    budget = compute_privacy_budget([q1, q2, q3])
    assert budget["total_epsilon"] == 2.0
    assert budget["query_count"] == 3


def test_retention_expiration_logic():
    """Verify datetime expiration calculation and check."""
    created = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    exp = calculate_expiration(created, retention_days=7)
    assert exp == datetime(2026, 1, 8, 12, 0, 0, tzinfo=timezone.utc)

    # Check expiration relative to reference date
    before = datetime(2026, 1, 5, 12, 0, 0, tzinfo=timezone.utc)
    after = datetime(2026, 1, 10, 12, 0, 0, tzinfo=timezone.utc)

    assert is_expired(created, retention_days=7, current_time=before) is False
    assert is_expired(created, retention_days=7, current_time=after) is True


def test_retention_cleanup_dry_run_safety(tmp_path):
    """Verify retention cleanup identifies expired files and enforces dry_run."""
    import os
    test_dir = tmp_path / "retention_test"
    test_dir.mkdir()

    file1 = test_dir / "recent.json"
    file1.write_text("{}", encoding="utf-8")
    # Set mtime to 10 days ago
    past_ts = (datetime.now(timezone.utc) - timedelta(days=10)).timestamp()
    os.utime(file1, (past_ts, past_ts))

    # Dry run should identify files without deleting
    report_dry = cleanup_expired_artifacts(test_dir, retention_days=7, dry_run=True)
    assert report_dry["dry_run"] is True
    assert report_dry["expired_count"] == 1
    assert report_dry["deleted_count"] == 0
    assert file1.exists()

    # Actual delete execution
    report_del = cleanup_expired_artifacts(test_dir, retention_days=7, dry_run=False)
    assert report_del["dry_run"] is False
    assert report_del["deleted_count"] == 1
    assert not file1.exists()


def test_retention_purge_participant_data(tmp_path):
    """Verify GDPR participant data purge on a test database."""
    test_db = tmp_path / "test_research.db"
    conn = sqlite3.connect(str(test_db))
    cur = conn.cursor()
    cur.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY, user_id TEXT, duration REAL);")
    cur.execute("INSERT INTO sessions (user_id, duration) VALUES ('usr_1111222233334444', 120.0);")
    cur.execute("INSERT INTO sessions (user_id, duration) VALUES ('usr_9999888877776666', 150.0);")
    conn.commit()
    conn.close()

    # Dry run purge
    dry_rep = purge_participant_data("usr_1111222233334444", database_path=test_db, dry_run=True)
    assert dry_rep["dry_run"] is True
    assert dry_rep["records_found"]["sessions.user_id"] == 1

    # Verify record still exists after dry run
    conn = sqlite3.connect(str(test_db))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM sessions;")
    assert cur.fetchone()[0] == 2
    conn.close()

    # Real purge execution
    exec_rep = purge_participant_data("usr_1111222233334444", database_path=test_db, dry_run=False)
    assert exec_rep["dry_run"] is False
    assert exec_rep["records_deleted"]["sessions.user_id"] == 1

    # Verify record is deleted
    conn = sqlite3.connect(str(test_db))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM sessions WHERE user_id = 'usr_1111222233334444';")
    assert cur.fetchone()[0] == 0
    cur.execute("SELECT COUNT(*) FROM sessions;")
    assert cur.fetchone()[0] == 1
    conn.close()


def test_security_audit_logging_and_redaction(tmp_path):
    """Verify audit log records events and sanitizes sensitive text/keys."""
    log_file = tmp_path / "test_audit.jsonl"

    details = {
        "user_pseudonym": "usr_abcd1234efgh5678",
        "action": "export_baseline",
        "key": "super_secret_raw_key",
        "password": "participant_password",
        "text": "forbidden typed keystrokes",
        "status": "success",
    }

    event = log_security_event(
        SecurityEventType.ACCESS_ATTEMPT,
        actor_role="USER",
        details=details,
        log_file=log_file,
    )

    assert log_file.exists()
    assert event["event_type"] == "ACCESS_ATTEMPT"
    assert event["details"]["key"] == "[REDACTED]"
    assert event["details"]["password"] == "[REDACTED]"
    assert event["details"]["text"] == "[REDACTED]"
    assert event["details"]["status"] == "success"

    recent = read_recent_audit_events(limit=5, log_file=log_file)
    assert len(recent) == 1
    assert recent[0]["actor_role"] == "USER"


def test_full_privacy_pipeline_integration(tmp_path):
    """End-to-end integration test verifying the complete data privacy lifecycle."""
    # 1. Raw incoming telemetry containing mixed columns (including text violation)
    raw_telemetry = pd.DataFrame(
        {
            "user_id": ["student_alpha", "student_alpha", "student_beta"],
            "press_time": [100.0, 250.0, 400.0],
            "release_time": [180.0, 330.0, 490.0],
            "key": ["a", "b", "c"],
            "text": ["hello", "world", "test"],
        }
    )

    # 2. Enforce Zero-Text and minimization
    clean_df, min_report = minimize_keystroke_dataframe(raw_telemetry, secret="pipeline_secret_test")
    assert min_report["zero_text_compliant"] is True
    assert "key" not in clean_df.columns
    assert "text" not in clean_df.columns

    # 3. Assert Zero-Text programmatic compliance
    assert_zero_raw_text(clean_df)

    # 4. Verify participant pseudonymization
    pseudonyms = clean_df["user_id"].tolist()
    assert all(is_valid_pseudonym(p) for p in pseudonyms)
    assert pseudonyms[0] == pseudonyms[1]
    assert pseudonyms[0] != pseudonyms[2]

    # 5. Compute Differentially Private cohort mean dwell time
    dwell_times = clean_df["release_time"] - clean_df["press_time"]
    dp_stat = private_mean(dwell_times, bounds=(50.0, 200.0), epsilon=1.0, seed=99)
    assert dp_stat["is_differentially_private"] is True

    # 6. Encrypt baseline record for persistent storage
    key = generate_encryption_key()
    baseline_record = {
        "user_pseudonym": pseudonyms[0],
        "dp_dwell_mean": dp_stat["private_value"],
        "calculated_at": datetime.now(timezone.utc).isoformat(),
    }
    enc_path = tmp_path / "baseline_usr.enc"
    enc_payload = encrypt_json(baseline_record, key=key)
    enc_path.write_bytes(enc_payload)
    assert enc_path.exists()

    # 7. Validate storage compliance under Storage Policy
    valid, violations = validate_artifact_storage_compliance(
        artifact_type="personal_baselines",
        has_raw_text=False,
        is_pseudonymized=True,
        is_encrypted=True,
    )
    assert valid is True
    assert len(violations) == 0

    # 8. Record audit log
    audit_file = tmp_path / "audit.jsonl"
    log_security_event(
        SecurityEventType.DECRYPTION_SUCCESS,
        actor_role="RESEARCHER",
        details={"artifact": str(enc_path.name), "user": pseudonyms[0]},
        log_file=audit_file,
    )
    events = read_recent_audit_events(log_file=audit_file)
    assert len(events) == 1
