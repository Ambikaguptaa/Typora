"""Unit tests for multi-backend database persistence, schema management, and privacy invariants.

Tests:
1. SQLite initialization and schema creation across all 5 persistent tables.
2. Connection failure handling and graceful error recovery.
3. Database URL parsing and dialect detection (SQLite vs PostgreSQL).
4. Secret exposure prevention (password masking in logs and sanitization).
5. Missing cloud driver / credentials handling and local fallback behavior.
6. Session metadata persistence and CRUD operations.
7. Assessment persistence, retrieval, and full report regeneration.
8. Baseline profile persistence and retrieval.
9. Strict privacy enforcement: rejection of forbidden raw text fields.
10. Assessment deletion (GDPR Article 17 Right-to-be-Forgotten).
"""

from pathlib import Path
import pytest

from database.database import (
    check_connection,
    delete_assessment_record,
    extract_safe_db_diagnostics,
    get_assessment_by_id,
    get_assessment_records,
    get_baseline_profile,
    get_connection,
    get_db_cursor,
    init_db,
    log_database_diagnostics,
    parse_database_url,
    record_audit_log,
    sanitize_database_url_for_logging,
    save_assessment_record,
    save_baseline_profile,
    save_session_record,
)
from src.config.settings import settings
from src.live_typing.privacy_filter import PrivacyViolationError


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Create a clean isolated SQLite database file."""
    db_file = tmp_path / "test_persistence.db"
    init_db(db_file)
    return db_file


def test_sqlite_initialization_and_schema(temp_db: Path):
    """Verify that all 5 tables exist after initialization."""
    with get_db_cursor(temp_db) as cursor:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cursor.fetchall()}

    expected_tables = {"sessions", "typing_metrics", "assessments", "baseline_profiles", "audit_logs"}
    assert expected_tables.issubset(tables)


def test_database_url_parsing_and_dialect_detection():
    """Verify engine detection for sqlite vs postgresql URLs."""
    # SQLite variants
    eng, target = parse_database_url("sqlite:///data/test.db")
    assert eng == "sqlite"
    assert target == "data/test.db"

    eng, target = parse_database_url("sqlite:///:memory:")
    assert eng == "sqlite"
    assert target == ":memory:"

    # PostgreSQL variants
    eng, target = parse_database_url("postgresql://user:secret@localhost:5432/testdb")
    assert eng == "postgresql"
    assert "postgresql://user:secret@localhost:5432/testdb" in target

    eng, target = parse_database_url("postgres://user:secret@localhost:5432/testdb")
    assert eng == "postgresql"


def test_sanitize_database_url_masks_passwords():
    """Verify that credentials are never leaked in logs or error messages."""
    dirty_url = "postgresql://db_user:ultra_secret_pass_123@db.example.com:5432/prod_db"
    clean_url = sanitize_database_url_for_logging(dirty_url)

    assert "ultra_secret_pass_123" not in clean_url
    assert "********" in clean_url
    assert "db_user" in clean_url
    assert "db.example.com" in clean_url


def test_connection_health_check(temp_db: Path):
    """Verify connection health check succeeds on valid DB and fails on invalid path."""
    assert check_connection(temp_db) is True
    # Non-existent invalid file path in a non-existent drive/directory
    assert check_connection(Path("Z:/invalid_dir_9999/invalid.db")) is False


def test_missing_postgres_driver_raises_informative_error():
    """Verify that attempting PostgreSQL connection when driver is missing raises clear error."""
    # Only test if psycopg2 / psycopg are actually not installed
    try:
        import psycopg2  # noqa: F401
        pytest.skip("psycopg2 is installed; skipping missing driver test.")
    except ImportError:
        pass

    with pytest.raises(ImportError) as exc_info:
        get_connection(database_url="postgresql://usr:pwd@localhost:5432/test")
    assert "PostgreSQL driver" in str(exc_info.value)


def test_session_persistence_crud(temp_db: Path):
    """Verify persisting and querying session metadata."""
    save_session_record(
        session_id="sess_db_001",
        user_id="usr_pseudo_abc",
        session_type="ANALYSIS",
        sample_count=42,
        status="completed",
        db_path=temp_db,
    )

    with get_db_cursor(temp_db) as cursor:
        cursor.execute("SELECT * FROM sessions WHERE session_id = ?;", ("sess_db_001",))
        row = dict(cursor.fetchone())
        assert row["session_id"] == "sess_db_001"
        assert row["pseudonymized_user_id"] == "usr_pseudo_abc"
        assert row["session_type"] == "ANALYSIS"
        assert row["sample_count"] == 42


def test_assessment_persistence_and_retrieval(temp_db: Path):
    """Verify complete derived assessment can be saved and reloaded."""
    assessment_payload = {
        "session_id": "sess_persist_test",
        "user_id": "usr_test_subject_1",
        "session_type": "ANALYSIS",
        "timestamp": "2026-09-20T18:30:00",
        "data_quality": {
            "verdict": "PASS",
            "is_valid": True,
            "metrics": {"active_duration_s": 15.2, "paired_events_count": 30},
        },
        "assessment_result": {
            "feature_summary": {
                "mean_dwell_time_ms": 110.5,
                "mean_flight_time_ms": 135.2,
                "mean_pause_duration_ms": 620.0,
                "pause_rate": 0.05,
                "typing_speed_wpm": 48.5,
                "backspace_rate": 0.02,
            }
        },
        "baseline_result": {
            "status": "READY",
            "typing_deviation_index": 22.4,
            "divergence_level": "NORMAL",
        },
        "model_result": {
            "status": "MODEL_NOT_READY",
            "predicted_class": None,
            "confidence": None,
        },
    }

    asmt_id = save_assessment_record(assessment_payload, db_path=temp_db)
    assert asmt_id == "asmt_sess_persist_test"

    # Query list
    records = get_assessment_records(db_path=temp_db)
    assert len(records) == 1
    r = records[0]
    assert r["session_id"] == "sess_persist_test"
    assert r["mean_dwell_ms"] == 110.5
    assert r["tdi_score"] == 22.4

    # Query detail
    detail = get_assessment_by_id("sess_persist_test", db_path=temp_db)
    assert detail is not None
    assert detail["session_id"] == "sess_persist_test"
    assert detail["data_quality"]["metrics"]["paired_events_count"] == 30


def test_assessment_privacy_rejection(temp_db: Path):
    """Verify that any assessment containing forbidden raw text fields is strictly rejected."""
    tainted_payload = {
        "session_id": "sess_tainted",
        "user_id": "usr_test",
        "typed_text": "Sensitive user password",
        "data_quality": {"is_valid": True},
    }

    with pytest.raises(PrivacyViolationError) as exc_info:
        save_assessment_record(tainted_payload, db_path=temp_db)
    assert "Forbidden raw text keys" in str(exc_info.value)


def test_baseline_profile_persistence_and_retrieval(temp_db: Path):
    """Verify personal baseline parameter profile persistence."""
    profile_data = {
        "session_count": 5,
        "is_ready": True,
        "mean_dwell_time_ms": 105.2,
        "std_dwell_time_ms": 14.1,
        "mean_flight_time_ms": 120.8,
        "std_flight_time_ms": 18.5,
        "mean_pause_duration_ms": 550.0,
        "std_pause_duration_ms": 80.0,
        "mean_pause_rate": 0.04,
        "std_pause_rate": 0.01,
        "mean_typing_speed_wpm": 55.0,
        "std_typing_speed_wpm": 4.2,
        "mean_backspace_rate": 0.03,
        "std_backspace_rate": 0.01,
    }

    saved = save_baseline_profile("usr_subject_alpha", profile_data, db_path=temp_db)
    assert saved is True

    loaded = get_baseline_profile("usr_subject_alpha", db_path=temp_db)
    assert loaded is not None
    assert loaded["user_id"] == "usr_subject_alpha"
    assert loaded["session_count"] == 5
    assert loaded["is_ready"] == 1
    assert loaded["mean_dwell_time_ms"] == 105.2


def test_delete_assessment_record(temp_db: Path):
    """Verify assessment record purging for GDPR compliance."""
    save_session_record("sess_to_del", "usr_1", db_path=temp_db)
    save_assessment_record({"session_id": "sess_to_del", "user_id": "usr_1"}, db_path=temp_db)

    assert get_assessment_by_id("sess_to_del", db_path=temp_db) is not None

    deleted = delete_assessment_record("sess_to_del", db_path=temp_db)
    assert deleted is True
    assert get_assessment_by_id("sess_to_del", db_path=temp_db) is None


def test_audit_log_recording(temp_db: Path):
    """Verify operational audit log entries are saved."""
    record_audit_log(
        event_type="DATA_PURGE",
        user_id="usr_anonymous_123",
        details="Purged session sess_999 per user request",
        db_path=temp_db,
    )

    with get_db_cursor(temp_db) as cursor:
        cursor.execute("SELECT * FROM audit_logs WHERE event_type = 'DATA_PURGE';")
        row = dict(cursor.fetchone())
        assert row["event_type"] == "DATA_PURGE"
        assert row["user_id"] == "usr_anonymous_123"
        assert "Purged session" in row["details"]


def test_extract_safe_db_diagnostics_masks_credentials():
    """Verify extract_safe_db_diagnostics extracts only safe metadata without password."""
    url = "postgresql://my_user:ultra_secret_pass_999@db.supabase.co:5432/my_database"
    diag = extract_safe_db_diagnostics(url, "postgresql")

    assert diag["backend_type"] == "postgresql"
    assert diag["host"] == "db.supabase.co"
    assert diag["port"] == 5432
    assert diag["database_name"] == "my_database"
    assert diag["username"] == "my_user"
    assert diag["has_database_url"] is True
    # Ensure password is not present anywhere in the dict values
    assert "ultra_secret_pass_999" not in str(diag)


def test_log_database_diagnostics_format(capsys):
    """Verify log_database_diagnostics outputs the exact 8 required fields and masks passwords."""
    log_database_diagnostics(
        backend_type="postgresql",
        host="aws-0-us-east-1.pooler.supabase.com",
        port=5432,
        database_name="postgres",
        username="postgres.myproject",
        has_database_url=True,
        exc_type="OperationalError",
        exc_message="connection timeout",
    )
    captured = capsys.readouterr()
    stderr = captured.err

    assert "Backend Type: postgresql" in stderr
    assert "Host: aws-0-us-east-1.pooler.supabase.com" in stderr
    assert "Port: 5432" in stderr
    assert "Database Name: postgres" in stderr
    assert "Username: postgres.myproject" in stderr
    assert "DATABASE_URL Exists: True" in stderr
    assert "psycopg2 Exception Type: OperationalError" in stderr
    assert "psycopg2 Exception Message: connection timeout" in stderr


def test_postgres_connection_failure_diagnostic_logging(capsys):
    """Verify that PostgreSQL connection failure logs safe diagnostics and conceals password."""
    unreachable_url = "postgresql://user_diag:super_secret_password_777@127.0.0.1:54329/fail_db"
    
    with pytest.raises(Exception) as exc_info:
        get_connection(database_url=unreachable_url)

    captured = capsys.readouterr()
    stderr = captured.err

    # Verify diagnostic block is printed to stderr
    assert "=== DATABASE CONNECTION DIAGNOSTIC ===" in stderr
    assert "Backend Type: postgresql" in stderr
    assert "Host: 127.0.0.1" in stderr
    assert "Port: 54329" in stderr
    assert "Database Name: fail_db" in stderr
    assert "Username: user_diag" in stderr
    assert "DATABASE_URL Exists: True" in stderr
    assert "psycopg2 Exception Type: OperationalError" in stderr

    # CRITICAL INVARIANT: Password must NEVER appear in output
    assert "super_secret_password_777" not in stderr

