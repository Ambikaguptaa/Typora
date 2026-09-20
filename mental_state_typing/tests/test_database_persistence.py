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
    ConfigurationError,
    DatabaseHealth,
    check_connection,
    check_database_health,
    delete_assessment_record,
    extract_safe_db_diagnostics,
    get_assessment_by_id,
    get_assessment_records,
    get_baseline_profile,
    get_connection,
    get_database_config_diagnostics,
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
from src.config.settings import is_cloud_environment, settings, to_canonical_source
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


def test_missing_postgres_driver_raises_informative_error(monkeypatch):
    """Verify that attempting PostgreSQL connection when driver is missing raises clear error."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name.startswith("pg8000"):
            raise ImportError("No module named 'pg8000'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(ImportError) as exc_info:
        get_connection(database_url="postgresql://usr:pwd@localhost:5432/test")
    assert "pg8000" in str(exc_info.value) or "PostgreSQL driver" in str(exc_info.value)


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
    """Verify log_database_diagnostics outputs the exact required fields and masks passwords."""
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

    assert "DATABASE CONFIG SOURCE:" in stderr
    assert "DB BACKEND: postgresql" in stderr
    assert "DB HOST: aws-0-us-east-1.*.pooler.supabase.com" in stderr
    assert "DB PORT: 5432" in stderr
    assert "DB USER: postgres.mypr****" in stderr
    assert "DB NAME: postgres" in stderr
    assert "POOLER DETECTED: True" in stderr
    assert "DIRECT SUPABASE HOST DETECTED: False" in stderr
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
    
    with pytest.raises(Exception):
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
    assert "DATABASE CONNECTION: FAILED" in stderr
    assert "DRIVER: pg8000" in stderr
    assert ("Exception Type: InterfaceError" in stderr or "Exception Type:" in stderr)

    # CRITICAL INVARIANT: Password must NEVER appear in output
    assert "super_secret_password_777" not in stderr


# ==============================================================================
# 10 MANDATORY REQUIREMENT TESTS (A THROUGH J - STEP 11)
# ==============================================================================

def test_req_a_streamlit_database_url_wins_over_env(monkeypatch):
    """A. Verify Streamlit DATABASE_URL wins over .env and environment variables."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://dotenv_user:dotenv_pass@dotenv-host:5432/dotenv_db")

    import streamlit as st
    monkeypatch.setattr(st, "secrets", {
        "DATABASE_URL": "postgresql://secret_user:secret_pass@aws-0-us-east-1.pooler.supabase.com:5432/secret_db"
    })

    url = settings.get_database_url()
    source = settings.get_database_source()
    canonical_source = settings.get_database_source(canonical=True)

    assert "aws-0-us-east-1.pooler.supabase.com" in url
    assert "dotenv-host" not in url
    assert "secret_user" in url
    assert "STREAMLIT SECRETS" in source
    assert canonical_source == "STREAMLIT_SECRET"


def test_req_b_existing_environment_database_url_wins_over_env(monkeypatch):
    """B. Verify existing process environment DATABASE_URL wins over .env."""
    import streamlit as st
    monkeypatch.setattr(st, "secrets", {})
    monkeypatch.setenv("DATABASE_URL", "postgresql://env_usr:env_pwd@env-host:5432/env_db")

    url = settings.get_database_url()
    backend = settings.get_database_backend()
    canonical_source = settings.get_database_source(canonical=True)

    assert "env-host" in url
    assert backend == "postgresql"
    assert canonical_source == "ENVIRONMENT"


def test_req_c_local_env_works_locally(monkeypatch):
    """C. Verify local .env works locally when no secrets or overriding env are present."""
    import sys
    import streamlit as st
    s_mod = sys.modules["src.config.settings"]

    monkeypatch.setattr(st, "secrets", {})
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("STREAMLIT_COMMUNITY_CLOUD", raising=False)
    monkeypatch.setattr(s_mod, "_INITIAL_OS_ENV_DATABASE_URL", None)
    monkeypatch.setattr(s_mod, "_read_dotenv_database_url", lambda: "postgresql://dot_u:dot_p@dot-host:5432/dot_db")

    url = settings.get_database_url()
    backend = settings.get_database_backend()
    canonical_source = settings.get_database_source(canonical=True)

    assert "dot-host" in url
    assert backend == "postgresql"
    assert canonical_source == "LOCAL_DOTENV"


def test_req_d_sqlite_fallback_works_locally(monkeypatch):
    """D. Verify SQLite fallback works locally when no secrets or env exist."""
    import sys
    import streamlit as st
    s_mod = sys.modules["src.config.settings"]

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("STREAMLIT_COMMUNITY_CLOUD", raising=False)
    monkeypatch.delenv("IS_STREAMLIT_CLOUD", raising=False)
    monkeypatch.delenv("STREAMLIT_SERVER_ENVIRONMENT", raising=False)
    monkeypatch.setattr(st, "secrets", {})
    monkeypatch.setattr(s_mod, "_read_dotenv_database_url", lambda: None)

    url = settings.get_database_url()
    backend = settings.get_database_backend()
    source = settings.get_database_source()
    canonical_source = settings.get_database_source(canonical=True)

    assert backend == "sqlite"
    assert url.startswith("sqlite:///")
    assert "mental_state.db" in url
    assert source == "LOCAL SQLITE FALLBACK"
    assert canonical_source == "SQLITE_FALLBACK"


def test_req_e_cloud_mode_does_not_fall_back_to_sqlite(monkeypatch):
    """E. Verify cloud mode never silently switches to SQLite on PostgreSQL failure."""
    import streamlit as st
    monkeypatch.setattr(st, "secrets", {
        "DATABASE_URL": "postgresql://usr:pass@127.0.0.1:54329/cloud_db"
    })

    # The backend MUST remain postgresql
    assert settings.get_database_backend() == "postgresql"

    # Failed connection check must report offline postgresql, NOT switch to sqlite
    health = check_database_health()
    assert health.is_ready is False
    assert health.backend_type == "postgresql"
    assert settings.get_database_backend() == "postgresql"


def test_req_f_direct_supabase_host_is_never_generated():
    """F. Verify that direct Supabase host db.<ref>.supabase.co is NOT generated from project id."""
    pooler_url = "postgresql://postgres.xhpztsckvjdrnfgmmuuq:pass@aws-0-us-east-1.pooler.supabase.com:5432/postgres?sslmode=require"
    eng, target = parse_database_url(pooler_url)

    assert eng == "postgresql"
    # Verify pooler host is preserved exactly and not converted to db.*.supabase.co
    assert "aws-0-us-east-1.pooler.supabase.com" in target
    assert "db.xhpztsckvjdrnfgmmuuq.supabase.co" not in target


def test_req_g_pooler_url_is_accepted():
    """G. Verify that Supabase Session Pooler URL is accepted and recognized as pooler."""
    pooler_url = "postgresql://postgres.myproject:my_pass@aws-0-eu-central-1.pooler.supabase.com:5432/postgres"
    diag = extract_safe_db_diagnostics(pooler_url, "postgresql")

    assert diag["backend_type"] == "postgresql"
    assert diag["port"] == 5432
    assert diag["is_pooler"] is True
    assert diag["is_direct_supabase"] is False
    assert diag["host"] == "aws-0-eu-central-1.pooler.supabase.com"


def test_req_h_sensitive_connection_values_are_masked_in_diagnostics(capsys):
    """H. Verify sensitive connection values (passwords, tokens) are masked in diagnostics."""
    secret_pass = "top_secret_token_xyz_987654"
    dirty_url = f"postgresql://usr:{secret_pass}@aws-0-us-east-1.pooler.supabase.com:5432/db"

    clean_url = sanitize_database_url_for_logging(dirty_url)
    assert secret_pass not in clean_url
    assert "********" in clean_url

    diag = get_database_config_diagnostics(url=dirty_url)
    assert secret_pass not in str(diag)
    assert "pooler.supabase.com" in diag["hostname"]
    assert "*" in diag["hostname"]

    log_database_diagnostics(
        backend_type="postgresql",
        host="aws-0-us-east-1.pooler.supabase.com",
        port=5432,
        database_name="db",
        username="usr",
        has_database_url=True,
    )
    captured = capsys.readouterr()
    assert secret_pass not in captured.err


def test_req_i_missing_database_url_produces_controlled_configuration_error(monkeypatch):
    """I. Verify missing DATABASE_URL in cloud mode produces controlled configuration error."""
    import sys
    import streamlit as st
    s_mod = sys.modules["src.config.settings"]

    monkeypatch.setenv("STREAMLIT_COMMUNITY_CLOUD", "1")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(st, "secrets", {})
    monkeypatch.setattr(s_mod, "_read_dotenv_database_url", lambda: None)

    # In cloud mode with no secrets, health check must report CONFIG_ERROR without crashing
    health = check_database_health()
    assert health.is_ready is False
    assert health.status_code == "CONFIG_ERROR"
    assert "DATABASE_URL" in health.message

    # get_connection() must raise ConfigurationError
    with pytest.raises(ConfigurationError):
        get_connection()


def test_req_j_select_1_health_check_works_with_test_database(temp_db: Path):
    """J. Verify SELECT 1 health check works with test database and closes connection."""
    init_db(temp_db)
    health = check_database_health(db_path=temp_db)

    assert health.is_ready is True
    assert health.status_code == "DATABASE_READY"
    assert health.backend_type == "sqlite"
    assert bool(health) is True


# ==============================================================================
# STEP 9 SEGMENTATION FAULT REGRESSION SUITE (PURE-PYTHON PG8000 DRIVER)
# ==============================================================================

def test_regression_1_configuration_resolution(monkeypatch):
    """1. Verify configuration resolution priority across secrets, env, and fallback."""
    import streamlit as st
    monkeypatch.setattr(st, "secrets", {
        "DATABASE_URL": "postgresql://pooler_user:pass123@aws-0-ap-northeast-2.pooler.supabase.com:5432/postgres?sslmode=require"
    })
    url = settings.get_database_url()
    assert "pooler.supabase.com" in url
    assert settings.get_database_source(canonical=True) == "STREAMLIT_SECRET"


def test_regression_2_session_pooler_detection():
    """2. Verify Session Pooler detection identifying port 5432 and pooler domain."""
    pooler_url = "postgresql://postgres.myproject:pass@aws-0-ap-northeast-2.pooler.supabase.com:5432/postgres"
    diag = extract_safe_db_diagnostics(pooler_url, "postgresql")
    assert diag["is_pooler"] is True
    assert diag["is_direct_supabase"] is False
    assert diag["port"] == 5432


def test_regression_3_postgresql_driver_selection():
    """3. Verify PostgreSQL driver selects pure-python pg8000 without compiled C segfault risks."""
    import pg8000
    assert pg8000.__file__ is not None
    diag = get_database_config_diagnostics(url="postgresql://usr:pwd@aws-0-pooler.supabase.com:5432/db")
    assert "pg8000" in diag["driver"]


def test_regression_4_successful_select_1_health_check(temp_db: Path):
    """4. Verify successful SELECT 1 health check validates operational readiness."""
    health = check_database_health(db_path=temp_db)
    assert health.is_ready is True
    assert health.status_code == "DATABASE_READY"
    assert "verified and operational" in health.message


def test_regression_5_failure_handling_without_crashing_process():
    """5. Verify unreachable PostgreSQL fails gracefully with DatabaseHealth error, never crashing process."""
    unreachable = "postgresql://mock_user:mock_pass@127.0.0.1:54328/mock_db"
    health = check_database_health(database_url=unreachable)
    assert health.is_ready is False
    assert health.status_code in ("CONNECTION_ERROR", "AUTH_ERROR", "DNS_ERROR")
    assert bool(health) is False


def test_regression_6_no_plaintext_password_in_diagnostics(capsys):
    """6. Verify absolute protection against plaintext password leakage in diagnostics and errors."""
    super_secret = "CLASSIFIED_VAULT_PASSWORD_999!"
    test_url = f"postgresql://usr_audit:{super_secret}@127.0.0.1:54328/test_db"
    
    health = check_database_health(database_url=test_url)
    captured = capsys.readouterr()
    combined_log = captured.err + captured.out + health.message + str(health.diagnostics)
    
    assert super_secret not in combined_log
    assert "CLASSIFIED" not in combined_log


def test_regression_7_no_direct_supabase_hostname_fallback():
    """7. Verify direct Supabase hostname is never generated or defaulted to."""
    pooler_url = "postgresql://postgres.xhpztsckvjdrnfgmmuuq:pwd@aws-0-ap-northeast-2.pooler.supabase.com:5432/postgres"
    _, target = parse_database_url(pooler_url)
    assert "aws-0-ap-northeast-2.pooler.supabase.com" in target
    assert "db.xhpztsckvjdrnfgmmuuq.supabase.co" not in target


def test_regression_8_sqlite_local_fallback_works_locally(monkeypatch):
    """8. Verify SQLite local fallback remains fully functional for local development."""
    import sys
    import streamlit as st
    s_mod = sys.modules["src.config.settings"]
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("STREAMLIT_COMMUNITY_CLOUD", raising=False)
    monkeypatch.delenv("IS_STREAMLIT_CLOUD", raising=False)
    monkeypatch.delenv("STREAMLIT_SERVER_ENVIRONMENT", raising=False)
    monkeypatch.setattr(st, "secrets", {})
    monkeypatch.setattr(s_mod, "_read_dotenv_database_url", lambda: None)

    backend = settings.get_database_backend()
    assert backend == "sqlite"
    health = check_database_health()
    assert health.backend_type == "sqlite"


