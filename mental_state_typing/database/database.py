"""Database connection, schema management, and persistent storage abstraction module.

Supports:
- Local development: SQLite (default fallback)
- Deployed Streamlit Cloud: PostgreSQL (configured via DATABASE_URL or Streamlit secrets)

Strict Security & Privacy Invariants:
- Zero raw text or keystroke characters are ever persisted.
- Forbidden fields (key, char, text, typed_text, password, word, etc.) trigger immediate PrivacyViolationError.
- Sensitive credentials are never exposed in log outputs or error messages.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import sqlite3
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from src.config.settings import settings
from src.live_typing.privacy_filter import (
    FORBIDDEN_PAYLOAD_FIELDS,
    PrivacyViolationError,
    audit_payload_for_sensitive_keys,
)

logger = logging.getLogger(__name__)


def sanitize_database_url_for_logging(url: str) -> str:
    """Mask credentials in database URLs for privacy-safe logging."""
    if not url:
        return "none"
    # Matches scheme://user:password@host...
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:********@", url)


def parse_database_url(url: Optional[str] = None) -> Tuple[str, str]:
    """Determine the engine type ('sqlite' or 'postgresql') and target path/connection string.

    Args:
        url: Optional database URL. If None, resolves from application settings.

    Returns:
        Tuple[str, str]: (engine_type, connection_target)
    """
    db_url = url or settings.get_database_url()
    if db_url.startswith("postgresql://") or db_url.startswith("postgres://"):
        return "postgresql", db_url

    # Normalize sqlite URL: sqlite:///path/to/file or sqlite:///:memory:
    if db_url.startswith("sqlite:///"):
        target = db_url[len("sqlite:///") :]
        return "sqlite", target
    elif db_url.startswith("sqlite://"):
        target = db_url[len("sqlite://") :]
        return "sqlite", target

    return "sqlite", str(settings.database_path)


def get_connection(
    db_path: Optional[Union[str, Path]] = None,
    database_url: Optional[str] = None,
) -> Any:
    """Create and return a configured database connection.

    Supports both SQLite and PostgreSQL backends based on configuration.
    Maintains exact backward compatibility for existing callers passing db_path.

    Args:
        db_path: Optional path to SQLite file or ':memory:'.
        database_url: Optional explicit DATABASE_URL connection string.

    Returns:
        Database connection object (sqlite3.Connection or psycopg2.connection).
    """
    if db_path is not None:
        target = str(db_path)
        if target != ":memory:":
            Path(target).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(target)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    engine_type, target = parse_database_url(database_url)

    if engine_type == "postgresql":
        try:
            import psycopg2
            import psycopg2.extras

            conn = psycopg2.connect(target)
            conn.autocommit = False
            return conn
        except ImportError:
            try:
                import psycopg

                conn = psycopg.connect(target)
                return conn
            except ImportError:
                raise ImportError(
                    "PostgreSQL driver (psycopg2 or psycopg) is required for PostgreSQL backend. "
                    "Install psycopg2-binary or configure a local SQLite database."
                )
    else:
        # SQLite
        if target != ":memory:":
            p = Path(target)
            p.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(target)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn


@contextmanager
def get_db_cursor(
    db_path: Optional[Union[str, Path]] = None,
    database_url: Optional[str] = None,
) -> Generator[Any, None, None]:
    """Context manager for executing database operations with auto-commit and rollback.

    Args:
        db_path: Optional SQLite database path.
        database_url: Optional database connection string.

    Yields:
        Database cursor.
    """
    conn = get_connection(db_path=db_path, database_url=database_url)
    cursor = conn.cursor()
    try:
        yield cursor
        conn.commit()
    except Exception as e:
        conn.rollback()
        safe_msg = sanitize_database_url_for_logging(str(e))
        logger.error(f"Database operation failed: {safe_msg}")
        raise
    finally:
        cursor.close()
        conn.close()


def _get_schema_statements(is_postgres: bool = False) -> List[str]:
    """Generate SQL DDL statements adapted for the target database dialect."""
    auto_inc = "SERIAL PRIMARY KEY" if is_postgres else "INTEGER PRIMARY KEY AUTOINCREMENT"

    return [
        f"""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            pseudonymized_user_id TEXT NOT NULL,
            session_type TEXT DEFAULT 'ANALYSIS',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            sample_count INTEGER DEFAULT 0,
            status TEXT DEFAULT 'active'
        );
        """,
        f"""
        CREATE TABLE IF NOT EXISTS typing_metrics (
            metric_id {auto_inc},
            session_id TEXT NOT NULL,
            mean_hold_time_ms REAL,
            mean_flight_time_ms REAL,
            pause_rate REAL,
            estimated_strain_score REAL,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        );
        """,
        f"""
        CREATE TABLE IF NOT EXISTS assessments (
            assessment_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            pseudonymized_user_id TEXT NOT NULL,
            session_type TEXT DEFAULT 'ANALYSIS',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            duration_s REAL DEFAULT 0.0,
            event_count INTEGER DEFAULT 0,
            mean_dwell_ms REAL,
            mean_flight_ms REAL,
            mean_pause_ms REAL,
            pause_rate REAL,
            typing_speed_wpm REAL,
            backspace_rate REAL,
            tdi_score REAL,
            tdi_divergence_level TEXT,
            model_predicted_class TEXT,
            model_confidence REAL,
            model_status TEXT DEFAULT 'MODEL_NOT_READY',
            assessment_json TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        );
        """,
        f"""
        CREATE TABLE IF NOT EXISTS baseline_profiles (
            user_id TEXT PRIMARY KEY,
            session_count INTEGER DEFAULT 0,
            is_ready INTEGER DEFAULT 0,
            mean_dwell_time_ms REAL,
            std_dwell_time_ms REAL,
            mean_flight_time_ms REAL,
            std_flight_time_ms REAL,
            mean_pause_duration_ms REAL,
            std_pause_duration_ms REAL,
            mean_pause_rate REAL,
            std_pause_rate REAL,
            mean_typing_speed_wpm REAL,
            std_typing_speed_wpm REAL,
            mean_backspace_rate REAL,
            std_backspace_rate REAL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        f"""
        CREATE TABLE IF NOT EXISTS audit_logs (
            log_id {auto_inc},
            event_type TEXT NOT NULL,
            user_id TEXT,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
    ]


def init_db(
    db_path: Optional[Union[str, Path]] = None,
    database_url: Optional[str] = None,
) -> None:
    """Initialize database tables across SQLite or PostgreSQL backends.

    Creates tables:
    - sessions
    - typing_metrics
    - assessments (persistent derived assessment records)
    - baseline_profiles (persistent personal baseline parameters)
    - audit_logs (security & privacy operational log)
    """
    engine_type, _ = parse_database_url(database_url if db_path is None else f"sqlite:///{db_path}")
    is_postgres = (engine_type == "postgresql") and (db_path is None)

    statements = _get_schema_statements(is_postgres=is_postgres)

    with get_db_cursor(db_path=db_path, database_url=database_url) as cursor:
        for stmt in statements:
            cursor.execute(stmt)

    logger.info(f"Database schema initialized successfully on {engine_type} backend.")


def check_connection(
    db_path: Optional[Union[str, Path]] = None,
    database_url: Optional[str] = None,
) -> bool:
    """Test if the database can be connected to and executed upon.

    Returns:
        bool: True if connection test succeeds, False otherwise.
    """
    try:
        with get_db_cursor(db_path=db_path, database_url=database_url) as cursor:
            cursor.execute("SELECT 1;")
            result = cursor.fetchone()
            return result is not None and result[0] == 1
    except Exception as e:
        safe_msg = sanitize_database_url_for_logging(str(e))
        logger.warning(f"Database health check failed: {safe_msg}")
        return False


# ==============================================================================
# PERSISTENT STORAGE CRUD OPERATIONS (STRICT ZERO-RAW-TEXT INVARIANT)
# ==============================================================================

def save_session_record(
    session_id: str,
    user_id: str,
    session_type: str = "ANALYSIS",
    sample_count: int = 0,
    status: str = "completed",
    db_path: Optional[Union[str, Path]] = None,
) -> None:
    """Persist session metadata record."""
    violations = audit_payload_for_sensitive_keys(
        {"session_id": session_id, "user_id": user_id, "session_type": session_type, "status": status}
    )
    if violations:
        raise PrivacyViolationError(f"Forbidden raw text keys in session metadata: {violations}")

    with get_db_cursor(db_path=db_path) as cursor:
        cursor.execute(
            """
            INSERT INTO sessions (session_id, pseudonymized_user_id, session_type, sample_count, status)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                pseudonymized_user_id = excluded.pseudonymized_user_id,
                session_type = excluded.session_type,
                sample_count = excluded.sample_count,
                status = excluded.status;
            """,
            (session_id, user_id, session_type, sample_count, status),
        )


def save_assessment_record(
    assessment_data: Dict[str, Any],
    db_path: Optional[Union[str, Path]] = None,
) -> str:
    """Persist derived behavioral assessment to the database.

    Enforces strict zero-raw-text validation before persisting.

    Args:
        assessment_data: Dictionary containing assessment result.
        db_path: Optional database path override.

    Returns:
        str: Assessment ID.
    """
    # Strict privacy audit on the entire payload
    violations = audit_payload_for_sensitive_keys(assessment_data)
    if violations:
        raise PrivacyViolationError(f"Forbidden raw text keys rejected in database assessment: {violations}")

    session_id = assessment_data.get("session_id", "unknown")
    assessment_id = f"asmt_{session_id}"
    user_id = assessment_data.get("user_id", "anonymous")
    session_type = assessment_data.get("session_type", "ANALYSIS")
    timestamp_str = assessment_data.get("timestamp", datetime.now(timezone.utc).isoformat())

    # Extract derived timing metrics safely
    dq = assessment_data.get("data_quality", {})
    dq_metrics = dq.get("metrics", {})
    duration_s = float(dq_metrics.get("active_duration_s", 0.0))
    event_count = int(dq_metrics.get("paired_events_count", 0))

    # Features / timing metrics
    asmt_res = assessment_data.get("assessment_result", {})
    feature_summary = asmt_res.get("feature_summary", {})
    mean_dwell = feature_summary.get("mean_dwell_time_ms")
    mean_flight = feature_summary.get("mean_flight_time_ms")
    mean_pause = feature_summary.get("mean_pause_duration_ms")
    pause_rate = feature_summary.get("pause_rate")
    typing_speed = feature_summary.get("typing_speed_wpm")
    backspace_rate = feature_summary.get("backspace_rate")

    # Baseline metrics
    base_res = assessment_data.get("baseline_result", {})
    tdi_score = base_res.get("typing_deviation_index")
    tdi_level = base_res.get("divergence_level", "NORMAL")

    # Model metrics
    model_res = assessment_data.get("model_result", {})
    model_status = model_res.get("status", "MODEL_NOT_READY")
    model_class = model_res.get("predicted_class")
    model_conf = model_res.get("confidence")

    # Ensure parent session exists
    save_session_record(
        session_id=session_id,
        user_id=user_id,
        session_type=session_type,
        sample_count=event_count,
        status="completed",
        db_path=db_path,
    )

    assessment_json = json.dumps(assessment_data)

    with get_db_cursor(db_path=db_path) as cursor:
        cursor.execute(
            """
            INSERT INTO assessments (
                assessment_id, session_id, pseudonymized_user_id, session_type,
                created_at, duration_s, event_count, mean_dwell_ms, mean_flight_ms,
                mean_pause_ms, pause_rate, typing_speed_wpm, backspace_rate,
                tdi_score, tdi_divergence_level, model_predicted_class, model_confidence,
                model_status, assessment_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(assessment_id) DO UPDATE SET
                duration_s = excluded.duration_s,
                event_count = excluded.event_count,
                mean_dwell_ms = excluded.mean_dwell_ms,
                mean_flight_ms = excluded.mean_flight_ms,
                mean_pause_ms = excluded.mean_pause_ms,
                pause_rate = excluded.pause_rate,
                typing_speed_wpm = excluded.typing_speed_wpm,
                backspace_rate = excluded.backspace_rate,
                tdi_score = excluded.tdi_score,
                tdi_divergence_level = excluded.tdi_divergence_level,
                model_predicted_class = excluded.model_predicted_class,
                model_confidence = excluded.model_confidence,
                model_status = excluded.model_status,
                assessment_json = excluded.assessment_json;
            """,
            (
                assessment_id,
                session_id,
                user_id,
                session_type,
                timestamp_str,
                duration_s,
                event_count,
                mean_dwell,
                mean_flight,
                mean_pause,
                pause_rate,
                typing_speed,
                backspace_rate,
                tdi_score,
                tdi_level,
                model_class,
                model_conf,
                model_status,
                assessment_json,
            ),
        )

    logger.info(f"Persisted assessment {assessment_id} to database.")
    return assessment_id


def get_assessment_records(
    user_id: Optional[str] = None,
    limit: int = 100,
    db_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve persisted assessment summaries from the database."""
    with get_db_cursor(db_path=db_path) as cursor:
        if user_id:
            cursor.execute(
                """
                SELECT assessment_id, session_id, pseudonymized_user_id, session_type,
                       created_at, duration_s, event_count, mean_dwell_ms, mean_flight_ms,
                       pause_rate, typing_speed_wpm, tdi_score, tdi_divergence_level,
                       model_status, model_predicted_class, model_confidence
                FROM assessments
                WHERE pseudonymized_user_id = ?
                ORDER BY created_at DESC
                LIMIT ?;
                """,
                (user_id, limit),
            )
        else:
            cursor.execute(
                """
                SELECT assessment_id, session_id, pseudonymized_user_id, session_type,
                       created_at, duration_s, event_count, mean_dwell_ms, mean_flight_ms,
                       pause_rate, typing_speed_wpm, tdi_score, tdi_divergence_level,
                       model_status, model_predicted_class, model_confidence
                FROM assessments
                ORDER BY created_at DESC
                LIMIT ?;
                """,
                (limit,),
            )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_assessment_by_id(
    session_id: str,
    db_path: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve complete assessment dictionary by session_id from database."""
    with get_db_cursor(db_path=db_path) as cursor:
        cursor.execute(
            "SELECT assessment_json FROM assessments WHERE session_id = ? OR assessment_id = ?;",
            (session_id, f"asmt_{session_id}"),
        )
        row = cursor.fetchone()
        if not row or not row["assessment_json"]:
            return None

        data = json.loads(row["assessment_json"])
        violations = audit_payload_for_sensitive_keys(data)
        if violations:
            raise PrivacyViolationError(f"Corrupted assessment contains forbidden keys: {violations}")
        return data


def delete_assessment_record(
    session_id: str,
    db_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Purge assessment and associated session from database."""
    with get_db_cursor(db_path=db_path) as cursor:
        cursor.execute(
            "DELETE FROM assessments WHERE session_id = ? OR assessment_id = ?;",
            (session_id, f"asmt_{session_id}"),
        )
        asmt_deleted = cursor.rowcount > 0
        cursor.execute("DELETE FROM sessions WHERE session_id = ?;", (session_id,))
        return asmt_deleted


def save_baseline_profile(
    user_id: str,
    profile_data: Dict[str, Any],
    db_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Persist user personal baseline profile to database."""
    violations = audit_payload_for_sensitive_keys(profile_data)
    if violations:
        raise PrivacyViolationError(f"Forbidden raw text keys in baseline profile: {violations}")

    session_count = int(profile_data.get("session_count", 0))
    is_ready = 1 if profile_data.get("is_ready", False) else 0

    with get_db_cursor(db_path=db_path) as cursor:
        cursor.execute(
            """
            INSERT INTO baseline_profiles (
                user_id, session_count, is_ready, mean_dwell_time_ms, std_dwell_time_ms,
                mean_flight_time_ms, std_flight_time_ms, mean_pause_duration_ms, std_pause_duration_ms,
                mean_pause_rate, std_pause_rate, mean_typing_speed_wpm, std_typing_speed_wpm,
                mean_backspace_rate, std_backspace_rate, last_updated
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                session_count = excluded.session_count,
                is_ready = excluded.is_ready,
                mean_dwell_time_ms = excluded.mean_dwell_time_ms,
                std_dwell_time_ms = excluded.std_dwell_time_ms,
                mean_flight_time_ms = excluded.mean_flight_time_ms,
                std_flight_time_ms = excluded.std_flight_time_ms,
                mean_pause_duration_ms = excluded.mean_pause_duration_ms,
                std_pause_duration_ms = excluded.std_pause_duration_ms,
                mean_pause_rate = excluded.mean_pause_rate,
                std_pause_rate = excluded.std_pause_rate,
                mean_typing_speed_wpm = excluded.mean_typing_speed_wpm,
                std_typing_speed_wpm = excluded.std_typing_speed_wpm,
                mean_backspace_rate = excluded.mean_backspace_rate,
                std_backspace_rate = excluded.std_backspace_rate,
                last_updated = CURRENT_TIMESTAMP;
            """,
            (
                user_id,
                session_count,
                is_ready,
                profile_data.get("mean_dwell_time_ms"),
                profile_data.get("std_dwell_time_ms"),
                profile_data.get("mean_flight_time_ms"),
                profile_data.get("std_flight_time_ms"),
                profile_data.get("mean_pause_duration_ms"),
                profile_data.get("std_pause_duration_ms"),
                profile_data.get("mean_pause_rate"),
                profile_data.get("std_pause_rate"),
                profile_data.get("mean_typing_speed_wpm"),
                profile_data.get("std_typing_speed_wpm"),
                profile_data.get("mean_backspace_rate"),
                profile_data.get("std_backspace_rate"),
            ),
        )
    return True


def get_baseline_profile(
    user_id: str,
    db_path: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve persisted baseline profile for user from database."""
    with get_db_cursor(db_path=db_path) as cursor:
        cursor.execute(
            "SELECT * FROM baseline_profiles WHERE user_id = ?;",
            (user_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return dict(row)


def record_audit_log(
    event_type: str,
    user_id: Optional[str] = None,
    details: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> None:
    """Record an operational audit log entry."""
    violations = audit_payload_for_sensitive_keys({"event_type": event_type, "details": details or ""})
    if violations:
        raise PrivacyViolationError(f"Forbidden raw text keys in audit log: {violations}")

    with get_db_cursor(db_path=db_path) as cursor:
        cursor.execute(
            "INSERT INTO audit_logs (event_type, user_id, details) VALUES (?, ?, ?);",
            (event_type, user_id, details),
        )
