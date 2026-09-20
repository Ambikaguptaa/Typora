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
import sys
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
from urllib.parse import urlparse

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


class DatabaseHealth:
    """Truthful database health status object with error categorization and boolean evaluation."""

    def __init__(
        self,
        is_ready: bool,
        status_code: str,
        message: str,
        backend_type: str,
        diagnostics: Optional[Dict[str, Any]] = None,
    ):
        self.is_ready = bool(is_ready)
        # One of: DATABASE_READY, DNS_ERROR, AUTH_ERROR, CONNECTION_ERROR, CONFIG_ERROR
        self.status_code = status_code
        self.message = message
        self.backend_type = backend_type
        self.diagnostics = diagnostics or {}

    def __bool__(self) -> bool:
        return self.is_ready

    def __repr__(self) -> str:
        return f"<DatabaseHealth status={self.status_code} ready={self.is_ready} backend={self.backend_type}>"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, bool):
            return self.is_ready == other
        if isinstance(other, DatabaseHealth):
            return self.is_ready == other.is_ready and self.status_code == other.status_code
        return False


def mask_hostname_safely(host: Optional[str]) -> str:
    """Mask sensitive host identifiers while preserving domain structure."""
    if not host or host in ("localhost", "127.0.0.1", "parse_error", "unknown", "N/A"):
        return host or "unknown"
    if "pooler.supabase.com" in host:
        # e.g. aws-0-us-east-1.pooler.supabase.com -> aws-0-*.pooler.supabase.com
        parts = host.split(".")
        return f"{parts[0]}.*.pooler.supabase.com"
    if host.endswith(".supabase.co"):
        # e.g. db.xhpztsckvjdrnfgmmuuq.supabase.co -> db.xhpz****.supabase.co
        parts = host.split(".")
        if len(parts) >= 3 and parts[0] == "db":
            sub = parts[1]
            masked_sub = sub[:4] + "****" if len(sub) > 4 else "****"
            return f"db.{masked_sub}.supabase.co"
        return "****.supabase.co"
    if len(host) > 12:
        return host[:4] + "****" + host[-6:]
    return host


def mask_username_safely(username: Optional[str]) -> str:
    """Mask username credentials while revealing pooler tenant structure."""
    if not username or username in ("local", "unknown", "N/A"):
        return username or "unknown"
    if "." in username:
        # e.g. postgres.xhpztsckvjdrnfgmmuuq -> postgres.xhpz****
        prefix, suffix = username.split(".", 1)
        masked_suffix = suffix[:4] + "****" if len(suffix) > 4 else "****"
        return f"{prefix}.{masked_suffix}"
    if len(username) > 4:
        return username[:3] + "****"
    return username[:1] + "****"


def extract_safe_db_diagnostics(
    target: str,
    engine_type: str,
    config_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract safe database metadata for diagnostic logging without exposing passwords.

    Returns dictionary containing only:
    - config_source
    - backend_type
    - host (with password removed)
    - masked_host
    - port
    - database_name
    - username (with password removed)
    - masked_username
    - is_pooler (boolean)
    - is_direct_supabase (boolean)
    - has_database_url (boolean)
    """
    resolved_source = config_source or settings.get_database_source()
    if not target or engine_type == "sqlite":
        return {
            "config_source": resolved_source,
            "backend_type": "sqlite",
            "host": "localhost",
            "masked_host": "localhost",
            "port": "N/A",
            "database_name": target or str(settings.database_path),
            "username": "local",
            "masked_username": "local",
            "is_pooler": False,
            "is_direct_supabase": False,
            "has_database_url": bool(target),
        }

    try:
        parsed = urlparse(target)
        raw_host = parsed.hostname or "unknown"
        raw_user = parsed.username or "unknown"
        is_pooler = "pooler.supabase.com" in raw_host
        is_direct = raw_host.endswith(".supabase.co") and not is_pooler

        return {
            "config_source": resolved_source,
            "backend_type": engine_type,
            "host": raw_host,
            "masked_host": mask_hostname_safely(raw_host),
            "port": parsed.port or 5432,
            "database_name": parsed.path.lstrip("/") if parsed.path else "unknown",
            "username": raw_user,
            "masked_username": mask_username_safely(raw_user),
            "is_pooler": is_pooler,
            "is_direct_supabase": is_direct,
            "has_database_url": bool(target),
        }
    except Exception:
        return {
            "config_source": resolved_source,
            "backend_type": engine_type,
            "host": "parse_error",
            "masked_host": "parse_error",
            "port": 5432,
            "database_name": "unknown",
            "username": "unknown",
            "masked_username": "unknown",
            "is_pooler": False,
            "is_direct_supabase": False,
            "has_database_url": bool(target),
        }


def log_database_diagnostics(
    backend_type: str,
    host: Optional[str],
    port: Optional[Union[int, str]],
    database_name: Optional[str],
    username: Optional[str],
    has_database_url: bool,
    exc_type: Optional[str] = None,
    exc_message: Optional[str] = None,
    config_source: Optional[str] = None,
    is_pooler: Optional[bool] = None,
    is_direct_supabase: Optional[bool] = None,
) -> None:
    """Print and log safe database connection diagnostics with no passwords exposed."""
    src = config_source or settings.get_database_source()
    raw_host = host or "N/A"
    raw_user = username or "N/A"
    masked_h = mask_hostname_safely(raw_host) if raw_host != "N/A" else "N/A"
    masked_u = mask_username_safely(raw_user) if raw_user != "N/A" else "N/A"
    pooler_flag = is_pooler if is_pooler is not None else ("pooler.supabase.com" in str(raw_host))
    direct_flag = (
        is_direct_supabase
        if is_direct_supabase is not None
        else (str(raw_host).endswith(".supabase.co") and not pooler_flag)
    )

    msg_lines = [
        "=== DATABASE CONNECTION DIAGNOSTIC ===",
        f"DATABASE CONFIG SOURCE: {src}",
        f"DB BACKEND: {backend_type}",
        f"DB HOST: {masked_h}",
        f"DB PORT: {port or 'N/A'}",
        f"DB USER: {masked_u}",
        f"DB NAME: {database_name or 'N/A'}",
        f"POOLER DETECTED: {pooler_flag}",
        f"DIRECT SUPABASE HOST DETECTED: {direct_flag}",
        f"DATABASE_URL Exists: {has_database_url}",
        f"Backend Type: {backend_type}",
        f"Host: {raw_host}",
        f"Port: {port or 'N/A'}",
        f"Database Name: {database_name or 'N/A'}",
        f"Username: {raw_user}",
        f"psycopg2 Exception Type: {exc_type or 'None'}",
        f"psycopg2 Exception Message: {exc_message or 'None'}",
        "======================================",
    ]
    formatted = "\n".join(msg_lines)
    logger.info(formatted)
    print(formatted, file=sys.stderr, flush=True)


def parse_database_url(url: Optional[str] = None) -> Tuple[str, str]:
    """Determine the engine type ('sqlite' or 'postgresql') and target path/connection string.

    Args:
        url: Optional database URL. If None, resolves from application settings.

    Returns:
        Tuple[str, str]: (engine_type, connection_target)
    """
    db_url = url or settings.get_database_url()
    if db_url.startswith("postgresql://") or db_url.startswith("postgres://"):
        # Normalize postgres:// to postgresql:// for driver compatibility
        if db_url.startswith("postgres://"):
            db_url = "postgresql://" + db_url[len("postgres://") :]

        # Ensure sslmode=require for cloud PostgreSQL if not already specified
        if "sslmode=" not in db_url:
            delim = "&" if "?" in db_url else "?"
            db_url = f"{db_url}{delim}sslmode=require"

        # Ensure connect_timeout=10 if not already specified to avoid indefinite hangs
        if "connect_timeout=" not in db_url:
            delim = "&" if "?" in db_url else "?"
            db_url = f"{db_url}{delim}connect_timeout=10"

        return "postgresql", db_url

    # Normalize sqlite URL: sqlite:///path/to/file or sqlite:///:memory:
    if db_url.startswith("sqlite:///"):
        target = db_url[len("sqlite:///") :]
        return "sqlite", target
    elif db_url.startswith("sqlite://"):
        target = db_url[len("sqlite://") :]
        return "sqlite", target
    elif "://" in db_url:
        return "unknown", db_url

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

    if engine_type == "unknown":
        raise ValueError(f"Unsupported database scheme in URL: {sanitize_database_url_for_logging(target)}")

    if engine_type == "postgresql":
        diag = extract_safe_db_diagnostics(target, engine_type)

        # Check for direct Supabase IPv6 host
        host = str(diag.get("host", ""))
        if host.endswith(".supabase.co") and not host.startswith("aws-0-") and "pooler" not in host:
            warning_msg = (
                f"[DATABASE CONFIGURATION WARNING] Host '{host}' appears to be a direct Supabase endpoint (*.supabase.co).\n"
                "Direct Supabase endpoints resolve only over IPv6, which is NOT supported by Streamlit Community Cloud.\n"
                "Please configure the Supabase Session Pooler host (aws-0-[region].pooler.supabase.com) on port 5432 instead."
            )
            logger.warning(warning_msg)
            print(warning_msg, file=sys.stderr, flush=True)

        try:
            import psycopg2
            import psycopg2.extras
        except ImportError:
            try:
                import psycopg

                conn = psycopg.connect(target)
                log_database_diagnostics(
                    backend_type="postgresql",
                    host=diag["host"],
                    port=diag["port"],
                    database_name=diag["database_name"],
                    username=diag["username"],
                    has_database_url=diag["has_database_url"],
                )
                return conn
            except ImportError:
                log_database_diagnostics(
                    backend_type="postgresql",
                    host=diag["host"],
                    port=diag["port"],
                    database_name=diag["database_name"],
                    username=diag["username"],
                    has_database_url=diag["has_database_url"],
                    exc_type="ImportError",
                    exc_message="psycopg2 and psycopg drivers are not installed.",
                )
                raise ImportError(
                    "PostgreSQL driver (psycopg2 or psycopg) is required for PostgreSQL backend. "
                    "Install psycopg2-binary or configure a local SQLite database."
                )

        # Log safe diagnostics prior to connection attempt so configuration is immediately visible in Streamlit logs
        log_database_diagnostics(
            backend_type="postgresql",
            host=diag["host"],
            port=diag["port"],
            database_name=diag["database_name"],
            username=diag["username"],
            has_database_url=diag["has_database_url"],
            exc_type=None,
            exc_message=None,
            config_source=diag.get("config_source"),
            is_pooler=diag.get("is_pooler"),
            is_direct_supabase=diag.get("is_direct_supabase"),
        )

        try:
            conn = psycopg2.connect(
                target,
                connect_timeout=10,
                cursor_factory=psycopg2.extras.DictCursor,
            )
            conn.autocommit = False
            logger.info("[DATABASE SUCCESS] PostgreSQL connection established successfully.")
            return conn
        except Exception as pg_err:
            exc_type = type(pg_err).__name__
            raw_msg = str(pg_err).strip()
            exc_msg = sanitize_database_url_for_logging(raw_msg)
            try:
                parsed = urlparse(target)
                if parsed.password and parsed.password in exc_msg:
                    exc_msg = exc_msg.replace(parsed.password, "********")
            except Exception:
                pass

            log_database_diagnostics(
                backend_type="postgresql",
                host=diag["host"],
                port=diag["port"],
                database_name=diag["database_name"],
                username=diag["username"],
                has_database_url=diag["has_database_url"],
                exc_type=exc_type,
                exc_message=exc_msg,
                config_source=diag.get("config_source"),
                is_pooler=diag.get("is_pooler"),
                is_direct_supabase=diag.get("is_direct_supabase"),
            )
            if host.endswith(".supabase.co") and not host.startswith("aws-0-") and "pooler" not in host:
                pooler_advice = (
                    "[DATABASE CONFIGURATION FIX]\n"
                    "Direct connection to db.*.supabase.co failed. Streamlit Community Cloud lacks IPv6 routing.\n"
                    "To fix this in Supabase:\n"
                    "1. Go to Supabase Dashboard > Project Settings > Database > Connection Pooling.\n"
                    "2. Select 'Session' mode on port 5432 (e.g. host: aws-0-[region].pooler.supabase.com).\n"
                    "3. Update DATABASE_URL in Streamlit Cloud Secrets with username 'postgres.[project-ref]':\n"
                    "   DATABASE_URL = 'postgresql://postgres.[project-ref]:[PASSWORD]@aws-0-[region].pooler.supabase.com:5432/postgres?sslmode=require'\n"
                )
                logger.error(pooler_advice)
                print(pooler_advice, file=sys.stderr, flush=True)
            raise
    else:
        # SQLite
        if target != ":memory:":
            p = Path(target)
            p.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(target)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn


class DialectCursorWrapper:
    """Cursor wrapper providing unified parameter translation across SQLite ('?') and PostgreSQL ('%s')."""

    def __init__(self, raw_cursor: Any, is_postgres: bool = False):
        self._cursor = raw_cursor
        self._is_postgres = is_postgres

    def execute(self, query: str, params: Optional[Union[tuple, list, dict]] = None) -> Any:
        if self._is_postgres and params is not None and isinstance(params, (tuple, list)):
            # Translate SQLite '?' parameter placeholders to PostgreSQL '%s'
            query = query.replace("?", "%s")
        if params is not None:
            return self._cursor.execute(query, params)
        return self._cursor.execute(query)

    def fetchone(self) -> Any:
        return self._cursor.fetchone()

    def fetchall(self) -> List[Any]:
        return self._cursor.fetchall()

    def fetchmany(self, size: Optional[int] = None) -> List[Any]:
        return self._cursor.fetchmany(size) if size else self._cursor.fetchmany()

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount

    def close(self) -> None:
        return self._cursor.close()

    def __iter__(self) -> Generator[Any, None, None]:
        return iter(self._cursor)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)


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
        Database cursor (wrapped for cross-dialect compatibility).
    """
    conn = get_connection(db_path=db_path, database_url=database_url)
    engine_type, _ = parse_database_url(database_url if db_path is None else f"sqlite:///{db_path}")
    is_postgres = (engine_type == "postgresql") and (db_path is None)

    cursor = conn.cursor()
    wrapped = DialectCursorWrapper(cursor, is_postgres=is_postgres)
    try:
        yield wrapped
        conn.commit()
    except Exception as e:
        conn.rollback()
        safe_msg = sanitize_database_url_for_logging(str(e))
        logger.error(f"Database operation failed: {safe_msg}")
        raise
    finally:
        wrapped.close()
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
    raise_on_error: bool = False,
) -> bool:
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

    try:
        with get_db_cursor(db_path=db_path, database_url=database_url) as cursor:
            for stmt in statements:
                cursor.execute(stmt)
        logger.info(f"Database schema initialized successfully on {engine_type} backend.")
        return True
    except Exception as e:
        safe_msg = sanitize_database_url_for_logging(str(e))
        logger.error(f"Database schema initialization failed: {safe_msg}")
        if raise_on_error:
            raise
        return False


def check_database_health(
    db_path: Optional[Union[str, Path]] = None,
    database_url: Optional[str] = None,
) -> DatabaseHealth:
    """Test database connectivity with truthful categorization and safe diagnostics.

    Returns:
        DatabaseHealth: Status object distinguishing:
        - DATABASE_READY
        - DNS_ERROR
        - AUTH_ERROR
        - CONNECTION_ERROR
        - CONFIG_ERROR
    """
    url_target = database_url if db_path is None else f"sqlite:///{db_path}"
    engine_type, target = parse_database_url(url_target)
    diag = extract_safe_db_diagnostics(target, engine_type)

    if engine_type == "unknown":
        log_database_diagnostics(
            backend_type="unknown",
            host=diag.get("host"),
            port=diag.get("port"),
            database_name=diag.get("database_name"),
            username=diag.get("username"),
            has_database_url=diag.get("has_database_url", False),
            exc_type="ConfigurationError",
            exc_message=f"Unsupported or malformed database URL scheme in: {sanitize_database_url_for_logging(target)}",
            config_source=diag.get("config_source"),
        )
        return DatabaseHealth(
            is_ready=False,
            status_code="CONFIG_ERROR",
            message=f"Unsupported or malformed database URL scheme in: {sanitize_database_url_for_logging(target)}",
            backend_type="unknown",
            diagnostics=diag,
        )

    if engine_type == "postgresql":
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError:
            log_database_diagnostics(
                backend_type="postgresql",
                host=diag["host"],
                port=diag["port"],
                database_name=diag["database_name"],
                username=diag["username"],
                has_database_url=diag["has_database_url"],
                exc_type="ImportError",
                exc_message="psycopg2 driver not installed",
                config_source=diag.get("config_source"),
                is_pooler=diag.get("is_pooler"),
                is_direct_supabase=diag.get("is_direct_supabase"),
            )
            return DatabaseHealth(
                is_ready=False,
                status_code="CONFIG_ERROR",
                message="PostgreSQL driver (psycopg2) is not installed.",
                backend_type="postgresql",
                diagnostics=diag,
            )

        conn = None
        cursor = None
        try:
            conn = psycopg2.connect(
                target,
                connect_timeout=10,
                cursor_factory=psycopg2.extras.DictCursor,
            )
            cursor = conn.cursor()
            cursor.execute("SELECT 1;")
            result = cursor.fetchone()
            is_ok = result is not None and result[0] == 1
            if is_ok:
                log_database_diagnostics(
                    backend_type="postgresql",
                    host=diag["host"],
                    port=diag["port"],
                    database_name=diag["database_name"],
                    username=diag["username"],
                    has_database_url=diag["has_database_url"],
                    config_source=diag.get("config_source"),
                    is_pooler=diag.get("is_pooler"),
                    is_direct_supabase=diag.get("is_direct_supabase"),
                )
                return DatabaseHealth(
                    is_ready=True,
                    status_code="DATABASE_READY",
                    message="PostgreSQL connection verified and operational.",
                    backend_type="postgresql",
                    diagnostics=diag,
                )
            else:
                return DatabaseHealth(
                    is_ready=False,
                    status_code="CONNECTION_ERROR",
                    message="Database query returned unexpected result.",
                    backend_type="postgresql",
                    diagnostics=diag,
                )
        except Exception as e:
            exc_type = type(e).__name__
            raw_msg = str(e).strip()
            exc_msg = sanitize_database_url_for_logging(raw_msg)
            try:
                parsed = urlparse(target)
                if parsed.password and parsed.password in exc_msg:
                    exc_msg = exc_msg.replace(parsed.password, "********")
            except Exception:
                pass

            msg_lower = exc_msg.lower()
            if any(k in msg_lower for k in ["could not translate host", "no address associated", "gaierror", "name or service not known"]):
                status_code = "DNS_ERROR"
            elif any(k in msg_lower for k in ["password authentication failed", "role", "access denied", "authentication failed"]):
                status_code = "AUTH_ERROR"
            elif any(k in msg_lower for k in ["timeout", "timed out", "unreachable", "refused", "could not connect", "ssl"]):
                status_code = "CONNECTION_ERROR"
            else:
                status_code = "CONNECTION_ERROR"

            log_database_diagnostics(
                backend_type="postgresql",
                host=diag["host"],
                port=diag["port"],
                database_name=diag["database_name"],
                username=diag["username"],
                has_database_url=diag["has_database_url"],
                exc_type=exc_type,
                exc_message=exc_msg,
                config_source=diag.get("config_source"),
                is_pooler=diag.get("is_pooler"),
                is_direct_supabase=diag.get("is_direct_supabase"),
            )
            return DatabaseHealth(
                is_ready=False,
                status_code=status_code,
                message=exc_msg,
                backend_type="postgresql",
                diagnostics=diag,
            )
        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
    else:
        # SQLite
        try:
            with get_db_cursor(db_path=db_path, database_url=database_url) as cursor:
                cursor.execute("SELECT 1;")
                result = cursor.fetchone()
                if result is not None and result[0] == 1:
                    return DatabaseHealth(
                        is_ready=True,
                        status_code="DATABASE_READY",
                        message="SQLite connection verified and operational.",
                        backend_type="sqlite",
                        diagnostics=diag,
                    )
                return DatabaseHealth(
                    is_ready=False,
                    status_code="CONNECTION_ERROR",
                    message="Unexpected result from SQLite.",
                    backend_type="sqlite",
                    diagnostics=diag,
                )
        except Exception as e:
            safe_msg = sanitize_database_url_for_logging(str(e))
            return DatabaseHealth(
                is_ready=False,
                status_code="CONNECTION_ERROR",
                message=safe_msg,
                backend_type="sqlite",
                diagnostics=diag,
            )


def check_connection(
    db_path: Optional[Union[str, Path]] = None,
    database_url: Optional[str] = None,
    return_health: bool = False,
) -> Union[bool, DatabaseHealth]:
    """Test if the database can be connected to and executed upon.

    Args:
        db_path: Optional SQLite database file path.
        database_url: Optional database connection string.
        return_health: If True, returns rich DatabaseHealth object instead of bool.

    Returns:
        bool by default (True if operational, False otherwise), or DatabaseHealth if return_health=True.
    """
    health = check_database_health(db_path=db_path, database_url=database_url)
    if return_health:
        return health
    return health.is_ready


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
