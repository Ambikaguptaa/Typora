"""SQLite database connection and schema initialization module.

Provides simple, robust connection management for storing non-sensitive typing
metrics and session metadata. No raw keystroke characters are ever persisted.
"""

from contextlib import contextmanager
import logging
from pathlib import Path
import sqlite3
from typing import Generator, Optional, Union

from src.config.settings import settings

logger = logging.getLogger(__name__)


def get_connection(
    db_path: Optional[Union[str, Path]] = None,
) -> sqlite3.Connection:
    """Create and return a configured SQLite connection.

    Args:
        db_path: Path to the SQLite database file, or ':memory:'.
                 If None, uses the path from application settings.

    Returns:
        sqlite3.Connection: Database connection with row factory enabled.
    """
    if db_path is None:
        target_path = settings.database_path
        # Ensure parent directory exists for file-based DB
        if str(target_path) != ":memory:":
            Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(target_path))
    else:
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))

    # Enable column access by name
    conn.row_factory = sqlite3.Row
    # Enforce foreign key constraints
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


@contextmanager
def get_db_cursor(
    db_path: Optional[Union[str, Path]] = None,
) -> Generator[sqlite3.Cursor, None, None]:
    """Context manager for executing database operations with auto-commit and rollback.

    Args:
        db_path: Optional database path.

    Yields:
        sqlite3.Cursor: Database cursor.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        yield cursor
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error encountered: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def init_db(db_path: Optional[Union[str, Path]] = None) -> None:
    """Initialize initial database tables if they do not exist.

    Creates:
    - sessions: Stores pseudonymized session metadata.
    - typing_metrics: Stores computed typing behavioral aggregates.
    """
    schema_statements = [
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            pseudonymized_user_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            sample_count INTEGER DEFAULT 0,
            status TEXT DEFAULT 'active'
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS typing_metrics (
            metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            mean_hold_time_ms REAL,
            mean_flight_time_ms REAL,
            pause_rate REAL,
            estimated_strain_score REAL,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        );
        """,
    ]

    with get_db_cursor(db_path) as cursor:
        for stmt in schema_statements:
            cursor.execute(stmt)

    logger.info("Database schema initialized successfully.")


def check_connection(db_path: Optional[Union[str, Path]] = None) -> bool:
    """Test if the database can be connected to and executed upon.

    Returns:
        bool: True if connection test succeeds, False otherwise.
    """
    try:
        with get_db_cursor(db_path) as cursor:
            cursor.execute("SELECT 1;")
            result = cursor.fetchone()
            return result is not None and result[0] == 1
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")
        return False
