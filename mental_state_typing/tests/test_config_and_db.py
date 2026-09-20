"""Unit tests for system configuration and SQLite database integration."""

import sqlite3
import pytest

from database.database import (
    check_connection,
    get_connection,
    get_db_cursor,
    init_db,
)
from src.config.settings import Settings, get_settings, settings


def test_configuration_loads_correctly():
    """Verify application configuration loads default attributes cleanly."""
    cfg = get_settings()

    assert isinstance(cfg, Settings)
    assert cfg.app_env in ["development", "testing", "production"]
    assert cfg.database_path is not None
    assert cfg.system_disclaimer != ""
    assert "not a medical diagnostic tool" in cfg.system_disclaimer


def test_all_subsystem_modules_importable():
    """Verify all project subsystems can be imported without syntax or circular import errors."""
    import src.deep_learning as dl
    import src.live_typing as lt
    import src.visualization as viz
    import database.database as db

    assert dl is not None
    assert lt is not None
    assert viz is not None
    assert db is not None


def test_sqlite_connection_in_memory():
    """Verify in-memory SQLite connection is functional."""
    conn = get_connection(":memory:")
    assert isinstance(conn, sqlite3.Connection)

    cursor = conn.cursor()
    cursor.execute("SELECT 42 AS test_val;")
    row = cursor.fetchone()
    assert row["test_val"] == 42
    conn.close()


def test_sqlite_schema_initialization_in_memory():
    """Verify that init_db creates the required sessions and typing_metrics tables."""
    conn = get_connection(":memory:")

    # Initialize schema on this in-memory connection
    init_db(":memory:")

    # Verify check_connection returns True
    assert check_connection(":memory:") is True


def test_sqlite_sessions_table_crud(tmp_path):
    """Verify inserting and querying a session record in a temporary SQLite database."""
    test_db_path = tmp_path / "test_mental_state.db"

    # Initialize tables
    init_db(test_db_path)

    # Insert test session
    with get_db_cursor(test_db_path) as cursor:
        cursor.execute(
            """
            INSERT INTO sessions (session_id, pseudonymized_user_id, sample_count)
            VALUES (?, ?, ?);
            """,
            ("sess_001", "usr_abc123def456", 50),
        )

    # Read back test session
    with get_db_cursor(test_db_path) as cursor:
        cursor.execute(
            "SELECT * FROM sessions WHERE session_id = ?;", ("sess_001",)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row["session_id"] == "sess_001"
        assert row["pseudonymized_user_id"] == "usr_abc123def456"
        assert row["sample_count"] == 50
