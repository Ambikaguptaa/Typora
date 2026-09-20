"""Database package for local SQLite and cloud PostgreSQL persistence."""

from database.database import (
    check_connection,
    delete_assessment_record,
    get_assessment_by_id,
    get_assessment_records,
    get_baseline_profile,
    get_connection,
    get_db_cursor,
    init_db,
    parse_database_url,
    record_audit_log,
    sanitize_database_url_for_logging,
    save_assessment_record,
    save_baseline_profile,
    save_session_record,
)

__all__ = [
    "check_connection",
    "delete_assessment_record",
    "get_assessment_by_id",
    "get_assessment_records",
    "get_baseline_profile",
    "get_connection",
    "get_db_cursor",
    "init_db",
    "parse_database_url",
    "record_audit_log",
    "sanitize_database_url_for_logging",
    "save_assessment_record",
    "save_baseline_profile",
    "save_session_record",
]
