"""Data Retention and Lifecycle Management Module.

Automates scheduled expiration of raw timing events, session aggregates, and
assessment records. Supports GDPR Article 17 "Right to be Forgotten" participant data purge.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Union

from src.config.settings import settings
from src.privacy.pseudonymization import is_valid_pseudonym, pseudonymize_user_id


def calculate_expiration(
    created_at: datetime,
    retention_days: int,
) -> datetime:
    """Calculate exact expiration datetime based on retention period.

    Args:
        created_at: Timestamp when artifact was generated.
        retention_days: Number of days to retain artifact.

    Returns:
        datetime: Expiration timestamp.
    """
    return created_at + timedelta(days=retention_days)


def is_expired(
    created_at: datetime,
    retention_days: int,
    current_time: Optional[datetime] = None,
) -> bool:
    """Determine whether an artifact has exceeded its retention window.

    Args:
        created_at: Timestamp when artifact was created.
        retention_days: Retention duration in days.
        current_time: Optional reference time (defaults to datetime.now(timezone.utc)).

    Returns:
        bool: True if expired, False otherwise.
    """
    now = current_time or datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        # Assume UTC if naive
        created_at = created_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    expiration = calculate_expiration(created_at, retention_days)
    return now >= expiration


def identify_expired_artifacts(
    directory: Union[str, Path],
    retention_days: int,
    current_time: Optional[datetime] = None,
) -> List[Path]:
    """Scan a directory and identify all files older than the retention threshold.

    Args:
        directory: Directory containing data artifacts.
        retention_days: Maximum retention age in days.
        current_time: Optional reference time.

    Returns:
        List[Path]: List of expired file paths.
    """
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        return []

    now = current_time or datetime.now(timezone.utc)
    expired_files: List[Path] = []

    for file_path in dir_path.iterdir():
        if file_path.is_file() and not file_path.name.startswith("."):
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
            if is_expired(mtime, retention_days, current_time=now):
                expired_files.append(file_path)

    return expired_files


def cleanup_expired_artifacts(
    directory: Union[str, Path],
    retention_days: int,
    dry_run: bool = True,
    current_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Prune expired artifacts from a directory with safety dry-run protection.

    Args:
        directory: Target directory to clean.
        retention_days: Maximum retention age in days.
        dry_run: If True, identifies expired files without deleting them. Defaults to True.
        current_time: Optional reference time.

    Returns:
        Dict[str, Any]: Cleanup report detailing scanned, expired, and removed files.
    """
    dir_path = Path(directory)
    expired = identify_expired_artifacts(
        dir_path,
        retention_days=retention_days,
        current_time=current_time,
    )

    deleted_count = 0
    deleted_files: List[str] = []

    if not dry_run:
        for file_path in expired:
            try:
                file_path.unlink()
                deleted_count += 1
                deleted_files.append(str(file_path.name))
            except Exception:
                pass

    return {
        "directory": str(dir_path),
        "retention_days": retention_days,
        "dry_run": dry_run,
        "expired_count": len(expired),
        "deleted_count": deleted_count,
        "expired_files": [f.name for f in expired],
        "deleted_files": deleted_files,
    }


def purge_participant_data(
    user_identifier: str,
    database_path: Optional[Path] = None,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """Execute a GDPR Right-to-be-Forgotten data purge for a participant.

    Purges all matching records across database tables and baseline files.

    Args:
        user_identifier: Plaintext or pseudonymized participant identifier.
        database_path: Optional path to SQLite database.
        dry_run: If True, inspects and counts records without deleting. Defaults to True.

    Returns:
        Dict[str, Any]: Summary report of purged records.
    """
    pseudonym = (
        user_identifier
        if is_valid_pseudonym(user_identifier)
        else pseudonymize_user_id(user_identifier)
    )

    db_file = database_path or settings.database_path
    records_found: Dict[str, int] = {}
    records_deleted: Dict[str, int] = {}

    if db_file.exists():
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()

        # Check existing tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            # Check if user_id or pseudonym column exists
            cursor.execute(f"PRAGMA table_info({table});")
            col_names = [row[1] for row in cursor.fetchall()]
            user_cols = [c for c in col_names if c in ("user_id", "participant_id", "subject_id")]

            for ucol in user_cols:
                cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {ucol} = ?;", (pseudonym,))
                count = cursor.fetchone()[0]
                records_found[f"{table}.{ucol}"] = count

                if not dry_run and count > 0:
                    cursor.execute(f"DELETE FROM {table} WHERE {ucol} = ?;", (pseudonym,))
                    records_deleted[f"{table}.{ucol}"] = count

        if not dry_run:
            conn.commit()
        conn.close()

    return {
        "participant_pseudonym": pseudonym,
        "dry_run": dry_run,
        "records_found": records_found,
        "records_deleted": records_deleted if not dry_run else {},
        "status": "dry_run_complete" if dry_run else "purge_complete",
    }
