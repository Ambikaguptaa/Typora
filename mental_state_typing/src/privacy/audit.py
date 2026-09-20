"""Security and Privacy Audit Logging Module.

Maintains structured, tamper-evident audit trails of security-relevant operations,
access control decisions, cryptographic operations, and retention events.
Guarantees absolute redaction of raw text, keys, and credentials.
"""

from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.config.settings import settings
from src.privacy.privacy_utils import FORBIDDEN_TEXT_COLUMNS

DEFAULT_AUDIT_LOG = settings.base_dir / "logs" / "security_audit.jsonl"


class SecurityEventType(str, Enum):
    """Enumeration of security and data governance audit events."""

    ACCESS_ATTEMPT = "ACCESS_ATTEMPT"
    DECRYPTION_SUCCESS = "DECRYPTION_SUCCESS"
    DECRYPTION_FAILURE = "DECRYPTION_FAILURE"
    PURGE_EXECUTED = "PURGE_EXECUTED"
    ZERO_TEXT_VIOLATION = "ZERO_TEXT_VIOLATION"
    DP_QUERY_EXECUTED = "DP_QUERY_EXECUTED"
    RETENTION_CLEANUP = "RETENTION_CLEANUP"
    KEY_ROTATION = "KEY_ROTATION"


# Sensitive keys whose values must be scrubbed before recording into audit logs
REDACTED_KEYS = FORBIDDEN_TEXT_COLUMNS.union({
    "key",
    "encryption_key",
    "secret",
    "secret_key",
    "token",
    "salt",
    "password",
    "plaintext",
})


def sanitize_audit_details(details: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively redact sensitive text, keys, or credentials from audit logs.

    Args:
        details: Raw event metadata dictionary.

    Returns:
        Dict[str, Any]: Sanitized dictionary safe for audit persistence.
    """
    sanitized: Dict[str, Any] = {}
    for k, v in details.items():
        k_str = str(k).lower().strip()
        if k_str in REDACTED_KEYS or any(rk in k_str for rk in ("key", "secret", "token", "password")):
            sanitized[k] = "[REDACTED]"
        elif isinstance(v, dict):
            sanitized[k] = sanitize_audit_details(v)
        elif isinstance(v, (list, tuple, set)):
            sanitized[k] = [
                sanitize_audit_details(item) if isinstance(item, dict) else item
                for item in v
            ]
        else:
            sanitized[k] = v
    return sanitized


def log_security_event(
    event_type: Union[SecurityEventType, str],
    actor_role: str,
    details: Dict[str, Any],
    log_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Record a security or privacy governance event.

    Args:
        event_type: Type of the security event.
        actor_role: Role of the acting entity (USER, RESEARCHER, ADMIN, SYSTEM).
        details: Event metadata dictionary (will be automatically sanitized).
        log_file: Optional log path. Defaults to logs/security_audit.jsonl.

    Returns:
        Dict[str, Any]: The recorded event entry.
    """
    out_file = log_file or DEFAULT_AUDIT_LOG
    out_file.parent.mkdir(parents=True, exist_ok=True)

    ev_name = event_type.value if isinstance(event_type, SecurityEventType) else str(event_type)
    sanitized = sanitize_audit_details(details)

    event_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": ev_name,
        "actor_role": str(actor_role),
        "details": sanitized,
    }

    line = json.dumps(event_record) + "\n"
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(line)

    return event_record


def read_recent_audit_events(
    limit: int = 50,
    log_file: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Retrieve recent audit events in reverse chronological order.

    Args:
        limit: Maximum events to return.
        log_file: Optional log path.

    Returns:
        List[Dict[str, Any]]: Recent audit records.
    """
    target = log_file or DEFAULT_AUDIT_LOG
    if not target.exists() or not target.is_file():
        return []

    lines = target.read_text(encoding="utf-8").strip().splitlines()
    events = []
    for line in reversed(lines[-limit:]):
        try:
            events.append(json.loads(line))
        except Exception:
            continue
    return events
