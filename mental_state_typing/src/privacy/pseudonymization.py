"""Pseudonymization module for user identifiers.

Provides salted SHA-256 cryptographic hashing to replace participant IDs,
usernames, or IP addresses with an irreversible pseudo-identifier.
"""

import hashlib
from typing import Optional

from src.config.settings import settings


def pseudonymize_user_id(
    raw_user_id: str,
    salt: Optional[str] = None,
) -> str:
    """Generate a pseudonymized hash identifier for a user or participant.

    Args:
        raw_user_id: Plaintext user identifier (e.g. email, student ID, username).
        salt: Optional cryptographic salt. Defaults to PSEUDONYMIZATION_SALT from settings.

    Returns:
        str: A deterministic, irreversible pseudonym formatted as 'usr_<16-char-hash>'.
    """
    if not raw_user_id:
        raise ValueError("raw_user_id cannot be empty")

    effective_salt = salt if salt is not None else settings.pseudonymization_salt
    salted_input = f"{effective_salt}:{raw_user_id}".encode("utf-8")
    hash_digest = hashlib.sha256(salted_input).hexdigest()

    # Truncate to 16 hex characters with prefix for compact readability
    return f"usr_{hash_digest[:16]}"
