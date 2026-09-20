"""Pseudonymization module for user identifiers.

Provides keyed HMAC-SHA256 cryptographic hashing to replace participant IDs,
usernames, student IDs, or IP addresses with irreversible pseudo-identifiers.
"""

import hashlib
import hmac
import re
from typing import Optional
import pandas as pd

from src.config.settings import settings

PSEUDONYM_PATTERN = re.compile(r"^usr_[0-9a-f]{16}$")


def pseudonymize_user_id(
    raw_user_id: str,
    secret: Optional[str] = None,
    salt: Optional[str] = None,
) -> str:
    """Generate a pseudonymized hash identifier for a user or participant using HMAC-SHA256.

    Args:
        raw_user_id: Plaintext user identifier (e.g. email, student ID, username).
        secret: Optional secret key for HMAC. Defaults to settings.pseudonymization_secret.
        salt: Deprecated alias for secret to maintain backward compatibility.

    Returns:
        str: A deterministic, irreversible pseudonym formatted as 'usr_<16-char-hash>'.

    Raises:
        ValueError: If raw_user_id or secret is empty/invalid.
    """
    if raw_user_id is None or not str(raw_user_id).strip():
        raise ValueError("raw_user_id cannot be empty or blank")

    effective_secret = secret if secret is not None else salt
    if effective_secret is None:
        effective_secret = settings.pseudonymization_secret

    if not effective_secret or not effective_secret.strip():
        raise ValueError("Pseudonymization secret/salt cannot be empty")

    key_bytes = effective_secret.encode("utf-8")
    msg_bytes = str(raw_user_id).strip().encode("utf-8")

    digest = hmac.new(key_bytes, msg_bytes, hashlib.sha256).hexdigest()

    # Form compact, uniform identifier
    return f"usr_{digest[:16]}"


def is_valid_pseudonym(identifier: str) -> bool:
    """Verify if a string matches the expected pseudonym format 'usr_<16-hex>'.

    Args:
        identifier: String to validate.

    Returns:
        bool: True if valid format, False otherwise.
    """
    if not isinstance(identifier, str):
        return False
    return bool(PSEUDONYM_PATTERN.match(identifier))


def pseudonymize_series(
    series: pd.Series,
    secret: Optional[str] = None,
) -> pd.Series:
    """Pseudonymize an entire pandas Series of participant identifiers.

    Args:
        series: Pandas Series of plaintext participant identifiers.
        secret: Optional HMAC secret key.

    Returns:
        pd.Series: Series with pseudonymized identifiers.
    """
    return series.astype(str).apply(lambda uid: pseudonymize_user_id(uid, secret=secret))
