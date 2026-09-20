"""Encryption and security helper module.

Contains cryptographic helpers for securing local data artifacts at rest.
Foundation stub: complex crypto suites will be introduced if required in later phases.
"""

import base64
import hashlib
from typing import Optional


def derive_key(secret_passphrase: str, salt: bytes) -> bytes:
    """Derive a fixed-length cryptographic key from a passphrase.

    Args:
        secret_passphrase: User or system secret string.
        salt: Salt bytes.

    Returns:
        bytes: Derived 32-byte key.
    """
    return hashlib.pbkdf2_hmac(
        "sha256",
        secret_passphrase.encode("utf-8"),
        salt,
        iterations=100_000,
        dklen=32,
    )


def simple_obfuscate(text: str) -> str:
    """Simple reversible base64 encoding helper for non-sensitive config values.

    Note: This is obfuscation, not high-grade encryption.
    """
    return base64.b64encode(text.encode("utf-8")).decode("utf-8")


def simple_deobfuscate(encoded_text: str) -> str:
    """Reverse simple base64 obfuscation."""
    return base64.b64decode(encoded_text.encode("utf-8")).decode("utf-8")
