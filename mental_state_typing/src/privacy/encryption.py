"""Encryption and security helper module.

Provides authenticated symmetric encryption at rest using Fernet (AES-128-CBC + HMAC-SHA256).
Enables secure storage of sensitive baselines, model artifacts, and assessment reports.
"""

import base64
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Union

try:
    from cryptography.fernet import Fernet, InvalidToken
    _CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    _CRYPTOGRAPHY_AVAILABLE = False
    Fernet = None
    InvalidToken = Exception

from src.config.settings import settings


class DecryptionError(Exception):
    """Raised when ciphertext decryption fails due to invalid key or tampered payload."""
    pass


def _ensure_cryptography() -> None:
    """Raise ImportError if cryptography package is not installed."""
    if not _CRYPTOGRAPHY_AVAILABLE:
        raise ImportError(
            "The 'cryptography' package is required for encryption operations. "
            "Install it with: pip install cryptography"
        )


def generate_encryption_key() -> str:
    """Generate a new url-safe base64-encoded 32-byte Fernet key.

    Returns:
        str: Fernet key as a string.
    """
    _ensure_cryptography()
    return Fernet.generate_key().decode("utf-8")


def get_fernet(key: Optional[Union[str, bytes]] = None):
    """Instantiate a Fernet cryptographic engine.

    Args:
        key: Optional Fernet key string or bytes. Defaults to settings.encryption_key.

    Returns:
        Fernet: Configured Fernet cipher.

    Raises:
        ValueError: If no key is provided and none is configured in settings.
    """
    _ensure_cryptography()
    effective_key = key if key is not None else settings.encryption_key

    if not effective_key:
        raise ValueError(
            "No encryption key provided or configured in settings. "
            "Set ENCRYPTION_KEY environment variable or pass a valid key."
        )

    if isinstance(effective_key, str):
        key_bytes = effective_key.strip().encode("utf-8")
    else:
        key_bytes = effective_key

    try:
        return Fernet(key_bytes)
    except Exception as e:
        raise ValueError(f"Invalid Fernet encryption key format: {e}") from e


def encrypt_bytes(
    data: bytes,
    key: Optional[Union[str, bytes]] = None,
) -> bytes:
    """Encrypt raw bytes using authenticated symmetric encryption.

    Args:
        data: Plaintext bytes.
        key: Optional Fernet encryption key.

    Returns:
        bytes: Authenticated ciphertext bytes.
    """
    cipher = get_fernet(key)
    return cipher.encrypt(data)


def decrypt_bytes(
    ciphertext: bytes,
    key: Optional[Union[str, bytes]] = None,
) -> bytes:
    """Decrypt authenticated ciphertext bytes.

    Args:
        ciphertext: Encrypted Fernet ciphertext bytes.
        key: Optional Fernet encryption key.

    Returns:
        bytes: Decrypted plaintext bytes.

    Raises:
        DecryptionError: If the ciphertext has been tampered with or the key is invalid.
    """
    cipher = get_fernet(key)
    try:
        return cipher.decrypt(ciphertext)
    except (InvalidToken, Exception) as e:
        raise DecryptionError(
            "Failed to decrypt: invalid encryption key or corrupted/tampered ciphertext."
        ) from e


def encrypt_json(
    obj: Any,
    key: Optional[Union[str, bytes]] = None,
) -> bytes:
    """Serialize a Python object to JSON and encrypt into ciphertext.

    Args:
        obj: JSON-serializable object (dict, list, etc.).
        key: Optional Fernet encryption key.

    Returns:
        bytes: Encrypted ciphertext bytes.
    """
    payload = json.dumps(obj, default=str).encode("utf-8")
    return encrypt_bytes(payload, key=key)


def decrypt_json(
    ciphertext: bytes,
    key: Optional[Union[str, bytes]] = None,
) -> Any:
    """Decrypt ciphertext and parse into a Python object from JSON.

    Args:
        ciphertext: Encrypted Fernet ciphertext bytes.
        key: Optional Fernet encryption key.

    Returns:
        Any: Decoded Python object.

    Raises:
        DecryptionError: If decryption or JSON decoding fails.
    """
    raw_bytes = decrypt_bytes(ciphertext, key=key)
    try:
        return json.loads(raw_bytes.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise DecryptionError(f"Decrypted payload is not valid JSON: {e}") from e


def encrypt_file(
    source_path: Union[str, Path],
    dest_path: Union[str, Path],
    key: Optional[Union[str, bytes]] = None,
) -> Path:
    """Encrypt a local file to a destination path.

    Args:
        source_path: Path to source plaintext file.
        dest_path: Destination path for ciphertext file.
        key: Optional Fernet key.

    Returns:
        Path: Destination path.
    """
    src = Path(source_path)
    dst = Path(dest_path)

    plaintext = src.read_bytes()
    ciphertext = encrypt_bytes(plaintext, key=key)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(ciphertext)
    return dst


def decrypt_file(
    source_path: Union[str, Path],
    dest_path: Union[str, Path],
    key: Optional[Union[str, bytes]] = None,
) -> Path:
    """Decrypt a local ciphertext file to a destination path.

    Args:
        source_path: Path to ciphertext file.
        dest_path: Destination path for plaintext file.
        key: Optional Fernet key.

    Returns:
        Path: Destination path.

    Raises:
        DecryptionError: If decryption fails.
    """
    src = Path(source_path)
    dst = Path(dest_path)

    ciphertext = src.read_bytes()
    plaintext = decrypt_bytes(ciphertext, key=key)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(plaintext)
    return dst


# --- Legacy / Utility functions for backward compatibility ---

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
    """Simple reversible base64 encoding helper for non-sensitive config values."""
    return base64.b64encode(text.encode("utf-8")).decode("utf-8")


def simple_deobfuscate(encoded_text: str) -> str:
    """Reverse simple base64 obfuscation."""
    return base64.b64decode(encoded_text.encode("utf-8")).decode("utf-8")
