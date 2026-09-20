"""Privacy and security package for participant data protection.

Enforces zero-raw-text invariant, HMAC-SHA256 pseudonymization, Fernet authenticated encryption,
storage policies, role-based access control, Laplace differential privacy, and data lifecycle management.
"""

from src.privacy.access_control import (
    AccessController,
    Permission,
    PermissionDeniedError,
    Role,
)
from src.privacy.audit import (
    SecurityEventType,
    log_security_event,
    read_recent_audit_events,
)
from src.privacy.data_classification import (
    DataSensitivityLevel,
    classify_field,
    classify_schema,
    is_storage_permitted,
)
from src.privacy.data_minimization import (
    enforce_data_minimization,
    minimize_keystroke_dataframe,
)
from src.privacy.differential_privacy import (
    compute_privacy_budget,
    laplace_noise,
    private_count,
    private_mean,
)
from src.privacy.encryption import (
    DecryptionError,
    decrypt_bytes,
    decrypt_file,
    decrypt_json,
    encrypt_bytes,
    encrypt_file,
    encrypt_json,
    generate_encryption_key,
    get_fernet,
)
from src.privacy.privacy_utils import (
    FORBIDDEN_TEXT_COLUMNS,
    assert_zero_raw_text,
    audit_zero_raw_text,
    is_safe_metadata_only,
    strip_character_data,
)
from src.privacy.pseudonymization import (
    is_valid_pseudonym,
    pseudonymize_series,
    pseudonymize_user_id,
)
from src.privacy.retention import (
    calculate_expiration,
    cleanup_expired_artifacts,
    identify_expired_artifacts,
    is_expired,
    purge_participant_data,
)
from src.privacy.storage_policy import (
    StoragePolicy,
    get_storage_policy,
    validate_artifact_storage_compliance,
)

__all__ = [
    # Data classification
    "DataSensitivityLevel",
    "classify_field",
    "classify_schema",
    "is_storage_permitted",
    # Data minimization
    "minimize_keystroke_dataframe",
    "enforce_data_minimization",
    # Pseudonymization
    "pseudonymize_user_id",
    "pseudonymize_series",
    "is_valid_pseudonym",
    # Encryption
    "generate_encryption_key",
    "get_fernet",
    "encrypt_bytes",
    "decrypt_bytes",
    "encrypt_json",
    "decrypt_json",
    "encrypt_file",
    "decrypt_file",
    "DecryptionError",
    # Storage Policy
    "StoragePolicy",
    "get_storage_policy",
    "validate_artifact_storage_compliance",
    # Access Control
    "Role",
    "Permission",
    "AccessController",
    "PermissionDeniedError",
    # Privacy Utils
    "FORBIDDEN_TEXT_COLUMNS",
    "strip_character_data",
    "is_safe_metadata_only",
    "audit_zero_raw_text",
    "assert_zero_raw_text",
    # Differential Privacy
    "private_mean",
    "private_count",
    "compute_privacy_budget",
    "laplace_noise",
    # Retention
    "calculate_expiration",
    "is_expired",
    "identify_expired_artifacts",
    "cleanup_expired_artifacts",
    "purge_participant_data",
    # Audit
    "SecurityEventType",
    "log_security_event",
    "read_recent_audit_events",
]
