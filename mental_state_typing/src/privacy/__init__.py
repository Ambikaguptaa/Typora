"""Privacy and security package for participant data protection.

Ensures no raw keystroke text is recorded and user identifiers are pseudonymized.
"""

from src.privacy.pseudonymization import pseudonymize_user_id
from src.privacy.privacy_utils import is_safe_metadata_only, strip_character_data

__all__ = [
    "pseudonymize_user_id",
    "is_safe_metadata_only",
    "strip_character_data",
]
