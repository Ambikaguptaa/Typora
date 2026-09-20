"""Data Sensitivity Classification Module.

Defines sensitivity levels and classification rules for keystroke behavioral data,
identifiers, and model outputs according to academic privacy governance standards.
"""

from enum import Enum
from typing import Dict, Iterable, Optional


class DataSensitivityLevel(str, Enum):
    """Classification levels for data sensitivity."""

    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    SENSITIVE = "SENSITIVE"
    HIGHLY_SENSITIVE = "HIGHLY_SENSITIVE"


# Prohibited content keywords that constitute highly sensitive private data
HIGHLY_SENSITIVE_KEYWORDS = {
    "key",
    "char",
    "character",
    "text",
    "word",
    "keystring",
    "content",
    "payload",
    "password",
    "raw_input",
    "secret",
    "token",
    "credential",
    "email",
    "student_id",
    "ssn",
    "real_name",
    "raw_user_id",
    "user_id",
    "subject_id",
    "participant_id",
}

# Behavioral timing features and ML inferences considered sensitive
SENSITIVE_KEYWORDS = {
    "dwell",
    "flight",
    "hold",
    "pause",
    "speed",
    "wpm",
    "cpm",
    "backspace",
    "error",
    "deviation",
    "strain",
    "entropy",
    "margin",
    "confidence",
    "baseline",
    "press_time",
    "release_time",
    "iqr",
    "mad",
    "tdi",
}

# Operational metadata considered internal
INTERNAL_KEYWORDS = {
    "session_id",
    "session_index",
    "event_index",
    "record_count",
    "feature_count",
    "version",
    "status",
    "created_at",
    "updated_at",
    "timestamp",
    "state",
    "window_size",
    "sequence_length",
}


def classify_field(field_name: str) -> DataSensitivityLevel:
    """Classify a field or column name into a DataSensitivityLevel.

    Args:
        field_name: The column or variable name to classify.

    Returns:
        DataSensitivityLevel: Assigned sensitivity classification.
    """
    cleaned = field_name.strip().lower()

    # Exact or keyword matches for highly sensitive attributes
    for kw in HIGHLY_SENSITIVE_KEYWORDS:
        if kw == cleaned or f"_{kw}" in cleaned or f"{kw}_" in cleaned:
            return DataSensitivityLevel.HIGHLY_SENSITIVE

    # Timing metrics and inferences
    for kw in SENSITIVE_KEYWORDS:
        if kw in cleaned:
            return DataSensitivityLevel.SENSITIVE

    # Internal operational flags
    for kw in INTERNAL_KEYWORDS:
        if kw in cleaned:
            return DataSensitivityLevel.INTERNAL

    # Default fallback for unspecified columns
    return DataSensitivityLevel.INTERNAL


def classify_schema(columns: Iterable[str]) -> Dict[str, DataSensitivityLevel]:
    """Classify an entire schema of columns.

    Args:
        columns: Iterable of column names.

    Returns:
        Dict[str, DataSensitivityLevel]: Mapping of column name to sensitivity level.
    """
    return {col: classify_field(col) for col in columns}


def is_storage_permitted(
    level: DataSensitivityLevel,
    is_raw_text: bool = False,
) -> bool:
    """Check if storage is permitted under the zero-text data governance policy.

    Args:
        level: DataSensitivityLevel of the artifact.
        is_raw_text: Flag indicating if the data represents unpseudonymized raw text.

    Returns:
        bool: False if raw typed characters or forbidden text, True otherwise.
    """
    if is_raw_text:
        return False
    return True
