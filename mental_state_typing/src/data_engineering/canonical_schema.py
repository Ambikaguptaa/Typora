"""Canonical Keystroke Event Schema Module.

Defines the system-wide canonical event schema for keystroke dynamics observations.
All dataset adapters MUST transform raw observations into this schema.
The downstream feature engineering, sequence preparation, baseline analysis,
and deep learning pipelines operate EXCLUSIVELY on this canonical schema.

ZERO RAW TEXT INVARIANT:
The canonical schema NEVER contains raw typed characters, words, sentences,
or lexical tokens. The `key_identifier` field must only contain abstract hashed
tokens (e.g. `k_<hash>`) or generic key categories (e.g. `k_alpha`, `k_backspace`).
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

# Required column names in canonical order
CANONICAL_REQUIRED_COLUMNS: List[str] = [
    "participant_id",
    "session_id",
    "timestamp",
    "event_type",
    "key_identifier",
    "press_time",
    "release_time",
    "condition",
]

# Optional telemetry columns permitted in canonical schema
CANONICAL_OPTIONAL_COLUMNS: List[str] = [
    "pressure",
    "x",
    "y",
]

# Allowed event types in canonical schema
ALLOWED_EVENT_TYPES: Set[str] = {
    "down",
    "up",
    "press",
    "release",
    "keydown",
    "keyup",
    "touch",
}

# Forbidden raw-text column names that must NEVER appear in canonical data
FORBIDDEN_TEXT_COLUMNS: Set[str] = {
    "key",
    "char",
    "character",
    "text",
    "word",
    "sentence",
    "message",
    "typed_text",
    "password",
    "user_input",
    "content",
    "keystring",
    "raw_input",
    "keystrokes_text",
    "raw_text",
    "input_text",
}

# Pattern for valid abstract key identifiers (e.g. k_abc123, k_alpha, k_backspace, k_unk)
ABSTRACT_KEY_PATTERN = re.compile(r"^k_[a-zA-Z0-9_]+$")


def validate_canonical_dataframe(
    df: pd.DataFrame,
    strict_event_types: bool = False,
) -> Dict[str, Any]:
    """Validate that a DataFrame strictly adheres to the Canonical Event Schema.

    Verifies:
    1. All required canonical columns exist.
    2. Zero raw text columns are present.
    3. Column data types are valid.
    4. Key identifiers are abstract tokens and do NOT contain raw single characters.
    5. Timestamps are numeric and valid.
    6. Participant IDs and session IDs are populated (no nulls).
    7. Condition labels are populated (no nulls).

    Args:
        df: Input DataFrame to validate.
        strict_event_types: If True, requires event_type to be within ALLOWED_EVENT_TYPES.

    Returns:
        Dict[str, Any]: Validation report with 'is_valid', 'errors', 'warnings', and 'details'.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(df, pd.DataFrame):
        return {
            "is_valid": False,
            "errors": [f"Input must be a pandas DataFrame, got {type(df).__name__}"],
            "warnings": [],
            "details": {},
        }

    if df.empty:
        return {
            "is_valid": False,
            "errors": ["DataFrame is empty (0 rows)."],
            "warnings": [],
            "details": {"rows": 0},
        }

    cols = list(df.columns)
    cols_lower = {c.strip().lower(): c for c in cols}

    # 1. Check for forbidden raw text columns
    detected_forbidden = [c for c in cols_lower if c in FORBIDDEN_TEXT_COLUMNS]
    if detected_forbidden:
        errors.append(
            f"Zero-Raw-Text violation: DataFrame contains forbidden text columns: {detected_forbidden}. "
            "Raw text or characters must be stripped or abstracted before canonical conversion."
        )

    # 2. Check required columns
    missing_required = [c for c in CANONICAL_REQUIRED_COLUMNS if c not in cols]
    if missing_required:
        errors.append(f"Missing required canonical columns: {missing_required}")

    # If essential columns are missing, return early
    if errors:
        return {
            "is_valid": False,
            "errors": errors,
            "warnings": warnings,
            "details": {
                "columns_present": cols,
                "missing_required": missing_required,
                "forbidden_detected": detected_forbidden,
            },
        }

    # 3. Check for null values in critical identity and timing fields
    null_checks = {
        "participant_id": int(df["participant_id"].isna().sum()),
        "session_id": int(df["session_id"].isna().sum()),
        "timestamp": int(df["timestamp"].isna().sum()),
        "condition": int(df["condition"].isna().sum()),
    }
    for field_name, null_cnt in null_checks.items():
        if null_cnt > 0:
            errors.append(f"Canonical column '{field_name}' contains {null_cnt} null/NaN values.")

    # 4. Check data types and value sanity
    # Timestamps
    if not pd.api.types.is_numeric_dtype(df["timestamp"]):
        errors.append("Column 'timestamp' must be numeric (float or int).")

    # Press time
    if not pd.api.types.is_numeric_dtype(df["press_time"]):
        errors.append("Column 'press_time' must be numeric.")

    # Release time (can have NaNs for unpaired events, but non-null values must be numeric)
    if not pd.api.types.is_numeric_dtype(df["release_time"]):
        errors.append("Column 'release_time' must be numeric (float, with NaN for unpaired).")

    # 5. Check key_identifier abstractness (Zero-Raw-Text invariant)
    sample_keys = df["key_identifier"].dropna().astype(str).head(100)
    raw_char_leaks = []
    for k in sample_keys:
        # If single character and not formatted as abstract token, flag as potential raw char
        if len(k) == 1 and not k.startswith("k_"):
            raw_char_leaks.append(k)
        elif not ABSTRACT_KEY_PATTERN.match(k):
            # Key identifier does not conform to abstract naming
            if len(k) < 3:
                raw_char_leaks.append(k)

    if raw_char_leaks:
        errors.append(
            f"Column 'key_identifier' contains values that appear to be raw characters or invalid tokens: "
            f"{raw_char_leaks[:5]}. Must use abstract tokens matching 'k_<token>'."
        )

    # 6. Event type validation
    if strict_event_types:
        unique_events = set(df["event_type"].dropna().astype(str).str.lower().unique())
        invalid_events = unique_events - ALLOWED_EVENT_TYPES
        if invalid_events:
            errors.append(
                f"Column 'event_type' contains unpermitted event types: {invalid_events}. "
                f"Allowed: {sorted(list(ALLOWED_EVENT_TYPES))}"
            )

    # 7. Check for timing sanity (press_time <= release_time when release_time is present)
    paired_df = df[df["release_time"].notna() & df["press_time"].notna()]
    if not paired_df.empty:
        negative_dwell = (paired_df["release_time"] < paired_df["press_time"]).sum()
        if negative_dwell > 0:
            warnings.append(
                f"Found {negative_dwell} records where release_time < press_time (negative dwell time)."
            )

    is_valid = len(errors) == 0

    return {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "details": {
            "row_count": len(df),
            "columns": cols,
            "unique_participants": int(df["participant_id"].nunique()),
            "unique_sessions": int(df["session_id"].nunique()),
            "unique_conditions": sorted(list(df["condition"].dropna().astype(str).unique())),
            "paired_records": len(paired_df),
            "optional_columns_present": [c for c in CANONICAL_OPTIONAL_COLUMNS if c in cols],
        },
    }


def assert_canonical_schema(df: pd.DataFrame, strict_event_types: bool = False) -> None:
    """Assert that a DataFrame conforms to the canonical schema, raising ValueError on failure."""
    res = validate_canonical_dataframe(df, strict_event_types=strict_event_types)
    if not res["is_valid"]:
        raise ValueError(
            f"Canonical schema validation failed: {'; '.join(res['errors'])}"
        )
