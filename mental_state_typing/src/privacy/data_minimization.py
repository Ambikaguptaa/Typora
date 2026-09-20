"""Data Minimization Module.

Implements data minimization principles by enforcing the zero-raw-text policy,
stripping unnecessary or forbidden columns, and ensuring participant identifiers
are pseudonymized prior to processing or storage.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from src.privacy.privacy_utils import FORBIDDEN_TEXT_COLUMNS, strip_character_data
from src.privacy.pseudonymization import is_valid_pseudonym, pseudonymize_user_id

USER_ID_COLUMNS = {
    "user_id",
    "subject_id",
    "participant_id",
    "student_id",
    "raw_user_id",
}


def minimize_keystroke_dataframe(
    df: pd.DataFrame,
    secret: Optional[str] = None,
    allowed_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Sanitize and minimize a keystroke dataframe for research and processing.

    Drops all raw text columns, pseudonymizes plaintext participant identifiers,
    and optionally filters to only explicitly allowed feature columns.

    Args:
        df: Input raw or semi-processed DataFrame.
        secret: Optional secret key for HMAC pseudonymization.
        allowed_columns: Optional whitelist of columns to retain.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]:
            - Sanitized DataFrame.
            - Audit report dictionary detailing dropped, retained, and pseudonymized columns.
    """
    if df.empty:
        return df.copy(), {
            "original_columns": list(df.columns),
            "retained_columns": list(df.columns),
            "dropped_columns": [],
            "pseudonymized_columns": [],
            "row_count": 0,
            "zero_text_compliant": True,
        }

    original_cols = list(df.columns)
    minimized_df = df.copy()

    # 1. Drop forbidden character/text columns
    minimized_df = strip_character_data(minimized_df)
    dropped_cols = [c for c in original_cols if c not in minimized_df.columns]

    # 2. Pseudonymize participant identifiers if present
    pseudonymized_cols = []
    for col in minimized_df.columns:
        if col.lower() in USER_ID_COLUMNS:
            pseudonymized_cols.append(col)
            minimized_df[col] = minimized_df[col].astype(str).apply(
                lambda val: val if is_valid_pseudonym(val) else pseudonymize_user_id(val, secret=secret)
            )

    # 3. Apply optional whitelist
    if allowed_columns is not None:
        keep = [c for c in minimized_df.columns if c in allowed_columns]
        extra_dropped = [c for c in minimized_df.columns if c not in keep]
        dropped_cols.extend(extra_dropped)
        minimized_df = minimized_df[keep].copy()

    retained_cols = list(minimized_df.columns)

    report = {
        "original_columns": original_cols,
        "retained_columns": retained_cols,
        "dropped_columns": dropped_cols,
        "pseudonymized_columns": pseudonymized_cols,
        "row_count": len(minimized_df),
        "zero_text_compliant": not any(
            c.lower() in FORBIDDEN_TEXT_COLUMNS for c in retained_cols
        ),
    }

    return minimized_df, report


def enforce_data_minimization(
    df: pd.DataFrame,
    allowed_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Convenience wrapper returning only the sanitized DataFrame.

    Args:
        df: Input DataFrame.
        allowed_features: Optional whitelist of feature names.

    Returns:
        pd.DataFrame: Sanitized DataFrame.
    """
    sanitized_df, _ = minimize_keystroke_dataframe(
        df,
        allowed_columns=allowed_features,
    )
    return sanitized_df
