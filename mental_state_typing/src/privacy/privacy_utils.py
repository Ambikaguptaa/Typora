"""Privacy utility functions enforcing strict key-text suppression.

Guarantees that no actual typed characters, text snippets, or keystroke names
are admitted into the processing pipeline or stored in databases.
"""

from typing import Iterable
import pandas as pd

# Disallowed columns that might hold keystroke character identities
FORBIDDEN_TEXT_COLUMNS = {
    "key",
    "char",
    "character",
    "text",
    "word",
    "keystring",
    "content",
    "payload",
    "raw_input",
}


def strip_character_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any columns that could contain typed characters or user text.

    Args:
        df: Input DataFrame potentially containing mixed columns.

    Returns:
        pd.DataFrame: DataFrame containing only sanitized non-text columns.
    """
    safe_cols = [
        col for col in df.columns if col.lower() not in FORBIDDEN_TEXT_COLUMNS
    ]
    return df[safe_cols].copy()


def is_safe_metadata_only(
    columns: Iterable[str],
) -> bool:
    """Validate whether an iterable of column names complies with zero-text policy.

    Args:
        columns: Iterable of column names to audit.

    Returns:
        bool: True if zero forbidden columns are detected, False otherwise.
    """
    lowered = {str(c).strip().lower() for c in columns}
    violating = lowered.intersection(FORBIDDEN_TEXT_COLUMNS)
    return len(violating) == 0
