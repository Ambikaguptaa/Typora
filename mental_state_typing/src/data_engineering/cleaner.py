"""Data cleaner module for keystroke timing data.

Filters out noise, missing values, physiological anomalies, and extreme outliers.
Provides traceable cleaning reports documenting every dropped or adjusted record.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.data_engineering.dataset_adapter import detect_sensitive_text_columns


def clean_keystroke_timing(
    df: pd.DataFrame,
    min_hold_ms: float = 10.0,
    max_hold_ms: float = 2000.0,
    max_flight_ms: float = 5000.0,
) -> pd.DataFrame:
    """Clean and filter keystroke timing records (legacy compatibility function).

    Args:
        df: Input DataFrame containing timing metrics.
        min_hold_ms: Minimum plausible hold time in ms.
        max_hold_ms: Maximum plausible hold time in ms.
        max_flight_ms: Maximum plausible flight time in ms.

    Returns:
        pd.DataFrame: Cleaned DataFrame with invalid entries removed.
    """
    cleaned = df.copy()

    # Drop records with missing values
    cleaned = cleaned.dropna()

    # If hold_time is present, filter within physiological bounds
    if "hold_time" in cleaned.columns:
        cleaned = cleaned[
            (cleaned["hold_time"] >= min_hold_ms)
            & (cleaned["hold_time"] <= max_hold_ms)
        ]

    # If flight_time is present, remove negative or excessively long pauses
    if "flight_time" in cleaned.columns:
        cleaned = cleaned[
            (cleaned["flight_time"] >= 0.0)
            & (cleaned["flight_time"] <= max_flight_ms)
        ]

    return cleaned.reset_index(drop=True)


def clean_dataset_with_report(
    df: pd.DataFrame,
    min_dwell_ms: float = 10.0,
    max_dwell_ms: float = 4000.0,
    max_flight_ms: float = 10000.0,
    user_col: Optional[str] = None,
    session_col: Optional[str] = None,
    timestamp_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean keystroke dataset with complete traceability and diagnostic reporting.

    Enforces:
    1. Zero-text data minimization: Drops sensitive text/message content immediately.
    2. Missing value auditing.
    3. Timestamp validity (non-null, positive).
    4. Physiological duration limits (dwell and flight intervals).
    5. Duplicate removal.
    6. Chronological ordering per user/session.

    Args:
        df: Input DataFrame.
        min_dwell_ms: Minimum plausible key hold in ms (physiological threshold).
        max_dwell_ms: Maximum plausible key hold in ms (distraction/sleep threshold).
        max_flight_ms: Maximum plausible inter-key interval in ms.
        user_col: User ID column name.
        session_col: Session ID column name.
        timestamp_col: Timestamp column name.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned DataFrame and traceability report.
    """
    input_rows = len(df)
    cleaned = df.copy()

    # 1. Zero-Text Data Minimization: drop any message / sentence / text content
    sensitive_cols = detect_sensitive_text_columns(cleaned)
    if sensitive_cols:
        cleaned = cleaned.drop(columns=sensitive_cols, errors="ignore")

    # 2. Duplicate Detection
    dup_mask = cleaned.duplicated(keep="first")
    duplicates_removed = int(dup_mask.sum())
    cleaned = cleaned[~dup_mask]

    # 3. Timestamp Validity Check
    invalid_timestamps = 0
    if timestamp_col and timestamp_col in cleaned.columns:
        # Check non-null and positive
        ts_valid_mask = cleaned[timestamp_col].notnull() & (cleaned[timestamp_col] >= 0)
        invalid_timestamps = int((~ts_valid_mask).sum())
        cleaned = cleaned[ts_valid_mask]

    # 4. Dwell Time Physiological Filtering
    negative_dwell = 0
    extreme_dwell = 0
    dwell_candidates = [c for c in ["dwell_time", "hold_time"] if c in cleaned.columns]

    for col in dwell_candidates:
        dwell_num = pd.to_numeric(cleaned[col], errors="coerce")
        neg_mask = dwell_num < min_dwell_ms
        ext_mask = dwell_num > max_dwell_ms
        negative_dwell += int(neg_mask.sum())
        extreme_dwell += int(ext_mask.sum())
        cleaned = cleaned[~neg_mask & ~ext_mask]

    # If press_time and release_time exist, check release >= press
    if "press_time" in cleaned.columns and "release_time" in cleaned.columns:
        pt = pd.to_numeric(cleaned["press_time"], errors="coerce")
        rt = pd.to_numeric(cleaned["release_time"], errors="coerce")
        computed_dwell = rt - pt
        bad_computed = (computed_dwell < min_dwell_ms) | (computed_dwell > max_dwell_ms)
        negative_dwell += int(bad_computed.sum())
        cleaned = cleaned[~bad_computed]

    # 5. Flight Time Filtering
    negative_flight = 0
    extreme_flight = 0
    flight_candidates = [c for c in ["flight_time", "iki"] if c in cleaned.columns]

    for col in flight_candidates:
        flight_num = pd.to_numeric(cleaned[col], errors="coerce")
        # Negative flight from slight rollover typing can be clamped to 0.0 or dropped if < -200ms
        extreme_neg = flight_num < -200.0
        ext_flight = flight_num > max_flight_ms
        negative_flight += int(extreme_neg.sum())
        extreme_flight += int(ext_flight.sum())
        cleaned = cleaned[~extreme_neg & ~ext_flight]
        # Clamp mild rollover negative flight times to 0.0
        cleaned[col] = cleaned[col].clip(lower=0.0)

    # 6. Chronological Sorting per User and Session
    sort_cols: List[str] = []
    if user_col and user_col in cleaned.columns:
        sort_cols.append(user_col)
    if session_col and session_col in cleaned.columns:
        sort_cols.append(session_col)
    if timestamp_col and timestamp_col in cleaned.columns:
        sort_cols.append(timestamp_col)
    elif "press_time" in cleaned.columns:
        sort_cols.append("press_time")

    if sort_cols:
        cleaned = cleaned.sort_values(by=sort_cols).reset_index(drop=True)
    else:
        cleaned = cleaned.reset_index(drop=True)

    final_rows = len(cleaned)
    removed_rows = input_rows - final_rows

    report = {
        "input_rows": input_rows,
        "removed_rows": removed_rows,
        "final_rows": final_rows,
        "duplicates_removed": duplicates_removed,
        "invalid_timestamps": invalid_timestamps,
        "negative_durations": negative_dwell + negative_flight,
        "extreme_outliers": extreme_dwell + extreme_flight,
        "sensitive_text_columns_quarantined": sensitive_cols,
        "retention_rate_pct": round((final_rows / max(input_rows, 1)) * 100, 2),
    }

    return cleaned, report
