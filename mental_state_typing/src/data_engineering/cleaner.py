"""Data cleaner module for keystroke timing data.

Filters out noise, missing values, and physiologically implausible timings
(e.g., negative duration or keys held down for an unrealistic duration).
"""

import pandas as pd


def clean_keystroke_timing(
    df: pd.DataFrame,
    min_hold_ms: float = 10.0,
    max_hold_ms: float = 2000.0,
    max_flight_ms: float = 5000.0,
) -> pd.DataFrame:
    """Clean and filter keystroke timing records.

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
