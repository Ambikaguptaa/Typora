"""Feature engineering module for keystroke dynamics metadata.

Derives temporal features:
- Hold Time (dwell time): Duration a key remains pressed (release_time - press_time)
- Flight Time: Time elapsed between releasing one key and pressing the next
- Pause Rate: Frequency of hesitations or elongated pauses (> pause_threshold_ms)
"""

from typing import Dict
import numpy as np
import pandas as pd


def extract_timing_features(
    df: pd.DataFrame,
    pause_threshold_ms: float = 500.0,
) -> Dict[str, float]:
    """Calculate aggregated temporal features from a sequence of keystroke timings.

    Args:
        df: DataFrame with 'press_time' and 'release_time' columns in milliseconds.
        pause_threshold_ms: Duration in milliseconds considered a behavioral pause.

    Returns:
        Dict[str, float]: Dictionary of derived temporal features.
    """
    if df.empty or len(df) < 2:
        return {
            "mean_hold_time_ms": 0.0,
            "std_hold_time_ms": 0.0,
            "mean_flight_time_ms": 0.0,
            "std_flight_time_ms": 0.0,
            "pause_rate": 0.0,
            "keystroke_count": float(len(df)),
        }

    # Calculate hold time for each key
    hold_times = df["release_time"].to_numpy() - df["press_time"].to_numpy()

    # Calculate flight time between consecutive keystrokes:
    # flight_time[i] = press_time[i] - release_time[i-1]
    flight_times = (
        df["press_time"].iloc[1:].to_numpy()
        - df["release_time"].iloc[:-1].to_numpy()
    )
    # Clip negative flight times from fast rollover typing to 0
    flight_times = np.maximum(flight_times, 0.0)

    # Calculate pause occurrences
    pause_count = np.sum(flight_times > pause_threshold_ms)
    pause_rate = float(pause_count) / max(len(flight_times), 1)

    return {
        "mean_hold_time_ms": float(np.mean(hold_times)),
        "std_hold_time_ms": float(np.std(hold_times)),
        "mean_flight_time_ms": float(np.mean(flight_times)),
        "std_flight_time_ms": float(np.std(flight_times)),
        "pause_rate": float(pause_rate),
        "keystroke_count": float(len(df)),
    }
