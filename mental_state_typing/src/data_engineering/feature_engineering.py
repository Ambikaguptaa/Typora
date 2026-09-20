"""Feature Engineering Module for Keystroke Dynamics.

Converts raw timing events into behavioral keystroke features:
- Dwell time distributions (mean, median, std, min, max, percentiles)
- Flight time / Inter-key interval dynamics
- Configurable pause metrics (frequency, duration, long pauses)
- Typing speed throughput (WPM)
- Motor variability metrics (CV, MAD)
- Backspace and error correction features
- Session-level behavioral vectors
- Feature table validation and zero-text audit
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def extract_timing_features(
    df: pd.DataFrame,
    pause_threshold_ms: float = 500.0,
) -> Dict[str, float]:
    """Calculate aggregated temporal features from a sequence of keystroke timings.

    Legacy compatibility function retained for existing components and tests.

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

    hold_times = df["release_time"].to_numpy() - df["press_time"].to_numpy()
    flight_times = (
        df["press_time"].iloc[1:].to_numpy()
        - df["release_time"].iloc[:-1].to_numpy()
    )
    flight_times = np.maximum(flight_times, 0.0)

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


def calculate_dwell_time(df: pd.DataFrame) -> Optional[pd.Series]:
    """Extract or compute key dwell times (hold duration) in milliseconds.

    Conceptually: dwell_time = release_time - press_time
    """
    if "dwell_time" in df.columns:
        return pd.to_numeric(df["dwell_time"], errors="coerce")
    elif "hold_time" in df.columns:
        return pd.to_numeric(df["hold_time"], errors="coerce")
    elif "press_time" in df.columns and "release_time" in df.columns:
        pt = pd.to_numeric(df["press_time"], errors="coerce")
        rt = pd.to_numeric(df["release_time"], errors="coerce")
        return rt - pt
    return None


def calculate_flight_time(
    df: pd.DataFrame,
    user_col: Optional[str] = None,
    session_col: Optional[str] = None,
    timestamp_col: Optional[str] = None,
) -> Optional[pd.Series]:
    """Extract or compute inter-key flight times (latency between keys) in milliseconds.

    Conceptually: flight_time[i] = press_time[i] - release_time[i-1]
    Strictly prevents cross-session or cross-user flight calculation.
    First keystroke of a session receives NaN.
    """
    if "flight_time" in df.columns:
        return pd.to_numeric(df["flight_time"], errors="coerce")
    elif "iki" in df.columns:
        return pd.to_numeric(df["iki"], errors="coerce")

    # If press_time and release_time exist, calculate consecutively
    if "press_time" in df.columns and "release_time" in df.columns:
        df_sorted = df.copy()
        sort_cols = [c for c in [user_col, session_col, timestamp_col, "press_time"] if c and c in df.columns]
        if sort_cols:
            df_sorted = df_sorted.sort_values(by=sort_cols)

        group_cols = [c for c in [user_col, session_col] if c and c in df.columns]
        
        if group_cols:
            # Shift within each group
            prev_release = df_sorted.groupby(group_cols)["release_time"].shift(1)
        else:
            prev_release = df_sorted["release_time"].shift(1)

        flight = pd.to_numeric(df_sorted["press_time"], errors="coerce") - pd.to_numeric(prev_release, errors="coerce")
        # Mild rollover typing can produce small negative values; clamp to 0.0 but leave first event NaN
        return flight.apply(lambda x: max(0.0, x) if pd.notnull(x) else np.nan)

    return None


def calculate_variability_features(
    series: pd.Series, prefix: str
) -> Dict[str, float]:
    """Calculate statistical variability measures (std, CV, MAD) safely.

    Handles zero/near-zero means safely without generating Inf or NaN.

    Args:
        series: Numeric series of observations.
        prefix: Metric name prefix (e.g. 'dwell_time', 'flight_time').

    Returns:
        Dict[str, float]: Dictionary of variability metrics.
    """
    valid = series.dropna()
    if len(valid) < 2:
        return {
            f"std_{prefix}": 0.0,
            f"cv_{prefix}": 0.0,
            f"mad_{prefix}": 0.0,
        }

    mean_val = float(valid.mean())
    std_val = float(valid.std(ddof=1))

    # Coefficient of Variation = std / mean (guarded against zero division)
    if abs(mean_val) > 1e-5:
        cv_val = float(std_val / mean_val)
    else:
        cv_val = 0.0

    # Median Absolute Deviation (MAD) = median(|x - median(x)|)
    median_val = float(valid.median())
    mad_val = float((valid - median_val).abs().median())

    return {
        f"std_{prefix}": round(std_val, 3),
        f"cv_{prefix}": round(cv_val, 4),
        f"mad_{prefix}": round(mad_val, 3),
    }


def calculate_pause_features(
    flight_series: pd.Series,
    pause_threshold_ms: float = 2000.0,
) -> Dict[str, float]:
    """Calculate pause statistics from an inter-key flight time series.

    Args:
        flight_series: Series of inter-key latency intervals in ms.
        pause_threshold_ms: Duration in ms considered a cognitive pause (default: 2000ms = 2.0s).

    Returns:
        Dict[str, float]: Pause metrics dictionary.
    """
    valid = flight_series.dropna()
    if len(valid) == 0:
        return {
            "pause_rate": 0.0,
            "mean_pause_duration": 0.0,
            "median_pause_duration": 0.0,
            "max_pause_duration": 0.0,
            "long_pause_count": 0.0,
            "long_pause_ratio": 0.0,
        }

    pauses = valid[valid >= pause_threshold_ms]
    pause_count = len(pauses)
    total_intervals = len(valid)

    pause_rate = float(pause_count / total_intervals) if total_intervals > 0 else 0.0
    mean_pause = float(pauses.mean()) if pause_count > 0 else 0.0
    median_pause = float(pauses.median()) if pause_count > 0 else 0.0
    max_pause = float(pauses.max()) if pause_count > 0 else 0.0

    # Extended long pause threshold (e.g. 4000ms / 4 seconds)
    extended_pauses = valid[valid >= 4000.0]
    extended_count = len(extended_pauses)
    extended_ratio = float(extended_count / total_intervals) if total_intervals > 0 else 0.0

    return {
        "pause_rate": round(pause_rate, 4),
        "mean_pause_duration": round(mean_pause, 2),
        "median_pause_duration": round(median_pause, 2),
        "max_pause_duration": round(max_pause, 2),
        "long_pause_count": float(extended_count),
        "long_pause_ratio": round(extended_ratio, 4),
    }


def calculate_typing_speed(
    df: pd.DataFrame,
    dwell_series: Optional[pd.Series] = None,
    flight_series: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """Calculate typing speed in Words Per Minute (WPM).

    Standard Definition:
        WPM = (number_of_characters / 5.0) / elapsed_minutes

    If dataset provides an explicit 'typing_speed' column, statistical aggregates
    of that column are used.
    """
    if "typing_speed" in df.columns:
        speed_col = pd.to_numeric(df["typing_speed"], errors="coerce").dropna()
        if len(speed_col) > 0:
            return {
                "mean_typing_speed_wpm": round(float(speed_col.mean()), 2),
                "median_typing_speed_wpm": round(float(speed_col.median()), 2),
                "std_typing_speed_wpm": round(float(speed_col.std(ddof=1)), 2) if len(speed_col) > 1 else 0.0,
                "min_typing_speed_wpm": round(float(speed_col.min()), 2),
                "max_typing_speed_wpm": round(float(speed_col.max()), 2),
            }

    # Empirical WPM calculation from timing intervals
    keystroke_count = len(df)
    if keystroke_count < 5:
        return {
            "mean_typing_speed_wpm": 0.0,
            "median_typing_speed_wpm": 0.0,
            "std_typing_speed_wpm": 0.0,
            "min_typing_speed_wpm": 0.0,
            "max_typing_speed_wpm": 0.0,
        }

    # Estimate elapsed time from sum of dwell and flight intervals
    total_elapsed_ms = 0.0
    if dwell_series is not None and len(dwell_series.dropna()) > 0:
        total_elapsed_ms += float(dwell_series.dropna().sum())
    if flight_series is not None and len(flight_series.dropna()) > 0:
        total_elapsed_ms += float(flight_series.dropna().sum())

    elapsed_minutes = total_elapsed_ms / 60000.0
    if elapsed_minutes > 0.001:
        wpm = (keystroke_count / 5.0) / elapsed_minutes
        wpm = round(float(min(wpm, 250.0)), 2)  # clamp at physiological typing ceiling
    else:
        wpm = 0.0

    return {
        "mean_typing_speed_wpm": wpm,
        "median_typing_speed_wpm": wpm,
        "std_typing_speed_wpm": 0.0,
        "min_typing_speed_wpm": wpm,
        "max_typing_speed_wpm": wpm,
    }


def calculate_backspace_features(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate backspace frequency and edit correction ratios."""
    keystroke_count = len(df)
    total_backspaces = 0

    if "backspace" in df.columns:
        total_backspaces = int(pd.to_numeric(df["backspace"], errors="coerce").fillna(0).sum())
    elif "is_backspace" in df.columns:
        total_backspaces = int(pd.to_numeric(df["is_backspace"], errors="coerce").fillna(0).sum())
    elif "key" in df.columns:
        backspace_labels = {"backspace", "key.backspace", "8", "[backspace]"}
        key_str = df["key"].astype(str).str.lower().str.strip()
        total_backspaces = int(key_str.isin(backspace_labels).sum())

    rate = float(total_backspaces / max(keystroke_count, 1))

    return {
        "total_backspaces": float(total_backspaces),
        "backspace_rate": round(rate, 4),
        "backspace_proportion": round(rate, 4),
    }


def calculate_error_features(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate explicit typing error rates if error columns exist.

    Does NOT fabricate error metrics if raw error flags are not present.
    """
    error_candidates = ["error_flag", "is_error", "error", "typing_error"]
    matched_col = next((c for c in error_candidates if c in df.columns), None)

    if not matched_col:
        # Return empty dictionary — feature is marked unavailable in manifest
        return {}

    errors = pd.to_numeric(df[matched_col], errors="coerce").fillna(0)
    total_errors = float(errors.sum())
    rate = total_errors / max(len(df), 1)

    return {
        "total_errors": total_errors,
        "error_rate": round(rate, 4),
    }


def calculate_word_features(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate word completion timing features if word boundary markers exist.

    Does NOT reconstruct words from sensitive text.
    """
    word_time_col = next((c for c in ["word_time", "word_duration"] if c in df.columns), None)
    if not word_time_col:
        # Unavailable without dedicated word boundary timing records
        return {}

    valid_words = pd.to_numeric(df[word_time_col], errors="coerce").dropna()
    if len(valid_words) < 2:
        return {}

    return {
        "mean_word_completion_time": round(float(valid_words.mean()), 2),
        "median_word_completion_time": round(float(valid_words.median()), 2),
        "word_completion_variability": round(float(valid_words.std(ddof=1)), 2),
    }


def calculate_session_statistics(
    session_df: pd.DataFrame,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    label_val: Optional[Any] = None,
    pause_threshold_ms: float = 2000.0,
) -> Dict[str, Any]:
    """Compute a consolidated session-level behavioral feature vector.

    Args:
        session_df: Keystroke records belonging to a single session.
        user_id: Participant identifier.
        session_id: Session identifier.
        label_val: Target behavioral/psychological state label.
        pause_threshold_ms: Milliseconds threshold for pause detection.

    Returns:
        Dict[str, Any]: Comprehensive session feature dictionary.
    """
    keystroke_count = len(session_df)
    record: Dict[str, Any] = {
        "user_id": user_id or "unknown_user",
        "session_id": session_id or "unknown_session",
        "keystroke_count": keystroke_count,
    }

    # Duration
    if "timestamp" in session_df.columns:
        ts = pd.to_numeric(session_df["timestamp"], errors="coerce").dropna()
        if len(ts) >= 2:
            duration_ms = ts.max() - ts.min()
            record["session_duration_sec"] = round(float(duration_ms) / 1000.0, 2)
        else:
            record["session_duration_sec"] = 0.0
    else:
        record["session_duration_sec"] = 0.0

    # Dwell Time
    dwell_series = calculate_dwell_time(session_df)
    if dwell_series is not None and len(dwell_series.dropna()) > 0:
        valid_dwell = dwell_series.dropna()
        record["mean_dwell_time"] = round(float(valid_dwell.mean()), 2)
        record["median_dwell_time"] = round(float(valid_dwell.median()), 2)
        record["min_dwell_time"] = round(float(valid_dwell.min()), 2)
        record["max_dwell_time"] = round(float(valid_dwell.max()), 2)
        record["p25_dwell_time"] = round(float(np.percentile(valid_dwell, 25)), 2)
        record["p75_dwell_time"] = round(float(np.percentile(valid_dwell, 75)), 2)

        # Dwell variability
        dwell_var = calculate_variability_features(valid_dwell, "dwell_time")
        record.update(dwell_var)

    # Flight Time
    flight_series = calculate_flight_time(session_df)
    if flight_series is not None and len(flight_series.dropna()) > 0:
        valid_flight = flight_series.dropna()
        record["mean_flight_time"] = round(float(valid_flight.mean()), 2)
        record["median_flight_time"] = round(float(valid_flight.median()), 2)
        record["min_flight_time"] = round(float(valid_flight.min()), 2)
        record["max_flight_time"] = round(float(valid_flight.max()), 2)

        # Flight variability
        flight_var = calculate_variability_features(valid_flight, "flight_time")
        record.update(flight_var)

        # Pause features
        pause_feats = calculate_pause_features(valid_flight, pause_threshold_ms=pause_threshold_ms)
        record.update(pause_feats)

    # Typing Speed
    speed_feats = calculate_typing_speed(session_df, dwell_series, flight_series)
    record.update(speed_feats)

    # Backspace & Error Features
    backspace_feats = calculate_backspace_features(session_df)
    record.update(backspace_feats)

    error_feats = calculate_error_features(session_df)
    record.update(error_feats)

    word_feats = calculate_word_features(session_df)
    record.update(word_feats)

    # Target Label
    if label_val is not None:
        record["target_label"] = label_val

    return record


def build_feature_table(
    df: pd.DataFrame,
    user_col: Optional[str] = "user_id",
    session_col: Optional[str] = "session_id",
    timestamp_col: Optional[str] = "timestamp",
    label_col: Optional[str] = "state",
    pause_threshold_ms: float = 2000.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Transform raw keystroke event records into a structured session-level feature matrix.

    Args:
        df: Cleaned keystroke events DataFrame.
        user_col: User column name.
        session_col: Session column name.
        timestamp_col: Timestamp column name.
        label_col: Target label column name.
        pause_threshold_ms: Threshold in ms for behavioral pauses.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: (Feature Table DataFrame, Extraction Summary).
    """
    rows_list: List[Dict[str, Any]] = []

    has_user = user_col and user_col in df.columns
    has_session = session_col and session_col in df.columns

    if has_user and has_session:
        grouped = df.groupby([user_col, session_col], sort=False)
        for (uid, sid), group in grouped:
            lbl = group[label_col].iloc[0] if (label_col and label_col in group.columns) else None
            feat_row = calculate_session_statistics(
                session_df=group,
                user_id=str(uid),
                session_id=str(sid),
                label_val=lbl,
                pause_threshold_ms=pause_threshold_ms,
            )
            rows_list.append(feat_row)
    elif has_session:
        grouped = df.groupby(session_col, sort=False)
        for sid, group in grouped:
            uid = group[user_col].iloc[0] if (has_user) else "unknown_user"
            lbl = group[label_col].iloc[0] if (label_col and label_col in group.columns) else None
            feat_row = calculate_session_statistics(
                session_df=group,
                user_id=str(uid),
                session_id=str(sid),
                label_val=lbl,
                pause_threshold_ms=pause_threshold_ms,
            )
            rows_list.append(feat_row)
    else:
        # Process entire DataFrame as a single session
        lbl = df[label_col].iloc[0] if (label_col and label_col in df.columns) else None
        feat_row = calculate_session_statistics(
            session_df=df,
            user_id="default_user",
            session_id="default_session",
            label_val=lbl,
            pause_threshold_ms=pause_threshold_ms,
        )
        rows_list.append(feat_row)

    feature_df = pd.DataFrame(rows_list)

    summary = {
        "total_sessions_processed": len(feature_df),
        "total_input_events": len(df),
        "feature_count": len([c for c in feature_df.columns if c not in ["user_id", "session_id", "target_label"]]),
        "has_label_column": "target_label" in feature_df.columns,
    }

    return feature_df, summary


def get_available_features(df: pd.DataFrame) -> Dict[str, bool]:
    """Identify which behavioral feature categories are supported by the dataset columns."""
    cols = {c.lower().strip() for c in df.columns}
    return {
        "dwell_time": any(c in cols for c in ["dwell_time", "hold_time"]) or ("press_time" in cols and "release_time" in cols),
        "flight_time": any(c in cols for c in ["flight_time", "iki"]) or ("press_time" in cols and "release_time" in cols),
        "pause_metrics": any(c in cols for c in ["flight_time", "iki", "pause_duration"]) or ("press_time" in cols and "release_time" in cols),
        "typing_speed": "typing_speed" in cols or ("press_time" in cols and "release_time" in cols),
        "backspaces": any(c in cols for c in ["backspace", "key"]),
        "errors": any(c in cols for c in ["error_flag", "is_error", "error"]),
        "word_completion": any(c in cols for c in ["word_time", "word_duration"]),
        "behavioral_label": any(c in cols for c in ["state", "strain", "stress", "fatigue", "workload", "label"]),
    }


def validate_feature_table(feature_df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate that numerical feature columns contain zero NaN or Infinite values.

    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_violations).
    """
    violations: List[str] = []
    metadata_cols = {"user_id", "session_id", "target_label"}
    num_cols = [c for c in feature_df.columns if c not in metadata_cols]

    for col in num_cols:
        series = feature_df[col]
        nan_c = int(series.isnull().sum())
        inf_c = int(np.isinf(pd.to_numeric(series, errors="coerce")).sum())
        if nan_c > 0:
            violations.append(f"Column '{col}' contains {nan_c} NaN values.")
        if inf_c > 0:
            violations.append(f"Column '{col}' contains {inf_c} Infinite values.")

    is_valid = len(violations) == 0
    return is_valid, violations
