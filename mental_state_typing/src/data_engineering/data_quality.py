"""Data Quality and Validation Module.

Provides comprehensive diagnostic functions to audit keystroke datasets:
- Missing value analysis
- Duplicate record detection
- Timestamp monotonicity and validity
- Non-negative duration verification
- Outlier / extreme anomaly identification
- Class balance metrics
- Composite quality health index
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a structured report of missing values per column.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: Summary table with columns ['column', 'missing_count', 'missing_percentage'].
    """
    total_rows = len(df)
    missing_series = df.isnull().sum()

    report_data = []
    for col, count in missing_series.items():
        pct = round((count / total_rows) * 100, 2) if total_rows > 0 else 0.0
        report_data.append(
            {
                "column": str(col),
                "missing_count": int(count),
                "missing_percentage": pct,
            }
        )

    return pd.DataFrame(report_data)


def check_duplicates(
    df: pd.DataFrame, subset: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Detect duplicate records across the dataset or a subset of key columns.

    Args:
        df: Input DataFrame.
        subset: Optional list of columns to consider for uniqueness.

    Returns:
        Dict[str, Any]: Count, percentage, and boolean flag.
    """
    total_rows = len(df)
    dup_mask = df.duplicated(subset=subset, keep="first")
    dup_count = int(dup_mask.sum())
    dup_pct = round((dup_count / total_rows) * 100, 2) if total_rows > 0 else 0.0

    return {
        "duplicate_count": dup_count,
        "duplicate_percentage": dup_pct,
        "has_duplicates": dup_count > 0,
        "total_rows": total_rows,
    }


def check_invalid_timestamps(
    df: pd.DataFrame, timestamp_col: Optional[str] = "timestamp"
) -> Dict[str, Any]:
    """Verify that timestamp records are positive, non-null, and monotonically increasing.

    Args:
        df: Input DataFrame.
        timestamp_col: Name of timestamp column.

    Returns:
        Dict[str, Any]: Diagnostic results on timestamp integrity.
    """
    if not timestamp_col or timestamp_col not in df.columns:
        return {
            "status": "column_missing",
            "negative_timestamps": 0,
            "null_timestamps": 0,
            "disordered_timestamps": 0,
            "is_valid": False,
        }

    series = df[timestamp_col].dropna()
    null_count = int(df[timestamp_col].isnull().sum())
    negative_count = int((series < 0).sum())

    # Check non-decreasing order if sorted by user/session or sequentially
    diffs = series.diff().dropna()
    disordered_count = int((diffs < 0).sum())

    is_valid = (null_count == 0) and (negative_count == 0)

    return {
        "status": "evaluated",
        "negative_timestamps": negative_count,
        "null_timestamps": null_count,
        "disordered_timestamps": disordered_count,
        "is_valid": is_valid,
    }


def check_negative_durations(
    df: pd.DataFrame, duration_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Ensure micro-timing durations (dwell time, flight time, pause) are non-negative.

    Args:
        df: Input DataFrame.
        duration_cols: Columns to evaluate. Defaults to common timing variables.

    Returns:
        Dict[str, Any]: Anomaly counts per duration column.
    """
    if duration_cols is None:
        common_candidates = [
            "dwell_time",
            "flight_time",
            "pause_duration",
            "hold_time",
            "interval",
            "iki",
        ]
        duration_cols = [c for c in common_candidates if c in df.columns]

    violations: Dict[str, int] = {}
    total_violations = 0

    for col in duration_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        neg_count = int((series < 0).sum())
        violations[col] = neg_count
        total_violations += neg_count

    return {
        "evaluated_columns": duration_cols,
        "violations_by_column": violations,
        "total_negative_durations": total_violations,
        "has_negative_durations": total_violations > 0,
    }


def check_numeric_anomalies(
    df: pd.DataFrame, numeric_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Audit numeric columns for infinite values, NaNs, and physiological extremes."""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    infinite_counts: Dict[str, int] = {}
    nan_counts: Dict[str, int] = {}
    extreme_dwell_count = 0

    for col in numeric_cols:
        series = df[col]
        inf_c = int(np.isinf(series).sum())
        nan_c = int(series.isnull().sum())
        if inf_c > 0:
            infinite_counts[col] = inf_c
        if nan_c > 0:
            nan_counts[col] = nan_c

    # Check physiological bounds for dwell time (> 4000ms is usually a key held during distraction)
    dwell_candidates = [c for c in ["dwell_time", "hold_time"] if c in df.columns]
    for col in dwell_candidates:
        extreme_dwell_count += int((df[col] > 4000.0).sum())

    return {
        "infinite_values": infinite_counts,
        "nan_values": nan_counts,
        "extreme_dwell_count": extreme_dwell_count,
        "has_numeric_anomalies": bool(infinite_counts or nan_counts or extreme_dwell_count > 0),
    }


def check_class_imbalance(
    df: pd.DataFrame, label_col: Optional[str] = None
) -> Dict[str, Any]:
    """Calculate frequency breakdown, imbalance ratio, and entropy of target labels.

    Args:
        df: Input DataFrame.
        label_col: Target label column name.

    Returns:
        Dict[str, Any]: Class distribution metrics.
    """
    if not label_col or label_col not in df.columns:
        return {
            "status": "no_label_column",
            "class_counts": {},
            "class_proportions": {},
            "imbalance_ratio": 1.0,
            "is_imbalanced": False,
        }

    counts = df[label_col].value_counts()
    total = len(df[label_col].dropna())
    if total == 0:
        return {
            "status": "empty_column",
            "class_counts": {},
            "class_proportions": {},
            "imbalance_ratio": 1.0,
            "is_imbalanced": False,
        }

    proportions = {str(k): round(float(v) / total, 4) for k, v in counts.items()}
    counts_dict = {str(k): int(v) for k, v in counts.items()}

    # Imbalance ratio = max_class_count / min_class_count
    max_c = max(counts_dict.values()) if counts_dict else 1
    min_c = min(counts_dict.values()) if counts_dict else 1
    ratio = round(max_c / max(min_c, 1), 2)

    return {
        "status": "evaluated",
        "label_column": label_col,
        "class_counts": counts_dict,
        "class_proportions": proportions,
        "imbalance_ratio": ratio,
        "is_imbalanced": ratio > 3.0,
    }


def generate_quality_summary(
    df: pd.DataFrame, config: Optional[Any] = None
) -> Dict[str, Any]:
    """Compute a composite dataset quality score (0 - 100) and actionable flags.

    Args:
        df: Input DataFrame.
        config: Optional DatasetConfig object.

    Returns:
        Dict[str, Any]: Comprehensive health rating and category evaluations.
    """
    missing_df = check_missing_values(df)
    total_missing = int(missing_df["missing_count"].sum())
    total_cells = df.shape[0] * df.shape[1] if df.shape[0] * df.shape[1] > 0 else 1
    missing_pct = (total_missing / total_cells) * 100.0

    dup_res = check_duplicates(df)
    neg_res = check_negative_durations(df)
    num_res = check_numeric_anomalies(df)

    # Base quality score starts at 100
    score = 100.0

    # Penalize missing values (up to 30 points)
    score -= min(missing_pct * 3.0, 30.0)

    # Penalize duplicate rows (up to 20 points)
    score -= min(dup_res["duplicate_percentage"] * 2.0, 20.0)

    # Penalize negative durations heavily (up to 30 points)
    if neg_res["has_negative_durations"]:
        score -= min(neg_res["total_negative_durations"] * 2.0, 30.0)

    # Penalize infinite / NaN anomalies (up to 20 points)
    if num_res["has_numeric_anomalies"]:
        score -= 15.0

    quality_score = max(0.0, min(100.0, round(score, 1)))

    if quality_score >= 85.0:
        status = "EXCELLENT"
        badge = "positive"
    elif quality_score >= 65.0:
        status = "ACCEPTABLE"
        badge = "warning"
    else:
        status = "CRITICAL_ISSUES"
        badge = "critical"

    return {
        "quality_score": quality_score,
        "status": status,
        "status_badge": badge,
        "missing_percentage": round(missing_pct, 2),
        "total_missing_values": total_missing,
        "duplicate_rows": dup_res["duplicate_count"],
        "negative_durations": neg_res["total_negative_durations"],
        "has_numeric_anomalies": num_res["has_numeric_anomalies"],
    }
