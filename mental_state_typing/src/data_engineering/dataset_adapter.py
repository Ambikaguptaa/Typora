"""Dataset Adapter Module for Keystroke Dynamics Datasets.

Provides schema-agnostic functions to inspect, adapt, and profile arbitrary
keystroke datasets without assuming fixed column naming conventions.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

# Heuristic keyword patterns for auto-detection
USER_PATTERNS = [
    "user_id",
    "user",
    "participant_id",
    "participant",
    "subject_id",
    "subject",
    "userid",
    "sub_id",
    "uid",
]

SESSION_PATTERNS = [
    "session_id",
    "session",
    "sess_id",
    "sess",
    "trial_id",
    "trial",
    "task_id",
    "task",
]

TIMESTAMP_PATTERNS = [
    "timestamp",
    "time",
    "press_time",
    "down_time",
    "datetime",
    "epoch",
    "press_timestamp",
]

KEYSTROKE_PATTERNS = [
    "dwell",
    "hold",
    "flight",
    "iki",
    "interval",
    "pause",
    "latency",
    "speed",
    "wpm",
    "backspace",
    "error",
    "press_time",
    "release_time",
    "down_time",
    "up_time",
    "keystroke",
]

LABEL_PATTERNS = [
    "state",
    "strain",
    "stress",
    "fatigue",
    "workload",
    "label",
    "target",
    "mood",
    "emotion",
    "condition",
    "class",
    "valence",
    "arousal",
    "mental_state",
]

SENSITIVE_TEXT_PATTERNS = [
    "text",
    "message",
    "sentence",
    "content",
    "typed_text",
    "word",
    "keystring",
    "raw_input",
    "keystrokes_text",
]


def load_dataset(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load a dataset from CSV or Excel file.

    Args:
        file_path: Path to dataset file (.csv, .xlsx, .xls).

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file extension is unsupported.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in [".xlsx", ".xls"]:
        try:
            return pd.read_excel(path)
        except ImportError as e:
            raise ImportError(
                "openpyxl is required to load Excel files. Install via `pip install openpyxl`."
            ) from e
    elif ext in [".json", ".jsonl"]:
        return pd.read_json(path)
    else:
        raise ValueError(
            f"Unsupported file format: '{ext}'. Supported formats: .csv, .xlsx, .xls, .json"
        )


def inspect_columns(df: pd.DataFrame) -> List[str]:
    """Return all column names of the DataFrame."""
    return [str(c) for c in df.columns]


def _match_column(columns: List[str], patterns: List[str]) -> Optional[str]:
    """Helper to find the best match for a heuristic pattern list."""
    col_map = {col.lower().strip(): col for col in columns}

    # 1. Exact match pass
    for pat in patterns:
        if pat in col_map:
            return col_map[pat]

    # 2. Substring match pass
    for pat in patterns:
        for col_lower, original_col in col_map.items():
            if pat in col_lower:
                return original_col

    return None


def detect_user_column(df: pd.DataFrame) -> Optional[str]:
    """Detect participant/user identifier column."""
    return _match_column(inspect_columns(df), USER_PATTERNS)


def detect_session_column(df: pd.DataFrame) -> Optional[str]:
    """Detect session/trial identifier column."""
    return _match_column(inspect_columns(df), SESSION_PATTERNS)


def detect_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    """Detect timestamp or press time column."""
    return _match_column(inspect_columns(df), TIMESTAMP_PATTERNS)


def detect_keystroke_columns(df: pd.DataFrame) -> List[str]:
    """Detect all columns related to keystroke dynamics micro-timings."""
    matched: List[str] = []
    for col in inspect_columns(df):
        col_lower = col.lower().strip()
        for pat in KEYSTROKE_PATTERNS:
            if pat in col_lower and col not in matched:
                matched.append(col)
                break
    return matched


def detect_label_column(df: pd.DataFrame) -> Optional[str]:
    """Detect behavioral/psychological state target label column."""
    return _match_column(inspect_columns(df), LABEL_PATTERNS)


def detect_sensitive_text_columns(df: pd.DataFrame) -> List[str]:
    """Detect columns containing textual or typed sentence content.

    These columns must be flagged as sensitive and NEVER displayed or stored.
    """
    sensitive: List[str] = []
    for col in inspect_columns(df):
        col_lower = col.lower().strip()
        for pat in SENSITIVE_TEXT_PATTERNS:
            if pat in col_lower and col not in sensitive:
                sensitive.append(col)
                break
    return sensitive


def analyze_missing_values(df: pd.DataFrame) -> Dict[str, int]:
    """Compute count of missing values per column."""
    missing = df.isnull().sum()
    return {col: int(count) for col, count in missing.items() if count > 0}


def analyze_duplicates(df: pd.DataFrame) -> int:
    """Return count of duplicate rows in the DataFrame."""
    return int(df.duplicated().sum())


def analyze_class_distribution(
    df: pd.DataFrame, label_column: Optional[str] = None
) -> Dict[str, int]:
    """Compute distribution of classes if label column is present."""
    if not label_column or label_column not in df.columns:
        return {}
    value_counts = df[label_column].value_counts()
    return {str(k): int(v) for k, v in value_counts.items()}


def generate_dataset_report(
    df: pd.DataFrame, dataset_name: Optional[str] = None
) -> Dict[str, Any]:
    """Generate a comprehensive, structured dataset report.

    Args:
        df: Input DataFrame.
        dataset_name: Optional name identifier for the dataset.

    Returns:
        Dict[str, Any]: Structured inspection report.
    """
    user_col = detect_user_column(df)
    session_col = detect_session_column(df)
    timestamp_col = detect_timestamp_column(df)
    keystroke_cols = detect_keystroke_columns(df)
    label_col = detect_label_column(df)
    sensitive_cols = detect_sensitive_text_columns(df)

    missing_dict = analyze_missing_values(df)
    total_cells = df.shape[0] * df.shape[1] if df.shape[0] * df.shape[1] > 0 else 1
    total_missing_cells = sum(missing_dict.values())
    missing_pct = round((total_missing_cells / total_cells) * 100, 2)

    duplicate_count = analyze_duplicates(df)
    duplicate_pct = (
        round((duplicate_count / len(df)) * 100, 2) if len(df) > 0 else 0.0
    )

    unique_users = int(df[user_col].nunique()) if user_col and user_col in df else 0
    unique_sessions = (
        int(df[session_col].nunique()) if session_col and session_col in df else 0
    )

    class_dist = analyze_class_distribution(df, label_col)

    return {
        "dataset_name": dataset_name or "Unnamed Dataset",
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": inspect_columns(df),
        "user_column": user_col,
        "session_column": session_col,
        "timestamp_column": timestamp_col,
        "keystroke_columns": keystroke_cols,
        "label_column": label_col,
        "sensitive_columns": sensitive_cols,
        "missing_values": missing_dict,
        "missing_percentage": missing_pct,
        "duplicate_rows": duplicate_count,
        "duplicate_percentage": duplicate_pct,
        "unique_users": unique_users,
        "unique_sessions": unique_sessions,
        "class_distribution": class_dist,
    }
