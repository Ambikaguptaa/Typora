"""Data engineering package for keystroke dynamics metadata.

Handles dataset adaptation, registry, validation quality checks, feature extraction,
and baseline strain metrics.
"""

from src.data_engineering.cleaner import clean_keystroke_timing
from src.data_engineering.feature_engineering import extract_timing_features
from src.data_engineering.loader import load_keystroke_dataset
from src.data_engineering.baseline import compute_baseline_strain
from src.data_engineering.dataset_adapter import (
    generate_dataset_report,
    load_dataset,
    inspect_columns,
    detect_user_column,
    detect_session_column,
    detect_timestamp_column,
    detect_keystroke_columns,
    detect_label_column,
    detect_sensitive_text_columns,
)
from src.data_engineering.dataset_registry import (
    DatasetConfig,
    get_dataset_config,
    list_registered_datasets,
    register_dataset,
)
from src.data_engineering.data_quality import (
    check_class_imbalance,
    check_duplicates,
    check_invalid_timestamps,
    check_missing_values,
    check_negative_durations,
    check_numeric_anomalies,
    generate_quality_summary,
)

__all__ = [
    "clean_keystroke_timing",
    "extract_timing_features",
    "load_keystroke_dataset",
    "compute_baseline_strain",
    "load_dataset",
    "inspect_columns",
    "detect_user_column",
    "detect_session_column",
    "detect_timestamp_column",
    "detect_keystroke_columns",
    "detect_label_column",
    "detect_sensitive_text_columns",
    "generate_dataset_report",
    "DatasetConfig",
    "get_dataset_config",
    "list_registered_datasets",
    "register_dataset",
    "check_missing_values",
    "check_duplicates",
    "check_invalid_timestamps",
    "check_negative_durations",
    "check_numeric_anomalies",
    "check_class_imbalance",
    "generate_quality_summary",
]
