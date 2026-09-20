"""Data engineering package for keystroke dynamics metadata.

Handles dataset adaptation, registry, validation quality checks,
feature engineering, personal baseline analysis, and pipeline orchestration.
"""

from src.data_engineering.cleaner import clean_dataset_with_report, clean_keystroke_timing
from src.data_engineering.data_quality import (
    check_class_imbalance,
    check_duplicates,
    check_invalid_timestamps,
    check_missing_values,
    check_negative_durations,
    check_numeric_anomalies,
    generate_quality_summary,
)
from src.data_engineering.dataset_adapter import (
    detect_keystroke_columns,
    detect_label_column,
    detect_sensitive_text_columns,
    detect_session_column,
    detect_timestamp_column,
    detect_user_column,
    generate_dataset_report,
    inspect_columns,
    load_dataset,
)
from src.data_engineering.dataset_registry import (
    DatasetConfig,
    get_dataset_config,
    list_registered_datasets,
    register_dataset,
)
from src.data_engineering.feature_engineering import (
    build_feature_table,
    calculate_backspace_features,
    calculate_dwell_time,
    calculate_error_features,
    calculate_flight_time,
    calculate_pause_features,
    calculate_session_statistics,
    calculate_typing_speed,
    calculate_variability_features,
    calculate_word_features,
    extract_timing_features,
    get_available_features,
    validate_feature_table,
)
from src.data_engineering.feature_manifest import (
    generate_feature_manifest,
    save_feature_manifest,
)
from src.data_engineering.baseline import (
    DEFAULT_BASELINE_FEATURES,
    FEATURE_DIRECTION_METADATA,
    build_all_user_baselines,
    build_user_baseline,
    calculate_baseline_deviation,
    compute_baseline_strain,
    generate_baseline_report,
    get_valid_baseline_features,
    load_user_baselines,
    save_user_baselines,
    update_user_baseline,
)
from src.data_engineering.pipeline import run_feature_pipeline
from src.data_engineering.dataset_acquisition import DatasetStatus, get_dataset_status
from src.data_engineering.real_dataset_validator import validate_real_dataset
from src.data_engineering.adapters.mobilestress_adapter import (
    MobileStressAdapter,
    adapt_mobilestress_dataset,
)
from src.data_engineering.leakage_audit import assert_no_data_leakage, audit_data_leakage

__all__ = [
    "check_numeric_anomalies",
    "clean_keystroke_timing",
    "clean_dataset_with_report",
    "extract_timing_features",
    "calculate_dwell_time",
    "calculate_flight_time",
    "calculate_pause_features",
    "calculate_typing_speed",
    "calculate_backspace_features",
    "calculate_error_features",
    "calculate_word_features",
    "calculate_variability_features",
    "calculate_session_statistics",
    "build_feature_table",
    "get_available_features",
    "validate_feature_table",
    "generate_feature_manifest",
    "save_feature_manifest",
    "compute_baseline_strain",
    "build_user_baseline",
    "build_all_user_baselines",
    "calculate_baseline_deviation",
    "generate_baseline_report",
    "update_user_baseline",
    "save_user_baselines",
    "load_user_baselines",
    "get_valid_baseline_features",
    "DEFAULT_BASELINE_FEATURES",
    "FEATURE_DIRECTION_METADATA",
    "run_feature_pipeline",
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
    "check_class_imbalance",
    "generate_quality_summary",
    "get_dataset_status",
    "DatasetStatus",
    "validate_real_dataset",
    "MobileStressAdapter",
    "adapt_mobilestress_dataset",
    "audit_data_leakage",
    "assert_no_data_leakage",
]
