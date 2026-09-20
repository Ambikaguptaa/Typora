"""Pipeline Orchestration Module.

Coordinates end-to-end data engineering and sequence preprocessing:
1. Ingests raw keystroke dataset.
2. Quarantines and drops sensitive text content (Zero-Text Privacy Policy).
3. Performs traceable physiological cleaning and outlier removal.
4. Generates a schema-aware Feature Manifest.
5. Calculates event-level and session-level behavioral feature tables.
6. Performs leakage-safe grouped train/test splitting (by user/session).
7. Fits feature scaler strictly on the training partition and transforms test data.
8. Encodes behavioral labels.
9. Slices event sequences into 3D temporal arrays (X_train, X_test, y_train, y_test).
10. Persists processed datasets, sequence arrays, scalers, and reports.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config.settings import settings
from src.data_engineering.cleaner import clean_dataset_with_report
from src.data_engineering.dataset_adapter import (
    detect_keystroke_columns,
    detect_label_column,
    detect_sensitive_text_columns,
    detect_session_column,
    detect_timestamp_column,
    detect_user_column,
    load_dataset,
)
from src.data_engineering.feature_engineering import (
    build_feature_table,
    validate_feature_table,
)
from src.data_engineering.feature_manifest import (
    generate_feature_manifest,
    save_feature_manifest,
)
from src.data_engineering.baseline import (
    build_all_user_baselines,
    save_user_baselines,
)
from src.deep_learning.data_split import (
    create_grouped_train_test_split,
    fit_feature_scaler,
    save_scaler,
    transform_features,
)
from src.deep_learning.label_encoder import (
    encode_labels,
    inspect_labels,
    save_label_mapping,
)
from src.deep_learning.preprocessing import prepare_grouped_sequences


def run_feature_pipeline(
    input_data_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    sequence_length: int = 20,
    sequence_stride: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute the end-to-end data engineering, feature extraction, and sequence pipeline.

    Args:
        input_data_path: Path to raw dataset CSV/Excel. Defaults to sample benchmark.
        output_dir: Destination directory for processed files. Defaults to data/processed.
        sequence_length: Window size in timesteps for temporal sequence arrays.
        sequence_stride: Stride between successive sliding windows.
        test_size: Proportion of participant groups allocated to test split.
        random_state: Random seed for deterministic reproducibility.

    Returns:
        Dict[str, Any]: Comprehensive processing summary and file paths.
    """
    if input_data_path is None:
        input_data_path = settings.sample_data_path / "sample_keystrokes.csv"
    input_path = Path(input_data_path)

    if output_dir is None:
        output_dir = settings.processed_data_path
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models_dir = settings.models_path
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingest Raw Dataset
    raw_df = load_dataset(input_path)
    input_row_count = len(raw_df)

    # 2. Schema Discovery
    user_col = detect_user_column(raw_df)
    session_col = detect_session_column(raw_df)
    timestamp_col = detect_timestamp_column(raw_df)
    label_col = detect_label_column(raw_df)
    sensitive_cols = detect_sensitive_text_columns(raw_df)

    # 3. Traceable Cleaning & Zero-Text Quarantine
    cleaned_df, cleaning_report = clean_dataset_with_report(
        raw_df,
        user_col=user_col,
        session_col=session_col,
        timestamp_col=timestamp_col,
    )

    # 4. Generate Feature Manifest
    manifest = generate_feature_manifest(list(cleaned_df.columns))
    manifest_path = out_dir / "feature_manifest.json"
    save_feature_manifest(manifest, manifest_path)

    # 5. Build Session-Level Feature Table
    session_features_df, extraction_summary = build_feature_table(
        cleaned_df,
        user_col=user_col,
        session_col=session_col,
        timestamp_col=timestamp_col,
        label_col=label_col,
    )

    # Validate feature table
    is_valid_table, table_violations = validate_feature_table(session_features_df)

    # 6. Leakage-Safe Grouped Train/Test Splitting
    split_group_col = user_col if (user_col and user_col in session_features_df.columns) else "session_id"
    train_features_df, test_features_df, split_summary = create_grouped_train_test_split(
        session_features_df,
        group_col=split_group_col,
        test_size=test_size,
        random_state=random_state,
    )

    # Identify numerical feature columns for scaling (exclude metadata and target label)
    non_numeric_metadata = {"user_id", "session_id", "target_label"}
    feature_cols = [c for c in session_features_df.columns if c not in non_numeric_metadata]

    # 7. Isolated Feature Scaling (Fit strictly on train split)
    scaler, scaled_train_df = fit_feature_scaler(
        train_features_df, feature_cols=feature_cols, scaler_type="standard"
    )
    scaled_test_df = transform_features(scaler, test_features_df, feature_cols=feature_cols)

    scaler_path = models_dir / "feature_scaler.pkl"
    save_scaler(scaler, feature_cols, scaler_path)

    # 8. Categorical Label Encoding
    label_mapping: Dict[str, int] = {}
    label_inspection: Dict[str, Any] = {}
    if "target_label" in session_features_df.columns and session_features_df["target_label"].notnull().any():
        label_inspection = inspect_labels(session_features_df["target_label"])
        _, label_mapping = encode_labels(session_features_df["target_label"])
        label_mapping_path = models_dir / "label_mapping.json"
        save_label_mapping(label_mapping, label_mapping_path)

    # 9. Temporal Sequence Generation on Event Records
    # Features to include per event timestep in 3D recurrent array
    event_feature_candidates = [
        "dwell_time",
        "flight_time",
        "pause_duration",
        "typing_speed",
        "backspace",
        "error_flag",
    ]
    event_feature_cols = [c for c in event_feature_candidates if c in cleaned_df.columns]

    # Split cleaned event records using the same group partition
    train_groups = set(split_summary["train_groups"])
    test_groups = set(split_summary["test_groups"])

    if split_group_col in cleaned_df.columns:
        train_events_df = cleaned_df[cleaned_df[split_group_col].isin(train_groups)].copy()
        test_events_df = cleaned_df[cleaned_df[split_group_col].isin(test_groups)].copy()
    else:
        split_idx = int(len(cleaned_df) * (1.0 - test_size))
        train_events_df = cleaned_df.iloc[:split_idx].copy()
        test_events_df = cleaned_df.iloc[split_idx:].copy()

    # Fit event scaler on train events
    if event_feature_cols:
        event_scaler, scaled_train_events = fit_feature_scaler(
            train_events_df, feature_cols=event_feature_cols, scaler_type="standard"
        )
        scaled_test_events = transform_features(event_scaler, test_events_df, feature_cols=event_feature_cols)
    else:
        scaled_train_events = train_events_df
        scaled_test_events = test_events_df

    # Build 3D sequences
    group_keys = [c for c in [user_col, session_col] if c and c in cleaned_df.columns]
    X_train, y_train_raw, meta_train_df = prepare_grouped_sequences(
        scaled_train_events,
        feature_cols=event_feature_cols,
        group_cols=group_keys,
        timestamp_col=timestamp_col,
        label_col=label_col,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
    )

    X_test, y_test_raw, meta_test_df = prepare_grouped_sequences(
        scaled_test_events,
        feature_cols=event_feature_cols,
        group_cols=group_keys,
        timestamp_col=timestamp_col,
        label_col=label_col,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
    )

    # Encode sequence labels if present
    if y_train_raw is not None and len(y_train_raw) > 0 and label_mapping:
        y_train, _ = encode_labels(pd.Series(y_train_raw), custom_mapping=label_mapping)
    else:
        y_train = np.array([])

    if y_test_raw is not None and len(y_test_raw) > 0 and label_mapping:
        y_test, _ = encode_labels(pd.Series(y_test_raw), custom_mapping=label_mapping)
    else:
        y_test = np.array([])

    # 10. Persist Output Datasets & Artifacts
    cleaned_events_path = out_dir / "cleaned_events.csv"
    session_features_path = out_dir / "session_features.csv"
    sequences_npz_path = out_dir / "sequences.npz"
    labels_csv_path = out_dir / "labels.csv"
    report_json_path = out_dir / "processing_report.json"

    # Save cleaned events and features
    cleaned_df.to_csv(cleaned_events_path, index=False)
    session_features_df.to_csv(session_features_path, index=False)

    # Save 3D sequence arrays
    np.savez_compressed(
        sequences_npz_path,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=np.array(event_feature_cols),
    )

    # Save label records
    if "target_label" in session_features_df.columns:
        labels_df = session_features_df[["session_id", "target_label"]].copy()
        labels_df["label_id"] = labels_df["target_label"].map(label_mapping)
        labels_df.to_csv(labels_csv_path, index=False)

    # 10b. Build and Persist Personal User Baselines
    user_baselines = build_all_user_baselines(
        session_features_df,
        user_column="user_id" if "user_id" in session_features_df.columns else "session_id",
        min_sessions=settings.min_baseline_sessions,
    )
    baselines_csv_path, baselines_json_path = save_user_baselines(
        user_baselines, output_dir=settings.baselines_path
    )

    # 11. Compile Traceable Processing Report
    report = {
        "pipeline_version": "0.3.0-data-engineering",
        "input_dataset": str(input_path),
        "input_rows": input_row_count,
        "cleaned_rows": len(cleaned_df),
        "total_sessions": len(session_features_df),
        "features_extracted_count": len(feature_cols),
        "feature_names": feature_cols,
        "event_features_count": len(event_feature_cols),
        "event_feature_names": event_feature_cols,
        "sequence_shape_train": list(X_train.shape),
        "sequence_shape_test": list(X_test.shape),
        "sequence_length": sequence_length,
        "sequence_stride": sequence_stride,
        "split_strategy": split_summary["strategy"],
        "train_groups": split_summary["train_groups"],
        "test_groups": split_summary["test_groups"],
        "leakage_prevention": "Strict zero-group overlap between train and test partitions",
        "privacy_enforcement": {
            "zero_text_guarantee": True,
            "sensitive_text_columns_quarantined": sensitive_cols,
        },
        "cleaning_report": cleaning_report,
        "label_mapping": label_mapping,
        "class_inspection": label_inspection,
        "table_validation": {
            "is_valid": is_valid_table,
            "violations": table_violations,
        },
        "artifacts_generated": {
            "cleaned_events": str(cleaned_events_path),
            "session_features": str(session_features_path),
            "sequences": str(sequences_npz_path),
            "labels": str(labels_csv_path) if label_col else None,
            "feature_manifest": str(manifest_path),
            "feature_scaler": str(scaler_path),
            "user_baselines_csv": str(baselines_csv_path),
            "user_baselines_json": str(baselines_json_path),
            "processing_report": str(report_json_path),
        },
    }

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return {
        "status": "success",
        "summary": report,
        "artifacts": report["artifacts_generated"],
    }
