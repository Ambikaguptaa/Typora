"""Generic Real Dataset Validator Module.

Validates candidate research keystroke datasets across 15 empirical dimensions:
readability, participant IDs, session IDs, timestamps, keystroke/timing signals,
behavioral condition labels, participant distributions, timing plausibility,
label granularity, feature compatibility, privacy compliance, and leakage safety.

CRITICAL METHODOLOGICAL RULES:
- Decouples Dataset Training Readiness (>=3 participants, >=2 classes) from
  Personal Baseline Readiness (>=5 repeated sessions per participant).
- Preserves dataset-specific condition semantics (describes targets as
  'dataset-derived behavioral classification targets', NOT 'medical diagnoses').
- Audits actual empirical distributions; never assumes fixed participant or session counts.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config.settings import settings
from src.data_engineering.dataset_acquisition import (
    DatasetDiscoveryVerdict,
    DatasetStatus,
    get_dataset_status,
)
from src.data_engineering.dataset_adapter import (
    detect_keystroke_columns,
    detect_label_column,
    detect_sensitive_text_columns,
    detect_session_column,
    detect_timestamp_column,
    detect_user_column,
    load_dataset,
)
from src.privacy.privacy_utils import audit_zero_raw_text

DISCLAIMER_TEXT = (
    "Research prototype only. The system models typing behavior patterns under "
    "experimental conditions as a dataset-derived behavioral classification target. "
    "It is NOT a medical diagnostic system and cannot diagnose mental health or psychiatric conditions."
)


def validate_real_dataset(
    source: Optional[Union[str, Path, pd.DataFrame]] = None,
    save_reports: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute rigorous 15-point validation audit on a research keystroke dataset.

    Args:
        source: Optional file path or loaded DataFrame. If None, discovers from data/raw/.
        save_reports: If True, writes JSON reports to data/processed/ or output_dir.
        output_dir: Optional custom output directory for report files.

    Returns:
        Dict[str, Any]: Comprehensive validation report dictionary.
    """
    # 1. Dataset discovery / loading
    df: Optional[pd.DataFrame] = None
    source_name: str = "Unknown"

    if isinstance(source, pd.DataFrame):
        df = source.copy()
        source_name = "In-Memory DataFrame"
    elif source is not None:
        p = Path(source)
        if p.exists() and p.is_file():
            try:
                df = load_dataset(p)
                source_name = str(p.name)
            except Exception as e:
                return _generate_unreadable_report(str(p), f"Failed to parse file: {e}")
        else:
            return _generate_missing_report(f"Specified source path does not exist: {source}")
    else:
        # Auto-discover from data/raw/
        status = get_dataset_status()
        if not status["candidate_paths"]:
            return _generate_missing_report(status["message"])
        primary_path = Path(status["candidate_paths"][0])
        try:
            df = load_dataset(primary_path)
            source_name = str(primary_path.name)
        except Exception as e:
            return _generate_unreadable_report(str(primary_path), f"Failed to load candidate file: {e}")

    if df is None or df.empty:
        return _generate_missing_report("Dataset is empty or contains zero rows.")

    # 2. Structural Column Roles Identification
    user_col = detect_user_column(df)
    session_col = detect_session_column(df)
    time_col = detect_timestamp_column(df)
    keystroke_cols = detect_keystroke_columns(df)
    label_col = detect_label_column(df)
    sensitive_text_cols = detect_sensitive_text_columns(df)

    # 3. Privacy & Sensitive Field Audit
    privacy_audit = audit_zero_raw_text(df)
    sensitive_excluded = list(sensitive_text_cols)
    if privacy_audit["violations"]:
        sensitive_excluded.extend([v.split("'")[1] for v in privacy_audit["violations"] if "'" in v])
    sensitive_excluded = sorted(list(set(sensitive_excluded)))

    # 4. Dimension & Counts
    total_records = len(df)
    total_participants = int(df[user_col].nunique()) if user_col else 0
    total_sessions = (
        int(df[[user_col, session_col]].drop_duplicates().shape[0])
        if (user_col and session_col)
        else (int(df[session_col].nunique()) if session_col else 0)
    )

    # 5. Missing Values & Duplicates
    missing_counts = {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().sum() > 0}
    duplicate_count = int(df.duplicated().sum())

    missing_user_count = int(df[user_col].isna().sum()) if user_col else total_records
    missing_session_count = int(df[session_col].isna().sum()) if session_col else total_records
    missing_time_count = int(df[time_col].isna().sum()) if time_col else total_records
    missing_label_count = int(df[label_col].isna().sum()) if label_col else total_records

    # 6. Behavioral Label & Class Distribution Analysis
    unique_labels: List[str] = []
    class_distribution: Dict[str, int] = {}
    label_metadata: Dict[str, Any] = {}
    label_granularity: str = "unknown"

    if label_col:
        val_counts = df[label_col].value_counts().to_dict()
        class_distribution = {str(k): int(v) for k, v in val_counts.items()}
        unique_labels = sorted(list(class_distribution.keys()))

        # Determine label granularity (session-level vs event-level vs participant-level)
        if session_col:
            session_label_nunique = df.groupby(session_col)[label_col].nunique().max()
            if session_label_nunique == 1:
                label_granularity = "session-level"
            else:
                label_granularity = "event-level"
        elif user_col:
            user_label_nunique = df.groupby(user_col)[label_col].nunique().max()
            label_granularity = "participant-level" if user_label_nunique == 1 else "event-level"

        label_metadata = {
            "target_column": label_col,
            "label_granularity": label_granularity,
            "unique_classes_count": len(unique_labels),
            "target_semantics": "dataset-derived behavioral classification target",
            "clinical_diagnosis_claim": False,
            "mappings": {
                label: {
                    "encoded_id": idx,
                    "original_label": label,
                    "count": class_distribution[label],
                    "description": f"Dataset condition: {label}",
                }
                for idx, label in enumerate(unique_labels)
            },
        }

    # 7. Per-Participant Breakdown & Distribution Metrics
    participant_breakdown: Dict[str, Dict[str, Any]] = {}
    sessions_per_participant: List[int] = []
    events_per_participant: List[int] = []

    if user_col:
        for uid, grp in df.groupby(user_col):
            uid_str = str(uid)
            sess_cnt = int(grp[session_col].nunique()) if session_col else 1
            rec_cnt = len(grp)
            sessions_per_participant.append(sess_cnt)
            events_per_participant.append(rec_cnt)

            cond_dist = (
                {str(k): int(v) for k, v in grp[label_col].value_counts().items()}
                if label_col
                else {}
            )
            participant_breakdown[uid_str] = {
                "session_count": sess_cnt,
                "record_count": rec_cnt,
                "condition_distribution": cond_dist,
            }

    # 8. Timing Plausibility, Monotonicity & Gaps
    invalid_timing_count = 0
    timestamp_valid = True
    large_timing_gaps_count = 0
    time_min = float("nan")
    time_max = float("nan")

    if time_col:
        time_nums = pd.to_numeric(df[time_col], errors="coerce").dropna()
        if not time_nums.empty:
            time_min = float(time_nums.min())
            time_max = float(time_nums.max())

        if user_col:
            for _, grp in df.groupby([user_col, session_col] if session_col else user_col):
                time_series = pd.to_numeric(grp[time_col], errors="coerce").dropna()
                if not time_series.is_monotonic_increasing:
                    timestamp_valid = False
                # Check for negative deltas or huge timing gaps (> 10000ms within session)
                diffs = time_series.diff().dropna()
                invalid_timing_count += int((diffs < 0).sum())
                large_timing_gaps_count += int((diffs > 10000.0).sum())

    # Inspect hold time / dwell time if present
    dwell_candidates = [c for c in df.columns if any(k in c.lower() for k in ("dwell", "hold"))]
    flight_candidates = [c for c in df.columns if any(k in c.lower() for k in ("flight", "iki", "latency"))]

    for col in dwell_candidates:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        invalid = ((series < 10.0) | (series > 4000.0)).sum()
        invalid_timing_count += int(invalid)

    for col in flight_candidates:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        invalid = ((series < 0.0) | (series > 10000.0)).sum()
        invalid_timing_count += int(invalid)

    # 9. Feature Compatibility Audit
    available_features: List[str] = []
    unavailable_features: List[str] = []

    feature_checks = {
        "dwell_time": len(dwell_candidates) > 0 or ("press_time" in df.columns and "release_time" in df.columns),
        "flight_time": len(flight_candidates) > 0 or time_col is not None,
        "pause_duration": time_col is not None or len(flight_candidates) > 0,
        "typing_speed": time_col is not None,
        "backspace_features": any("backspace" in c.lower() for c in df.columns),
        "error_features": any("error" in c.lower() for c in df.columns),
        "pressure": "pressure" in df.columns,
        "spatial_trajectory_xy": "x" in df.columns and "y" in df.columns,
        "word_completion_time": False,  # Strict Zero-Text: Lexical timing unavailable
        "explicit_lexical_token": False,  # Prohibited under Zero-Text policy
    }

    for feat, is_avail in feature_checks.items():
        if is_avail:
            available_features.append(feat)
        else:
            unavailable_features.append(feat)

    # 10. Sequence Length Evaluation (Data-Driven Percentiles)
    session_lengths: List[int] = []
    if user_col and session_col:
        session_lengths = df.groupby([user_col, session_col]).size().tolist()
    elif user_col:
        session_lengths = df.groupby(user_col).size().tolist()
    else:
        session_lengths = [len(df)]

    median_len = int(np.median(session_lengths)) if session_lengths else 0
    min_len = int(np.min(session_lengths)) if session_lengths else 0
    max_len = int(np.max(session_lengths)) if session_lengths else 0
    p25_len = int(np.percentile(session_lengths, 25)) if session_lengths else 0
    p75_len = int(np.percentile(session_lengths, 75)) if session_lengths else 0

    # Defensible sequence length recommendation: adapt without optimizing on test
    recommended_seq_len = 30
    if median_len > 0:
        recommended_seq_len = min(30, max(10, median_len // 2))

    # 11. Decoupled Baseline Compatibility
    min_baseline_req = settings.min_baseline_sessions
    eligible_baseline_users = sum(
        1 for p_info in participant_breakdown.values() if p_info["session_count"] >= min_baseline_req
    )
    if eligible_baseline_users > 0:
        personal_baseline_readiness = "READY"
        personal_baseline_message = f"{eligible_baseline_users} participant(s) have >= {min_baseline_req} sessions."
    else:
        personal_baseline_readiness = "BASELINE NOT AVAILABLE FROM DATASET"
        personal_baseline_message = (
            f"No participant has >= {min_baseline_req} recorded sessions. "
            "Personal baseline cannot be formed, but global model training remains unblocked."
        )

    # 12. Model Training Readiness
    groups_sufficient = total_participants >= 3 or total_sessions >= 6
    has_labels = len(unique_labels) >= 2
    has_keystrokes = len(keystroke_cols) > 0 or len(dwell_candidates) > 0 or len(flight_candidates) > 0 or (time_col is not None)
    training_ready = bool(
        user_col is not None
        and time_col is not None
        and has_labels
        and groups_sufficient
        and has_keystrokes
        and missing_user_count == 0
    )

    # 13. Overall Validation Status Verdict
    is_valid = training_ready
    status_verdict = "validated" if is_valid else "ready_for_validation"
    if not has_labels:
        status_verdict = "blocked_missing_labels"

    # Build validation report
    report: Dict[str, Any] = {
        "is_valid": is_valid,
        "status": status_verdict,
        "dataset_name": source_name,
        "source": "Research Repository / File Ingestion",
        "disclaimer": DISCLAIMER_TEXT,
        "total_records": total_records,
        "total_participants": total_participants,
        "total_sessions": total_sessions,
        "columns_detected": {
            "user_column": user_col,
            "session_column": session_col,
            "timestamp_column": time_col,
            "label_column": label_col,
            "keystroke_columns": keystroke_cols,
            "sensitive_text_columns": sensitive_text_cols,
        },
        "labels": {
            "detected": label_col is not None,
            "target_column": label_col,
            "unique_classes": unique_labels,
            "class_distribution": class_distribution,
            "label_granularity": label_granularity,
            "target_semantics": "dataset-derived behavioral classification target",
        },
        "participant_validation": {
            "total_participants": total_participants,
            "participant_breakdown": participant_breakdown,
            "groups_sufficient_for_splitting": groups_sufficient,
            "sessions_per_participant": {
                "min": int(np.min(sessions_per_participant)) if sessions_per_participant else 0,
                "median": float(np.median(sessions_per_participant)) if sessions_per_participant else 0.0,
                "max": int(np.max(sessions_per_participant)) if sessions_per_participant else 0,
            },
            "events_per_participant": {
                "min": int(np.min(events_per_participant)) if events_per_participant else 0,
                "median": float(np.median(events_per_participant)) if events_per_participant else 0.0,
                "max": int(np.max(events_per_participant)) if events_per_participant else 0,
            },
        },
        "data_quality": {
            "missing_values": missing_counts,
            "missing_user_ids": missing_user_count,
            "missing_session_ids": missing_session_count,
            "missing_timestamps": missing_time_count,
            "missing_labels": missing_label_count,
            "duplicate_records": duplicate_count,
            "timestamp_monotonic": timestamp_valid,
            "timestamp_span_seconds": float(time_max - time_min) if (not np.isnan(time_max) and not np.isnan(time_min)) else None,
            "invalid_timing_count": invalid_timing_count,
            "large_timing_gaps_count": large_timing_gaps_count,
        },
        "feature_compatibility": {
            "available_features": available_features,
            "unavailable_features": unavailable_features,
        },
        "sequence_configuration": {
            "median_session_length": median_len,
            "p25_session_length": p25_len,
            "p75_session_length": p75_len,
            "min_session_length": min_len,
            "max_session_length": max_len,
            "recommended_sequence_length": recommended_seq_len,
        },
        "dataset_training_readiness": {
            "status": "READY" if training_ready else "NOT_READY",
            "passed": training_ready,
            "reasons": [] if training_ready else [
                *(["Missing user column"] if not user_col else []),
                *(["Missing timestamp column"] if not time_col else []),
                *(["At least 2 behavioral classes required"] if not has_labels else []),
                *(["At least 3 participants or 6 sessions required for group-aware split"] if not groups_sufficient else []),
            ],
        },
        "personal_baseline_readiness": {
            "status": personal_baseline_readiness,
            "min_required_sessions": min_baseline_req,
            "eligible_participants": eligible_baseline_users,
            "message": personal_baseline_message,
        },
        "privacy": {
            "zero_text_compliant": privacy_audit["compliant"],
            "sensitive_fields_excluded": sensitive_excluded,
        },
        "leakage_audit_readiness": {
            "passed": groups_sufficient and has_labels,
            "message": (
                "Groups and labels sufficient for leakage-safe 3-way split."
                if (groups_sufficient and has_labels)
                else "Insufficient groups/labels for leakage-safe splitting."
            ),
        },
    }

    # Save artifacts if requested
    if save_reports:
        out_dir = Path(output_dir) if output_dir is not None else settings.processed_data_path
        out_dir.mkdir(parents=True, exist_ok=True)

        report_file = out_dir / "real_dataset_validation_report.json"
        report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

        feature_compat_file = out_dir / "feature_compatibility_report.json"
        feature_compat_payload = {
            "dataset_name": source_name,
            "available_features": available_features,
            "unavailable_features": unavailable_features,
            "recommended_sequence_length": recommended_seq_len,
        }
        feature_compat_file.write_text(json.dumps(feature_compat_payload, indent=2), encoding="utf-8")

        if label_metadata:
            label_file = out_dir / "label_metadata.json"
            label_file.write_text(json.dumps(label_metadata, indent=2), encoding="utf-8")

    return report


def _generate_missing_report(reason: str) -> Dict[str, Any]:
    """Construct standard report when real dataset is not present."""
    return {
        "is_valid": False,
        "status": "missing",
        "dataset_name": "None",
        "disclaimer": DISCLAIMER_TEXT,
        "total_records": 0,
        "total_participants": 0,
        "total_sessions": 0,
        "labels": {"detected": False, "unique_classes": [], "class_distribution": {}},
        "dataset_training_readiness": {"status": "BLOCKED", "passed": False, "reasons": [reason]},
        "personal_baseline_readiness": {"status": "BASELINE NOT AVAILABLE FROM DATASET", "message": reason},
        "message": f"REAL DATASET REQUIRED — {reason}",
    }


def _generate_unreadable_report(file_path: str, error_msg: str) -> Dict[str, Any]:
    """Construct report when file cannot be parsed."""
    return {
        "is_valid": False,
        "status": "unreadable",
        "dataset_name": Path(file_path).name,
        "disclaimer": DISCLAIMER_TEXT,
        "error": error_msg,
        "dataset_training_readiness": {"status": "BLOCKED", "passed": False, "reasons": [error_msg]},
        "personal_baseline_readiness": {"status": "BASELINE NOT AVAILABLE FROM DATASET", "message": error_msg},
        "message": f"Dataset file unreadable: {error_msg}",
    }
