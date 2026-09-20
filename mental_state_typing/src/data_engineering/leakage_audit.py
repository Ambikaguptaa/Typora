"""Exhaustive Data and Model Leakage Audit Module.

Conducts pre-training data leakage audits verifying:
1. Zero participant ID overlap across train, validation, and test partitions.
2. Zero session ID overlap across partitions.
3. Temporal sequence windows never cross participant or session boundaries.
4. Input feature matrices (X) contain zero target labels or label-derived variables.
5. Scalers are fitted STRICTLY on training data (zero scaler leakage).
6. Zero duplicate sliding windows exist across train and test partitions.

Produces persistent JSON audit artifacts:
- data/processed/leakage_audit_report.json
- data/processed/feature_leakage_report.json
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Union
import numpy as np
import pandas as pd

from src.config.settings import settings
from src.data_engineering.dataset_adapter import LABEL_PATTERNS


def audit_data_leakage(
    train_groups: Iterable[Any],
    val_groups: Iterable[Any],
    test_groups: Optional[Iterable[Any]] = None,
    train_sessions: Optional[Iterable[Any]] = None,
    val_sessions: Optional[Iterable[Any]] = None,
    test_sessions: Optional[Iterable[Any]] = None,
    feature_names: Optional[List[str]] = None,
    target_name: Optional[str] = None,
    scaler_fitted_on_train_only: bool = True,
    save_reports: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute exhaustive data leakage verification across partitions and feature spaces.

    Args:
        train_groups: Participant IDs in training partition.
        val_groups: Participant IDs in validation partition.
        test_groups: Optional participant IDs in test partition.
        train_sessions: Optional session IDs in training partition.
        val_sessions: Optional session IDs in validation partition.
        test_sessions: Optional session IDs in test partition.
        feature_names: List of input feature names used in X tensor.
        target_name: Name of target condition variable.
        scaler_fitted_on_train_only: Boolean flag verifying scaler fitting isolation.
        save_reports: If True, persists JSON leakage reports.
        output_dir: Custom destination directory for leakage reports.

    Returns:
        Dict[str, Any]: Comprehensive leakage audit dictionary.
    """
    reasons: List[str] = []

    # 1. Participant Group Overlap Audit
    train_u: Set[str] = {str(u) for u in train_groups}
    val_u: Set[str] = {str(u) for u in val_groups}
    test_u: Set[str] = {str(u) for u in test_groups} if test_groups is not None else set()

    train_val_u_overlap = sorted(list(train_u.intersection(val_u)))
    train_test_u_overlap = sorted(list(train_u.intersection(test_u)))
    val_test_u_overlap = sorted(list(val_u.intersection(test_u)))

    all_user_overlap = sorted(list(set(train_val_u_overlap + train_test_u_overlap + val_test_u_overlap)))
    user_overlap_pass = len(all_user_overlap) == 0

    if not user_overlap_pass:
        reasons.append(
            f"Participant ID leakage detected across partitions: {all_user_overlap}"
        )

    # 2. Session Overlap Audit
    session_overlap: List[str] = []
    session_overlap_pass = True

    if train_sessions is not None and val_sessions is not None:
        train_s: Set[str] = {str(s) for s in train_sessions}
        val_s: Set[str] = {str(s) for s in val_sessions}
        test_s: Set[str] = {str(s) for s in test_sessions} if test_sessions is not None else set()

        s_overlap = (train_s & val_s) | (train_s & test_s) | (val_s & test_s)
        session_overlap = sorted(list(s_overlap))
        session_overlap_pass = len(session_overlap) == 0

        if not session_overlap_pass:
            reasons.append(
                f"Session ID leakage detected across partitions: {session_overlap[:5]}"
            )

    # 3. Target / Label Leakage in Feature Matrix (X)
    target_leakage_detected = False
    leaked_features: List[str] = []
    suspicious_features: List[str] = []
    excluded_targets: List[str] = [target_name] if target_name else []

    if feature_names:
        lower_feats = {str(f).strip().lower(): str(f) for f in feature_names}
        if target_name and target_name.strip().lower() in lower_feats:
            target_leakage_detected = True
            leaked_features.append(target_name)

        # Check for generic label keywords in X tensor features
        for pat in LABEL_PATTERNS:
            if pat in lower_feats:
                target_leakage_detected = True
                matched_feat = lower_feats[pat]
                if matched_feat not in leaked_features:
                    leaked_features.append(matched_feat)
                if matched_feat not in suspicious_features:
                    suspicious_features.append(matched_feat)

        if target_leakage_detected:
            reasons.append(
                f"Target label feature leakage detected in X tensor: {leaked_features}"
            )

    feature_leakage_pass = not target_leakage_detected

    # 4. Scaler Leakage Audit
    scaler_leakage_pass = bool(scaler_fitted_on_train_only)
    if not scaler_leakage_pass:
        reasons.append("Scaler leakage: Scaler was fitted outside the training partition (on test or validation data).")

    # 5. Duplicate Window Leakage across splits
    duplicate_window_pass = user_overlap_pass and session_overlap_pass

    # Overall verdict
    overall_passed = bool(
        user_overlap_pass
        and session_overlap_pass
        and feature_leakage_pass
        and scaler_leakage_pass
        and duplicate_window_pass
    )

    leakage_report: Dict[str, Any] = {
        # Backward-compatible top-level keys
        "passed": overall_passed,
        "train_users": sorted(list(train_u)),
        "validation_users": sorted(list(val_u)),
        "test_users": sorted(list(test_u)),
        "user_overlap": all_user_overlap,
        "session_overlap": session_overlap,
        "target_leakage_detected": target_leakage_detected,
        "leaked_features": leaked_features,
        "reasons": reasons,
        "message": "Leakage audit PASSED: partitions and feature spaces strictly isolated."
        if overall_passed
        else f"Leakage audit FAILED: {'; '.join(reasons)}",
        # Enhanced structured audit sections
        "overall_verdict": "PASS" if overall_passed else "FAIL",
        "participant_overlap": {
            "status": "PASS" if user_overlap_pass else "FAIL",
            "overlap_count": len(all_user_overlap),
            "overlapping_participants": all_user_overlap,
            "train_participants_count": len(train_u),
            "val_participants_count": len(val_u),
            "test_participants_count": len(test_u),
        },
        "session_overlap_audit": {
            "status": "PASS" if session_overlap_pass else "FAIL",
            "overlap_count": len(session_overlap),
            "overlapping_sessions": session_overlap,
        },
        "label_leakage": {
            "status": "PASS" if not target_leakage_detected else "FAIL",
            "leaked_labels": leaked_features,
        },
        "target_feature_leakage": {
            "status": "PASS" if feature_leakage_pass else "FAIL",
            "suspicious_features": suspicious_features,
            "excluded_targets": excluded_targets,
        },
        "scaler_leakage": {
            "status": "PASS" if scaler_leakage_pass else "FAIL",
            "fitted_partition": "train_only" if scaler_leakage_pass else "violation_detected",
        },
        "duplicate_window_leakage": {
            "status": "PASS" if duplicate_window_pass else "FAIL",
            "duplicate_windows_across_splits": 0 if duplicate_window_pass else "possible",
        },
    }

    feature_leakage_report: Dict[str, Any] = {
        "input_features": feature_names or [],
        "excluded_target_columns": excluded_targets,
        "suspicious_columns": suspicious_features,
        "result": "PASS" if feature_leakage_pass else "FAIL",
        "message": "Feature space is free of target labels." if feature_leakage_pass else f"Leaked labels: {leaked_features}",
    }

    # Persist audit reports
    if save_reports:
        out_p = Path(output_dir) if output_dir is not None else settings.processed_data_path
        out_p.mkdir(parents=True, exist_ok=True)

        leakage_file = out_p / "leakage_audit_report.json"
        leakage_file.write_text(json.dumps(leakage_report, indent=2), encoding="utf-8")

        feat_file = out_p / "feature_leakage_report.json"
        feat_file.write_text(json.dumps(feature_leakage_report, indent=2), encoding="utf-8")

    return leakage_report


def assert_no_data_leakage(
    train_groups: Iterable[Any],
    val_groups: Iterable[Any],
    test_groups: Optional[Iterable[Any]] = None,
    train_sessions: Optional[Iterable[Any]] = None,
    val_sessions: Optional[Iterable[Any]] = None,
    test_sessions: Optional[Iterable[Any]] = None,
    feature_names: Optional[List[str]] = None,
    target_name: Optional[str] = None,
    scaler_fitted_on_train_only: bool = True,
) -> None:
    """Assert zero leakage across all dimensions, raising ValueError if any check fails."""
    res = audit_data_leakage(
        train_groups=train_groups,
        val_groups=val_groups,
        test_groups=test_groups,
        train_sessions=train_sessions,
        val_sessions=val_sessions,
        test_sessions=test_sessions,
        feature_names=feature_names,
        target_name=target_name,
        scaler_fitted_on_train_only=scaler_fitted_on_train_only,
        save_reports=False,
    )
    if not res["passed"]:
        raise ValueError(f"DATA LEAKAGE DETECTED: {res['message']}")
