"""Data Quality Gate and Input Integrity Checks for Behavioral Assessment.

Validates sequence observation counts, numerical validity (NaN/Inf), feature manifest
compatibility, and subsystem readiness before generating behavioral assessments.
Enforces explicit assessment states to prevent misleading or degraded outputs.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


# Explicit, non-ambiguous assessment states
STATE_READY = "ready"
STATE_PREDICTION_AVAILABLE = "prediction_available"
STATE_BASELINE_AVAILABLE = "baseline_available"
STATE_MODEL_UNAVAILABLE = "model_unavailable"
STATE_BASELINE_UNAVAILABLE = "baseline_unavailable"
STATE_INSUFFICIENT_DATA = "insufficient_data"
STATE_NOT_READY = "not_ready"

VALID_ASSESSMENT_STATES = {
    STATE_READY,
    STATE_PREDICTION_AVAILABLE,
    STATE_BASELINE_AVAILABLE,
    STATE_MODEL_UNAVAILABLE,
    STATE_BASELINE_UNAVAILABLE,
    STATE_INSUFFICIENT_DATA,
    STATE_NOT_READY,
}


def validate_assessment_data(
    keystroke_count: int,
    sequence: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    expected_features: Optional[List[str]] = None,
    expected_sequence_length: Optional[int] = None,
    model_available: bool = False,
    baseline_status: str = "unavailable",
    min_keystrokes: int = 5,
) -> Dict[str, Any]:
    """Execute pre-assessment validation to determine system readiness and state.

    Args:
        keystroke_count: Total valid keystrokes in the session window.
        sequence: Optional 2D or 3D timing feature sequence array.
        feature_names: Feature names present in the incoming sequence.
        expected_features: Feature list required by the trained model / manifest.
        expected_sequence_length: Required sequence window length (timesteps).
        model_available: Flag indicating whether trained model is loaded.
        baseline_status: 'available', 'insufficient_history', or 'unavailable'.
        min_keystrokes: Minimum required keystrokes (default: 5).

    Returns:
        Dict[str, Any]: Structured quality check report.
    """
    blocking_reasons: List[str] = []
    warnings: List[str] = []

    # 1. Observation Count Verification
    if keystroke_count < min_keystrokes:
        blocking_reasons.append(
            f"Insufficient keystroke observations: {keystroke_count} recorded (minimum required: {min_keystrokes})."
        )

    # 2. Sequence Tensor Verification (if provided)
    if sequence is not None:
        if not isinstance(sequence, np.ndarray):
            blocking_reasons.append(
                f"Sequence must be a numpy.ndarray, got {type(sequence).__name__}."
            )
        elif sequence.size == 0:
            blocking_reasons.append("Input sequence is empty (0 elements).")
        else:
            # Check numerical validity (no NaNs or Infs)
            nan_count = int(np.isnan(sequence).sum())
            if nan_count > 0:
                blocking_reasons.append(f"Input sequence contains {nan_count} NaN values.")

            inf_count = int(np.isinf(sequence).sum())
            if inf_count > 0:
                blocking_reasons.append(f"Input sequence contains {inf_count} infinite values.")

            # Shape verification: 2D (timesteps, features) or 3D (1, timesteps, features)
            if sequence.ndim == 2:
                actual_timesteps, actual_feats = sequence.shape
            elif sequence.ndim == 3:
                if sequence.shape[0] != 1:
                    blocking_reasons.append(
                        f"Assessment processes a single sequence window [1, timesteps, features], got shape {sequence.shape}."
                    )
                actual_timesteps, actual_feats = sequence.shape[1], sequence.shape[2]
            else:
                blocking_reasons.append(
                    f"Sequence must be 2D [timesteps, features] or 3D [1, timesteps, features], got {sequence.ndim}D shape {sequence.shape}."
                )
                actual_timesteps, actual_feats = 0, 0

            # Sequence length compatibility
            if expected_sequence_length is not None and actual_timesteps > 0:
                if actual_timesteps != expected_sequence_length:
                    blocking_reasons.append(
                        f"Sequence length mismatch: expected {expected_sequence_length} timesteps, got {actual_timesteps}."
                    )

            # Feature dimension compatibility
            if expected_features is not None and actual_feats > 0:
                if actual_feats != len(expected_features):
                    blocking_reasons.append(
                        f"Feature count mismatch: expected {len(expected_features)} features, got {actual_feats}."
                    )

    # 3. Feature Order and Manifest Alignment (if feature names provided)
    if feature_names is not None and expected_features is not None:
        if feature_names != expected_features:
            missing_from_incoming = set(expected_features) - set(feature_names)
            if missing_from_incoming:
                blocking_reasons.append(
                    f"Incoming features lack required model inputs: {sorted(list(missing_from_incoming))}."
                )
            else:
                warnings.append(
                    "Incoming feature order differs from model manifest; reordering is required."
                )

    # 4. Determine Explicit Assessment State
    has_blocking_errors = len(blocking_reasons) > 0
    baseline_is_available = (baseline_status == "available")

    if has_blocking_errors:
        if keystroke_count < min_keystrokes:
            assessment_state = STATE_INSUFFICIENT_DATA
        else:
            assessment_state = STATE_NOT_READY
        assessment_ready = False
    else:
        # Determine operational state based on component availability
        if model_available and baseline_is_available:
            assessment_state = STATE_READY
            assessment_ready = True
        elif model_available and not baseline_is_available:
            assessment_state = STATE_PREDICTION_AVAILABLE
            assessment_ready = True
            warnings.append(
                f"Personal baseline status is '{baseline_status}'. Model prediction available without personal baseline comparison."
            )
        elif not model_available and baseline_is_available:
            assessment_state = STATE_BASELINE_AVAILABLE
            assessment_ready = True
            warnings.append(
                "Trained deep learning model is not available. Personal baseline deviation available without model classification."
            )
        else:
            # Neither model nor baseline available
            assessment_state = STATE_MODEL_UNAVAILABLE
            assessment_ready = False
            warnings.append(
                "Neither trained model nor personal baseline is available for assessment."
            )

    return {
        "assessment_ready": assessment_ready,
        "assessment_state": assessment_state,
        "warnings": warnings,
        "blocking_reasons": blocking_reasons,
        "keystroke_count": keystroke_count,
        "model_available": model_available,
        "baseline_status": baseline_status,
    }
