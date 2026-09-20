"""Training Readiness and Data Integrity Validation Module.

Performs rigorous pre-flight checks on temporal sequence tensors and behavioral labels
prior to model compilation and training to prevent numerical errors, dimension mismatches,
target class degeneracy, and data leakage.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


def validate_training_data(
    X: Optional[np.ndarray],
    y: Optional[Union[np.ndarray, List[Any], pd.Series]],
    metadata_df: Optional[pd.DataFrame] = None,
    min_samples_per_class: int = 2,
    expected_timesteps: Optional[int] = None,
    expected_features: Optional[int] = None,
) -> Dict[str, Any]:
    """Validate 3D feature sequences and target labels for LSTM training readiness.

    Verifies:
    1. Tensors exist and are non-empty.
    2. X is numeric with no NaN or infinite values.
    3. X has exact 3D shape [samples, timesteps, features].
    4. Sample count in X matches label count in y.
    5. Target y contains at least two distinct behavioral classes.
    6. Each class has at least min_samples_per_class observations.
    7. Sequence length and feature dimensions are consistent.
    8. User/session grouping metadata exists when provided for group-aware splitting.

    Args:
        X: 3D numpy array of shape (samples, timesteps, features).
        y: 1D array, list, or Series of class labels.
        metadata_df: Optional DataFrame with sequence metadata (user_id, session_id).
        min_samples_per_class: Minimum required instances per unique class (default: 2).
        expected_timesteps: Optional expected sequence length for consistency checks.
        expected_features: Optional expected feature count for consistency checks.

    Returns:
        Dict[str, Any]: Structured readiness report.
    """
    blocking_errors: List[str] = []
    warnings: List[str] = []

    samples = 0
    timesteps = 0
    features = 0
    classes: List[str] = []
    class_distribution: Dict[str, int] = {}
    users_count = 0

    # 1. Existence and Type of X
    if X is None:
        blocking_errors.append("Input tensor X is None.")
    elif not isinstance(X, np.ndarray):
        blocking_errors.append(f"Input tensor X must be a numpy.ndarray, got {type(X).__name__}.")
    else:
        if X.size == 0:
            blocking_errors.append("Input tensor X is empty (0 elements).")
        elif X.ndim != 3:
            blocking_errors.append(
                f"Input tensor X must be 3-dimensional [samples, timesteps, features], got {X.ndim}D shape {X.shape}."
            )
        else:
            samples, timesteps, features = X.shape

            # Numeric verification
            if not np.issubdtype(X.dtype, np.number):
                blocking_errors.append(f"Input tensor X must have numeric dtype, got {X.dtype}.")
            else:
                # NaN and Infinite verification
                nan_count = int(np.isnan(X).sum())
                if nan_count > 0:
                    blocking_errors.append(f"Input tensor X contains {nan_count} NaN values.")

                inf_count = int(np.isinf(X).sum())
                if inf_count > 0:
                    blocking_errors.append(f"Input tensor X contains {inf_count} infinite values.")

            # Dimensional consistency checks
            if expected_timesteps is not None and timesteps != expected_timesteps:
                blocking_errors.append(
                    f"Sequence length mismatch: expected {expected_timesteps} timesteps, got {timesteps}."
                )
            if expected_features is not None and features != expected_features:
                blocking_errors.append(
                    f"Feature count mismatch: expected {expected_features} features, got {features}."
                )

    # 2. Existence and Verification of y
    if y is None:
        blocking_errors.append("Target labels y is None.")
    else:
        if isinstance(y, pd.Series):
            y_arr = y.to_numpy()
        elif isinstance(y, list):
            y_arr = np.array(y)
        elif isinstance(y, np.ndarray):
            y_arr = y
        else:
            y_arr = np.array(y)

        if len(y_arr) == 0:
            blocking_errors.append("Target labels y is empty (0 elements).")
        else:
            # Check length alignment with X
            if X is not None and isinstance(X, np.ndarray) and X.ndim == 3:
                if len(y_arr) != samples:
                    blocking_errors.append(
                        f"Sample count mismatch: X has {samples} sequences, but y has {len(y_arr)} labels."
                    )

            # Class count & distribution
            # Filter nulls if any
            null_labels = pd.isnull(y_arr).sum()
            if null_labels > 0:
                blocking_errors.append(f"Target labels y contains {null_labels} null/missing labels.")

            non_null_y = [str(val) for val in y_arr if not pd.isnull(val)]
            unique_labels = sorted(list(set(non_null_y)))
            classes = unique_labels
            for cls in unique_labels:
                class_distribution[cls] = non_null_y.count(cls)

            if len(unique_labels) < 2:
                blocking_errors.append(
                    f"At least two distinct classes are required for classification, found {len(unique_labels)}: {unique_labels}."
                )

            # Per-class observation thresholds
            for cls, count in class_distribution.items():
                if count < min_samples_per_class:
                    blocking_errors.append(
                        f"Class '{cls}' has only {count} observations (minimum required: {min_samples_per_class})."
                    )

            # Class imbalance warning
            if len(unique_labels) >= 2 and len(non_null_y) > 0:
                max_c = max(class_distribution.values())
                min_c = min(class_distribution.values())
                ratio = max_c / max(min_c, 1)
                if ratio > 3.0:
                    warnings.append(
                        f"Class imbalance detected: majority-to-minority ratio is {ratio:.1f}:1."
                    )

    # 3. Metadata & Grouping Verification
    if metadata_df is not None:
        if not isinstance(metadata_df, pd.DataFrame):
            warnings.append("Provided metadata is not a pandas DataFrame.")
        else:
            if "user_id" in metadata_df.columns:
                users_count = int(metadata_df["user_id"].nunique())
                if users_count < 2:
                    warnings.append(
                        f"Only {users_count} user found in metadata. Grouped user-level splitting requires >= 2 users; session-level fallback will be needed."
                    )
            else:
                warnings.append("Metadata DataFrame lacks 'user_id' column for user-grouped splitting.")

            if len(metadata_df) != samples and samples > 0:
                warnings.append(
                    f"Metadata row count ({len(metadata_df)}) does not match sequence count ({samples})."
                )
    else:
        warnings.append("No metadata_df provided. User-level group-aware splitting cannot be verified.")

    ready = len(blocking_errors) == 0

    return {
        "ready": ready,
        "samples": samples,
        "timesteps": timesteps,
        "features": features,
        "classes": classes,
        "class_distribution": class_distribution,
        "users": users_count,
        "warnings": warnings,
        "blocking_errors": blocking_errors,
    }
