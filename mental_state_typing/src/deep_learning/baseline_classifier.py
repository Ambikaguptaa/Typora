"""Non-Deep-Learning Baseline Classifier Comparison Module.

Trains a lightweight traditional machine learning classifier (Random Forest / Logistic Regression)
on temporally aggregated features using the exact same group-partitioned splits to establish
a benchmark against which the LSTM's temporal representation learning can be evaluated.
"""

import json
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Union
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.deep_learning.evaluate import EVALUATION_DISCLAIMER, evaluate_classification_model


def aggregate_sequence_features(X: np.ndarray) -> np.ndarray:
    """Aggregate 3D temporal sequence arrays into 2D tabular features.

    Computes mean, standard deviation, median, min, and max across the time axis.

    Args:
        X: 3D sequence array of shape (samples, timesteps, features).

    Returns:
        np.ndarray: 2D feature matrix of shape (samples, features * 5).
    """
    if X.ndim != 3:
        raise ValueError(f"X must be 3D [samples, timesteps, features], got shape {X.shape}.")

    # Aggregate along time axis (axis=1)
    mean_feat = np.mean(X, axis=1)
    std_feat = np.std(X, axis=1)
    median_feat = np.median(X, axis=1)
    min_feat = np.min(X, axis=1)
    max_feat = np.max(X, axis=1)

    # Concatenate aggregates into a single 2D feature vector per sequence
    agg_2d = np.hstack([mean_feat, std_feat, median_feat, min_feat, max_feat])
    return np.nan_to_num(agg_2d, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def train_baseline_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    model_type: str = "random_forest",
    random_state: int = 42,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Train and evaluate a non-deep-learning baseline model on aggregated features.

    Uses the exact same leakage-safe data partitions as the LSTM.

    Args:
        X_train: 3D training sequence tensor.
        y_train: 1D training label IDs.
        X_val: Optional 3D validation sequence tensor.
        y_val: Optional 1D validation label IDs.
        X_test: 3D test sequence tensor.
        y_test: 1D test label IDs.
        class_names: List of string class names.
        model_type: 'random_forest' or 'logistic_regression'.
        random_state: Random seed.
        output_dir: Directory to save baseline_classifier.pkl and baseline_metrics.json.

    Returns:
        Dict[str, Any]: Baseline model evaluation summary.
    """
    if len(X_train) == 0 or len(y_train) == 0:
        return {
            "status": "insufficient_data",
            "message": "Training set is empty; baseline classifier cannot be fitted.",
            "disclaimer": EVALUATION_DISCLAIMER,
        }

    # Verify minimum class representation in training set
    unique_train_classes = np.unique(y_train)
    if len(unique_train_classes) < 2:
        return {
            "status": "insufficient_data",
            "message": f"Training set contains only {len(unique_train_classes)} class; minimum 2 required.",
            "disclaimer": EVALUATION_DISCLAIMER,
        }

    # Aggregate features across time axis
    X_train_agg = aggregate_sequence_features(X_train)
    X_test_agg = aggregate_sequence_features(X_test)
    X_val_agg = aggregate_sequence_features(X_val) if (X_val is not None and len(X_val) > 0) else None

    # Instantiate model
    if model_type.lower() == "logistic_regression":
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            class_weight="balanced",
            random_state=random_state,
        )

    # Fit on training data ONLY
    model.fit(X_train_agg, y_train)

    # Evaluate on Train, Val, and Test
    train_probs = model.predict_proba(X_train_agg)
    train_metrics = evaluate_classification_model(y_train, train_probs, class_names)

    val_metrics = None
    if X_val_agg is not None and y_val is not None and len(y_val) > 0:
        val_probs = model.predict_proba(X_val_agg)
        val_metrics = evaluate_classification_model(y_val, val_probs, class_names)

    test_probs = model.predict_proba(X_test_agg)
    test_metrics = evaluate_classification_model(y_test, test_probs, class_names)

    results: Dict[str, Any] = {
        "status": "success",
        "model_type": type(model).__name__,
        "train_metrics": {
            "accuracy": train_metrics["accuracy"],
            "macro_f1": train_metrics["macro_f1"],
            "weighted_f1": train_metrics["weighted_f1"],
        },
        "val_metrics": {
            "accuracy": val_metrics["accuracy"],
            "macro_f1": val_metrics["macro_f1"],
            "weighted_f1": val_metrics["weighted_f1"],
        } if val_metrics else None,
        "test_metrics": test_metrics,
        "disclaimer": EVALUATION_DISCLAIMER,
    }

    # Persist artifacts if output_dir specified
    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        model_file = out_path / "baseline_classifier.pkl"
        metrics_file = out_path / "baseline_metrics.json"

        with open(model_file, "wb") as f:
            pickle.dump({"model": model, "model_type": model_type, "class_names": class_names}, f)

        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        results["artifacts"] = {
            "baseline_model": str(model_file),
            "baseline_metrics": str(metrics_file),
        }

    return results
