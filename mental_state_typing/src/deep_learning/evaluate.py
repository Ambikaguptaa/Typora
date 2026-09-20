"""Model Evaluation, Metrics Calculation, and Overfitting Diagnostics Module.

Calculates multi-dimensional classification metrics (accuracy, balanced accuracy,
macro/weighted F1, confusion matrix, per-class breakdown), saves machine-readable
and visual evaluation artifacts, and performs objective overfitting audits.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

# Academic disclaimer constant
EVALUATION_DISCLAIMER = (
    "Performance metrics reflect pattern classification against dataset annotations only. "
    "This system estimates behavioral typing dynamics and does not diagnose clinical or mental health conditions."
)


def evaluate_classification_model(
    y_true: Union[np.ndarray, List[int]],
    y_pred_probs: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics for model predictions.

    Args:
        y_true: 1D array of ground truth integer class IDs.
        y_pred_probs: 2D array of predicted class probabilities (N, num_classes) or (N, 1).
        class_names: List of string names corresponding to class indices 0..K-1.

    Returns:
        Dict[str, Any]: Comprehensive evaluation report dictionary.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_probs_arr = np.asarray(y_pred_probs, dtype=float)

    if len(y_true_arr) == 0:
        return {
            "samples": 0,
            "error": "Empty evaluation dataset.",
            "disclaimer": EVALUATION_DISCLAIMER,
        }

    # Determine predicted class IDs from probabilities
    if y_probs_arr.ndim == 1 or (y_probs_arr.ndim == 2 and y_probs_arr.shape[1] == 1):
        # Binary sigmoid probabilities
        probs_1d = y_probs_arr.flatten()
        y_pred = (probs_1d >= 0.5).astype(int)
    elif y_probs_arr.ndim == 2:
        y_pred = np.argmax(y_probs_arr, axis=1)
    else:
        raise ValueError(f"Unexpected prediction probability shape: {y_probs_arr.shape}")

    num_classes = len(class_names)
    labels_range = list(range(num_classes))

    # Primary Aggregate Metrics
    acc = float(accuracy_score(y_true_arr, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred))
    macro_p = float(precision_score(y_true_arr, y_pred, average="macro", zero_division=0))
    weighted_p = float(precision_score(y_true_arr, y_pred, average="weighted", zero_division=0))
    macro_r = float(recall_score(y_true_arr, y_pred, average="macro", zero_division=0))
    weighted_r = float(recall_score(y_true_arr, y_pred, average="weighted", zero_division=0))
    macro_f1 = float(f1_score(y_true_arr, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_arr, y_pred, average="weighted", zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_true_arr, y_pred, labels=labels_range)

    # Per-Class Precision, Recall, F1, Support
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true_arr,
        y_pred,
        labels=labels_range,
        zero_division=0,
    )

    per_class_metrics: Dict[str, Dict[str, Any]] = {}
    for idx, cls_name in enumerate(class_names):
        per_class_metrics[cls_name] = {
            "class_id": idx,
            "precision": round(float(precisions[idx]), 4),
            "recall": round(float(recalls[idx]), 4),
            "f1_score": round(float(f1s[idx]), 4),
            "support": int(supports[idx]),
        }

    return {
        "samples_evaluated": len(y_true_arr),
        "num_classes": num_classes,
        "class_names": class_names,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_precision": round(macro_p, 4),
        "weighted_precision": round(weighted_p, 4),
        "macro_recall": round(macro_r, 4),
        "weighted_recall": round(weighted_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class_metrics,
        "disclaimer": EVALUATION_DISCLAIMER,
    }


def save_confusion_matrix_artifacts(
    cm: np.ndarray,
    class_names: List[str],
    output_dir: Union[str, Path],
) -> Dict[str, str]:
    """Save machine-readable and visual confusion matrix artifacts.

    Generates:
    - confusion_matrix.json (counts and percentages)
    - confusion_matrix.csv (tabular format)
    - confusion_matrix.png (high-resolution styled heatmap)

    Args:
        cm: 2D confusion matrix array (num_classes, num_classes).
        class_names: List of class labels for row/column annotation.
        output_dir: Destination directory.

    Returns:
        Dict[str, str]: Paths of generated artifacts.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_file = out_path / "confusion_matrix.json"
    csv_file = out_path / "confusion_matrix.csv"
    png_file = out_path / "confusion_matrix.png"

    # 1. Machine-Readable JSON
    cm_row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_cm = np.where(cm_row_sums > 0, cm / cm_row_sums, 0.0)

    json_data = {
        "class_names": class_names,
        "raw_matrix": cm.tolist(),
        "normalized_matrix": [[round(float(val), 4) for val in row] for row in norm_cm],
        "total_samples": int(cm.sum()),
    }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    # 2. Tabular CSV
    df_cm = pd.DataFrame(cm, index=[f"True_{c}" for c in class_names], columns=[f"Pred_{c}" for c in class_names])
    df_cm.to_csv(csv_file)

    # 3. High-Resolution PNG via Matplotlib
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        fig.patch.set_facecolor("#111318")
        ax.set_facecolor("#1B1F26")

        cax = ax.matshow(cm, cmap="Blues", alpha=0.85)
        fig.colorbar(cax)

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="left", color="#E9EEF5", fontsize=9)
        ax.set_yticklabels(class_names, color="#E9EEF5", fontsize=9)

        ax.set_xlabel("Predicted Class", color="#35D6FF", fontsize=10, labelpad=10)
        ax.set_ylabel("True Class", color="#35D6FF", fontsize=10, labelpad=10)
        ax.set_title("Typing Dynamics Confusion Matrix", color="#E9EEF5", fontsize=11, pad=15)

        # Annotate numbers in cells
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = cm[i, j]
                ax.text(
                    j,
                    i,
                    str(val),
                    ha="center",
                    va="center",
                    color="white" if val > cm.max() / 2 else "#111318",
                    fontweight="bold",
                    fontsize=10,
                )

        plt.tight_layout()
        plt.savefig(png_file, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
    except Exception:
        # Fallback if graphical backend fails
        pass

    return {
        "json_path": str(json_file),
        "csv_path": str(csv_file),
        "png_path": str(png_file) if png_file.exists() else "",
    }


def assess_overfitting(
    train_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    test_metrics: Optional[Dict[str, Any]] = None,
    f1_divergence_threshold: float = 0.15,
) -> Dict[str, Any]:
    """Objectively compare train, validation, and test metrics to check for model overfitting.

    Reports statistical observations neutrally without diagnostic assumptions.

    Args:
        train_metrics: Metrics dictionary from training partition.
        val_metrics: Metrics dictionary from validation partition.
        test_metrics: Optional metrics dictionary from untouched test partition.
        f1_divergence_threshold: Difference between train and val F1 considered notable.

    Returns:
        Dict[str, Any]: Overfitting audit report.
    """
    train_acc = train_metrics.get("accuracy", 0.0)
    val_acc = val_metrics.get("accuracy", 0.0)
    train_f1 = train_metrics.get("macro_f1", 0.0)
    val_f1 = val_metrics.get("macro_f1", 0.0)

    f1_gap = train_f1 - val_f1
    acc_gap = train_acc - val_acc

    observations: List[str] = []
    is_overfitting = False

    if f1_gap > f1_divergence_threshold:
        is_overfitting = True
        observations.append(
            f"Training macro F1 ({train_f1:.3f}) exceeds validation macro F1 ({val_f1:.3f}) "
            f"by {f1_gap:.3f}, indicating potential overfitting to training user rhythms."
        )
    elif f1_gap < -0.10:
        observations.append(
            f"Validation macro F1 ({val_f1:.3f}) exceeds training macro F1 ({train_f1:.3f}), "
            "often observed with high dropout regularization or favorable validation group distribution."
        )
    else:
        observations.append(
            f"Train/Validation generalization gap is narrow (F1 difference: {f1_gap:.3f}), "
            "indicating stable motor cadence generalization across groups."
        )

    if test_metrics is not None:
        test_f1 = test_metrics.get("macro_f1", 0.0)
        val_test_gap = abs(val_f1 - test_f1)
        if val_test_gap > 0.15:
            observations.append(
                f"Noticeable divergence between validation F1 ({val_f1:.3f}) and untouched test F1 ({test_f1:.3f}) "
                f"of {val_test_gap:.3f}. Group-level user variance across independent participants may contribute."
            )

    return {
        "overfitting_detected": is_overfitting,
        "train_macro_f1": round(float(train_f1), 4),
        "val_macro_f1": round(float(val_f1), 4),
        "test_macro_f1": round(float(test_metrics.get("macro_f1", 0.0)), 4) if test_metrics else None,
        "f1_gap_train_val": round(float(f1_gap), 4),
        "accuracy_gap_train_val": round(float(acc_gap), 4),
        "observations": observations,
        "disclaimer": EVALUATION_DISCLAIMER,
    }
