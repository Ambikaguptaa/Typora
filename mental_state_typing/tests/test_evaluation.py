"""Unit tests for model evaluation, confusion matrix generation, and overfitting checks (evaluate.py)."""

from pathlib import Path
import numpy as np
import pytest

from src.deep_learning.evaluate import (
    assess_overfitting,
    evaluate_classification_model,
    save_confusion_matrix_artifacts,
)


def test_evaluate_classification_model_multiclass():
    """Verify multiclass metrics calculation and report structure."""
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    # Probabilities matching y_true
    y_probs = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.7, 0.2],
        [0.1, 0.2, 0.7],
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
        [0.6, 0.2, 0.2],
        [0.2, 0.6, 0.2],
        [0.2, 0.2, 0.6],
    ])
    class_names = ["Calm", "Fatigued", "High_Workload"]

    metrics = evaluate_classification_model(y_true, y_probs, class_names)

    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0
    assert metrics["num_classes"] == 3
    assert len(metrics["per_class_metrics"]) == 3
    assert "Calm" in metrics["per_class_metrics"]
    assert "disclaimer" in metrics


def test_evaluate_classification_model_binary():
    """Verify binary metrics calculation with single sigmoid probability output."""
    y_true = np.array([0, 1, 0, 1])
    y_probs = np.array([[0.1], [0.9], [0.2], [0.85]])
    class_names = ["Baseline", "Elevated_Strain"]

    metrics = evaluate_classification_model(y_true, y_probs, class_names)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert len(metrics["per_class_metrics"]) == 2


def test_save_confusion_matrix_artifacts(tmp_path: Path):
    """Verify saving confusion matrix JSON, CSV, and PNG artifacts."""
    cm = np.array([[10, 2], [1, 15]])
    class_names = ["ClassA", "ClassB"]

    artifacts = save_confusion_matrix_artifacts(cm, class_names, tmp_path)

    assert Path(artifacts["json_path"]).exists()
    assert Path(artifacts["csv_path"]).exists()
    assert Path(artifacts["png_path"]).exists()


def test_assess_overfitting_detected():
    """Verify that a large train-validation divergence is flagged."""
    train_metrics = {"accuracy": 0.98, "macro_f1": 0.97}
    val_metrics = {"accuracy": 0.65, "macro_f1": 0.60}

    report = assess_overfitting(train_metrics, val_metrics)

    assert report["overfitting_detected"] is True
    assert report["f1_gap_train_val"] > 0.15
    assert len(report["observations"]) > 0


def test_assess_overfitting_narrow_gap():
    """Verify that narrow generalization gap is reported as stable."""
    train_metrics = {"accuracy": 0.85, "macro_f1": 0.84}
    val_metrics = {"accuracy": 0.83, "macro_f1": 0.82}

    report = assess_overfitting(train_metrics, val_metrics)

    assert report["overfitting_detected"] is False
    assert any("stable" in obs or "narrow" in obs for obs in report["observations"])
