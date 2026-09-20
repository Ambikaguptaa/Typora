"""Unit tests for inference engine and prediction module (predict.py)."""

from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf

from src.deep_learning.data_split import fit_sequence_scaler, save_scaler
from src.deep_learning.label_encoder import save_label_mapping
from src.deep_learning.model import ModelConfig, build_lstm_classifier
from src.deep_learning.predict import predict_sequence, predict_strain_sequence


def test_predict_sequence_missing_model_raises_error(tmp_path: Path):
    """Verify that attempting inference without a trained model raises FileNotFoundError."""
    dummy_seq = np.random.randn(20, 6).astype(np.float32)

    with pytest.raises(FileNotFoundError, match="Trained LSTM model not found"):
        predict_sequence(dummy_seq, models_dir=tmp_path)


def test_predict_sequence_invalid_dimensions(tmp_path: Path):
    """Verify that malformed sequences (1D, 4D, or multi-batch) raise ValueError."""
    # Create valid artifacts first
    cfg = ModelConfig(sequence_length=10, num_features=4, num_classes=2)
    model = build_lstm_classifier(cfg)
    model.save(str(tmp_path / "lstm_model.keras"))

    X_train = np.random.randn(5, 10, 4).astype(np.float32)
    scaler, _ = fit_sequence_scaler(X_train)
    save_scaler(scaler, ["f1", "f2", "f3", "f4"], tmp_path / "feature_scaler.pkl")
    save_label_mapping({"ClassA": 0, "ClassB": 1}, tmp_path / "label_mapping.json")

    # 1D sequence
    with pytest.raises(ValueError, match="Input sequence must be 2D"):
        predict_sequence(np.array([1.0, 2.0]), models_dir=tmp_path)

    # 3D sequence with batch size > 1
    with pytest.raises(ValueError, match="single sequence window"):
        predict_sequence(np.random.randn(3, 10, 4), models_dir=tmp_path)

    # Sequence containing NaNs
    nan_seq = np.random.randn(10, 4)
    nan_seq[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        predict_sequence(nan_seq, models_dir=tmp_path)


def test_predict_sequence_feature_mismatch(tmp_path: Path):
    """Verify that feature count mismatch is rejected."""
    cfg = ModelConfig(sequence_length=10, num_features=4, num_classes=2)
    model = build_lstm_classifier(cfg)
    model.save(str(tmp_path / "lstm_model.keras"))

    X_train = np.random.randn(5, 10, 4).astype(np.float32)
    scaler, _ = fit_sequence_scaler(X_train)
    save_scaler(scaler, ["f1", "f2", "f3", "f4"], tmp_path / "feature_scaler.pkl")
    save_label_mapping({"ClassA": 0, "ClassB": 1}, tmp_path / "label_mapping.json")

    # Input sequence has 6 features instead of 4
    wrong_features_seq = np.random.randn(10, 6).astype(np.float32)
    with pytest.raises(ValueError, match="Feature count mismatch"):
        predict_sequence(wrong_features_seq, models_dir=tmp_path)


def test_predict_sequence_successful_inference_and_unseen_user(tmp_path: Path):
    """Verify successful inference, probability sum ~= 1.0, and unseen user evaluation."""
    seq_len = 12
    n_feats = 4
    n_classes = 3
    classes = {"Calm": 0, "Fatigued": 1, "High_Workload": 2}

    cfg = ModelConfig(sequence_length=seq_len, num_features=n_feats, num_classes=n_classes)
    model = build_lstm_classifier(cfg)
    model.save(str(tmp_path / "lstm_model.keras"))

    X_train = np.random.randn(10, seq_len, n_feats).astype(np.float32)
    scaler, _ = fit_sequence_scaler(X_train)
    save_scaler(scaler, [f"feat_{i}" for i in range(n_feats)], tmp_path / "feature_scaler.pkl")
    save_label_mapping(classes, tmp_path / "label_mapping.json")

    # Unseen user's sequence (shape: 12, 4)
    unseen_user_seq = np.random.randn(seq_len, n_feats).astype(np.float32)
    result = predict_sequence(unseen_user_seq, models_dir=tmp_path)

    assert "predicted_class" in result
    assert result["predicted_class"] in classes
    assert "class_probabilities" in result
    assert len(result["class_probabilities"]) == n_classes

    # Sum of probabilities should be approx 1.0
    prob_sum = sum(result["class_probabilities"].values())
    assert pytest.approx(prob_sum, abs=1e-2) == 1.0

    assert "disclaimer" in result
    assert "does not diagnose" in result["disclaimer"]


def test_predict_strain_sequence_backward_compatibility():
    """Verify legacy stub function remains callable."""
    res = predict_strain_sequence(np.zeros((10, 4)))
    assert "probabilities" in res
    assert "predicted_label" in res
