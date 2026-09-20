"""Synthetic Smoke Test Module for LSTM Training Pipeline.

CRITICAL NOTE:
This test is exclusively a technical pipeline smoke test verifying tensor flow,
training callbacks, artifact serialization, and inference on toy data.
IT MUST NEVER BE PRESENTED AS MODEL OR RESEARCH PERFORMANCE.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.deep_learning.model import ModelConfig
from src.deep_learning.predict import predict_sequence
from src.deep_learning.train import (
    inspect_real_dataset_availability,
    train_lstm_model,
)


def test_real_dataset_inspection_blocks_training_when_absent():
    """Verify that train_lstm_model blocks execution when no real dataset is in data/raw/."""
    # When allow_synthetic_smoke_test is False (production mode)
    result = train_lstm_model(
        X=np.random.randn(10, 10, 4),
        y=np.array(["A"] * 5 + ["B"] * 5),
        allow_synthetic_smoke_test=False,
    )

    assert result["status"] == "blocked"
    assert "REAL MODEL TRAINING BLOCKED" in result["message"]


def test_synthetic_smoke_training_pipeline_end_to_end(tmp_path: Path):
    """[SMOKE TEST ONLY] Technical verification of data -> scaler -> LSTM -> training -> prediction."""
    seq_len = 10
    n_feats = 4
    n_samples = 24

    # Create deterministic synthetic sequence tensor for code verification
    np.random.seed(42)
    X_toy = np.random.randn(n_samples, seq_len, n_feats).astype(np.float32)
    # 4 distinct users, each with 2 sessions (1 Calm, 1 Elevated_Strain), 3 samples per session (24 samples total)
    user_ids = []
    session_ids = []
    labels = []
    for u in range(4):
        for s in range(2):
            for _ in range(3):
                user_ids.append(f"user_{u:02d}")
                session_ids.append(f"sess_u{u}_s{s}")
                labels.append("Calm" if s == 0 else "Elevated_Strain")

    y_toy = np.array(labels)
    meta_toy = pd.DataFrame({"user_id": user_ids, "session_id": session_ids})
    feature_names = ["dwell_time", "flight_time", "pause_duration", "typing_speed"]

    # Short 2-epoch config for fast smoke testing
    cfg = ModelConfig(
        sequence_length=seq_len,
        num_features=n_feats,
        num_classes=2,
        lstm_units_1=16,
        lstm_units_2=8,
        dense_units=8,
        epochs=2,
        batch_size=4,
        random_seed=42,
    )

    models_dir = tmp_path / "models"
    eval_dir = tmp_path / "evaluation"

    # Run training in smoke test mode
    result = train_lstm_model(
        X=X_toy,
        y=y_toy,
        metadata_df=meta_toy,
        feature_names=feature_names,
        config=cfg,
        models_dir=models_dir,
        eval_dir=eval_dir,
        allow_synthetic_smoke_test=True,
    )

    # 1. Verification of training status
    assert result["status"] == "success"
    assert result["is_synthetic_smoke_test"] is True
    assert result["train_samples"] > 0
    assert result["num_classes"] == 2

    # 2. Verification of saved artifacts
    assert (models_dir / "lstm_model.keras").exists()
    assert (models_dir / "feature_scaler.pkl").exists()
    assert (models_dir / "label_mapping.json").exists()
    assert (models_dir / "feature_manifest.json").exists()
    assert (models_dir / "training_config.json").exists()
    assert (models_dir / "training_metrics.json").exists()
    assert (models_dir / "training_history.json").exists()
    assert (models_dir / "baseline_classifier.pkl").exists()
    assert (models_dir / "baseline_metrics.json").exists()

    # 3. Verification of evaluation metrics presence
    assert "accuracy" in result["train_metrics"]
    assert "macro_f1" in result["train_metrics"]
    assert "overfitting_report" in result

    # 4. Technical verification of inference using saved model artifacts
    unseen_seq = np.random.randn(seq_len, n_feats).astype(np.float32)
    inference_result = predict_sequence(unseen_seq, models_dir=models_dir)

    assert "predicted_class" in inference_result
    assert inference_result["predicted_class"] in ["Calm", "Elevated_Strain"]
    assert "class_probabilities" in inference_result
    assert pytest.approx(sum(inference_result["class_probabilities"].values()), abs=1e-2) == 1.0


def test_train_cli_blocks_without_real_dataset(monkeypatch, capsys):
    """Verify train_cli terminates and prints training blocked message when no real data exists."""
    from src.deep_learning.train import train_cli
    import sys

    # Emulate running python -m src.deep_learning.train with no flags
    monkeypatch.setattr(sys, "argv", ["train.py"])

    with pytest.raises(SystemExit) as exc_info:
        train_cli()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "TRAINING BLOCKED" in captured.out
    assert "REAL DATASET REQUIRED" in captured.out

