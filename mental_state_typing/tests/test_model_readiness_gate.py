"""Unit tests for the 9 production model readiness criteria and honest gating."""

import json
import pytest
from src.integration.behavioral_engine import BehavioralEngine
from src.integration.pipeline_state import SessionType


def test_model_readiness_gate_missing_model_file(tmp_path):
    """Verify gate reports not ready when lstm_model.keras is missing."""
    engine = BehavioralEngine(
        models_dir=tmp_path / "models",
        baselines_dir=tmp_path / "baselines",
        assessments_dir=tmp_path / "assessments",
    )
    is_ready, reason, details = engine.check_model_readiness()
    assert not is_ready
    assert "No verified production LSTM model file exists" in reason
    assert details["missing"] == "lstm_model.keras"


def test_model_readiness_gate_missing_metadata(tmp_path):
    """Verify gate reports not ready when training_metadata.json is missing."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "lstm_model.keras").write_text("dummy_model", encoding="utf-8")

    engine = BehavioralEngine(models_dir=models_dir)
    is_ready, reason, details = engine.check_model_readiness()
    assert not is_ready
    assert "metadata file is missing" in reason


def test_model_readiness_gate_smoke_test_model_rejected(tmp_path):
    """Verify gate rejects smoke test / development models (is_production_model=False)."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "lstm_model.keras").write_text("dummy_model", encoding="utf-8")

    meta = {
        "is_production_model": False,
        "trained_on_real_dataset": False,
    }
    with open(models_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    engine = BehavioralEngine(models_dir=models_dir)
    is_ready, reason, _ = engine.check_model_readiness()
    assert not is_ready
    assert "development/smoke artifact" in reason


def test_model_readiness_gate_not_trained_on_real_dataset(tmp_path):
    """Verify gate rejects models not trained on approved real dataset."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "lstm_model.keras").write_text("dummy_model", encoding="utf-8")

    meta = {
        "is_production_model": True,
        "trained_on_real_dataset": False,  # Not trained on real dataset!
    }
    with open(models_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    engine = BehavioralEngine(models_dir=models_dir)
    is_ready, reason, _ = engine.check_model_readiness()
    assert not is_ready
    assert "not trained on an approved real research dataset" in reason


def test_model_readiness_gate_manifest_dimension_mismatch(tmp_path):
    """Verify gate rejects models when sequence_length or num_features do not match."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "lstm_model.keras").write_text("dummy_model", encoding="utf-8")

    meta = {
        "is_production_model": True,
        "trained_on_real_dataset": True,
    }
    with open(models_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    # Manifest with wrong sequence length (50 instead of 30)
    manifest = {
        "sequence_length": 50,
        "num_features": 6,
    }
    with open(models_dir / "feature_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    engine = BehavioralEngine(models_dir=models_dir, sequence_length=30)
    is_ready, reason, _ = engine.check_model_readiness()
    assert not is_ready
    assert "Sequence length mismatch" in reason
