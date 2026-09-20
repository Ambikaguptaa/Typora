"""Unit tests for LSTM model architecture and configuration (model.py)."""

from pathlib import Path
import pytest
import tensorflow as tf

from src.deep_learning.model import (
    ModelConfig,
    build_lstm_classifier,
    build_placeholder_model,
)


def test_model_config_defaults_and_serialization(tmp_path: Path):
    """Verify ModelConfig default values, dict conversion, and JSON persistence."""
    cfg = ModelConfig()
    assert cfg.sequence_length == 30
    assert cfg.num_features == 6
    assert cfg.num_classes == 3
    assert cfg.lstm_units_1 == 64
    assert cfg.lstm_units_2 == 32
    assert cfg.dropout == 0.2
    assert cfg.learning_rate == 0.001

    # Serialization test
    cfg_file = tmp_path / "test_config.json"
    cfg.save(cfg_file)
    assert cfg_file.exists()

    loaded_cfg = ModelConfig.load(cfg_file)
    assert loaded_cfg.sequence_length == cfg.sequence_length
    assert loaded_cfg.lstm_units_1 == cfg.lstm_units_1
    assert loaded_cfg.dropout == cfg.dropout


def test_build_lstm_classifier_binary():
    """Verify LSTM architecture for binary classification (1 output unit, sigmoid)."""
    cfg = ModelConfig(sequence_length=25, num_features=4, num_classes=2)
    model = build_lstm_classifier(cfg)

    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None, 25, 4)
    assert model.output_shape == (None, 1)

    # Verify output layer activation is sigmoid
    output_layer = model.layers[-1]
    assert output_layer.activation.__name__ == "sigmoid"
    assert model.loss == "binary_crossentropy"


def test_build_lstm_classifier_multiclass():
    """Verify LSTM architecture for multiclass classification (K units, softmax)."""
    cfg = ModelConfig(sequence_length=20, num_features=5, num_classes=3)
    model = build_lstm_classifier(cfg)

    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None, 20, 5)
    assert model.output_shape == (None, 3)

    # Verify output layer activation is softmax
    output_layer = model.layers[-1]
    assert output_layer.activation.__name__ == "softmax"
    assert model.loss == "sparse_categorical_crossentropy"


def test_build_lstm_classifier_layer_structure():
    """Verify the 2-layer LSTM + Dense projection architecture."""
    cfg = ModelConfig(sequence_length=15, num_features=6, num_classes=4)
    model = build_lstm_classifier(cfg)

    # Inspect layer types
    layer_types = [type(layer).__name__ for layer in model.layers]
    lstm_count = layer_types.count("LSTM")
    dropout_count = layer_types.count("Dropout")
    dense_count = layer_types.count("Dense")

    assert lstm_count == 2, f"Expected 2 LSTM layers, found {lstm_count}"
    assert dropout_count == 3, f"Expected 3 Dropout layers, found {dropout_count}"
    assert dense_count == 2, f"Expected 2 Dense layers (1 projection + 1 output), found {dense_count}"

    # Verify first LSTM returns sequences, second does not
    lstm_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.LSTM)]
    assert lstm_layers[0].return_sequences is True
    assert lstm_layers[1].return_sequences is False


def test_build_lstm_classifier_invalid_inputs():
    """Verify that invalid dimensions or class counts raise ValueError."""
    with pytest.raises(ValueError, match="positive integers"):
        build_lstm_classifier(sequence_length=0, num_features=5, num_classes=3)

    with pytest.raises(ValueError, match="positive integers"):
        build_lstm_classifier(sequence_length=20, num_features=-1, num_classes=3)

    with pytest.raises(ValueError, match="num_classes must be >= 2"):
        build_lstm_classifier(sequence_length=20, num_features=4, num_classes=1)


def test_build_placeholder_model_backward_compatibility():
    """Verify legacy placeholder model remains callable for Phase 1-3 compatibility."""
    stub = build_placeholder_model(input_shape=(30, 6), num_classes=3)
    assert stub["input_shape"] == (30, 6)
    assert stub["num_classes"] == 3
