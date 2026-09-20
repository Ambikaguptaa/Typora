"""LSTM Sequential Model Architecture for Keystroke Dynamics.

Implements a standard, defensible recurrent neural network (stacked LSTM)
tailored for temporal typing feature sequences (dwell times, flight times, pauses, variability).
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import tensorflow as tf


@dataclass
class ModelConfig:
    """Hyperparameter and architectural configuration for the LSTM classifier."""

    sequence_length: int = 30
    num_features: int = 6
    num_classes: int = 3
    lstm_units_1: int = 64
    lstm_units_2: int = 32
    dense_units: int = 32
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 50
    random_seed: int = 42
    patience_early_stopping: int = 7
    patience_reduce_lr: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Instantiate configuration from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, output_path: Union[str, Path]) -> None:
        """Serialize configuration to a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, input_path: Union[str, Path]) -> "ModelConfig":
        """Deserialize configuration from a JSON file."""
        path = Path(input_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def build_lstm_classifier(
    config: Optional[ModelConfig] = None,
    sequence_length: Optional[int] = None,
    num_features: Optional[int] = None,
    num_classes: Optional[int] = None,
) -> tf.keras.Model:
    """Construct and compile an academic stacked LSTM classifier for behavioral sequences.

    Architecture:
        Input(shape=(sequence_length, num_features))
            ↓
        LSTM(lstm_units_1, return_sequences=True)
            ↓
        Dropout(dropout)
            ↓
        LSTM(lstm_units_2, return_sequences=False)
            ↓
        Dropout(dropout)
            ↓
        Dense(dense_units, activation="relu")
            ↓
        Dropout(dropout)
            ↓
        Output Layer:
            - If num_classes == 2: Dense(1, activation="sigmoid")
            - If num_classes > 2:  Dense(num_classes, activation="softmax")

    Args:
        config: ModelConfig dataclass instance (if None, default is used).
        sequence_length: Optional override for sequence_length.
        num_features: Optional override for num_features.
        num_classes: Optional override for num_classes.

    Returns:
        tf.keras.Model: Compiled Keras Sequential model.
    """
    cfg = config or ModelConfig()
    seq_len = sequence_length if sequence_length is not None else cfg.sequence_length
    n_feats = num_features if num_features is not None else cfg.num_features
    n_cls = num_classes if num_classes is not None else cfg.num_classes

    if seq_len <= 0 or n_feats <= 0:
        raise ValueError(f"sequence_length ({seq_len}) and num_features ({n_feats}) must be positive integers.")
    if n_cls < 2:
        raise ValueError(f"num_classes must be >= 2, got {n_cls}.")

    # Deterministic weight initialization
    tf.random.set_seed(cfg.random_seed)

    model = tf.keras.Sequential(name="keystroke_lstm_classifier")

    # Input layer
    model.add(tf.keras.layers.Input(shape=(seq_len, n_feats), name="temporal_input"))

    # Recurrent Layer 1
    model.add(
        tf.keras.layers.LSTM(
            units=cfg.lstm_units_1,
            return_sequences=True,
            name="lstm_layer_1",
        )
    )
    model.add(tf.keras.layers.Dropout(rate=cfg.dropout, name="dropout_1"))

    # Recurrent Layer 2
    model.add(
        tf.keras.layers.LSTM(
            units=cfg.lstm_units_2,
            return_sequences=False,
            name="lstm_layer_2",
        )
    )
    model.add(tf.keras.layers.Dropout(rate=cfg.dropout, name="dropout_2"))

    # Dense Projection
    model.add(
        tf.keras.layers.Dense(
            units=cfg.dense_units,
            activation="relu",
            name="dense_projection",
        )
    )
    model.add(tf.keras.layers.Dropout(rate=cfg.dropout, name="dropout_3"))

    # Output Layer & Loss Compilation
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)

    if n_cls == 2:
        # Binary classification
        model.add(
            tf.keras.layers.Dense(
                units=1,
                activation="sigmoid",
                name="binary_output",
            )
        )
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
    else:
        # Multiclass classification
        model.add(
            tf.keras.layers.Dense(
                units=n_cls,
                activation="softmax",
                name="multiclass_output",
            )
        )
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    return model


def build_placeholder_model(
    input_shape: tuple = (50, 4),
    num_classes: int = 3,
) -> Dict[str, Any]:
    """Stub definition for backward compatibility with legacy Phase 1-3 tests."""
    return {
        "model_type": "Recurrent_Sequential_Stub",
        "input_shape": input_shape,
        "num_classes": num_classes,
        "status": "upgraded_to_production_lstm",
    }
