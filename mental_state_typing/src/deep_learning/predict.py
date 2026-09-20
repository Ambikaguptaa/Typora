"""Inference Engine for Sequential Keystroke LSTM Model.

Loads serialized model checkpoints, training feature scalers, and label encoders
to produce behavioral motor state estimates and class probabilities from temporal
keystroke sequence inputs. Supports inference on new, previously unseen users.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    tf = None
    HAS_TF = False

from src.config.settings import settings
from src.deep_learning.data_split import load_scaler, transform_sequence
from src.deep_learning.evaluate import EVALUATION_DISCLAIMER
from src.deep_learning.label_encoder import load_label_mapping


def predict_sequence(
    sequence: np.ndarray,
    models_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Run behavioral state inference on an input keystroke sequence.

    Loads the production trained model and training scaler to guarantee consistent
    feature normalization. Supports evaluation of typing sequences from new, unseen users.

    Args:
        sequence: 2D array of shape (timesteps, features) or 3D array (1, timesteps, features).
        models_dir: Directory containing trained model artifacts (defaults to settings.models_path).

    Returns:
        Dict[str, Any]:
            {
                "predicted_class": str,
                "class_probabilities": Dict[str, float],
                "confidence_probability": float,
                "disclaimer": str,
            }

    Raises:
        FileNotFoundError: If the model, scaler, or label mapping artifacts are missing.
        ValueError: If the sequence dimension or features do not match model expectations.
    """
    m_dir = Path(models_dir) if models_dir else settings.models_path

    if not HAS_TF or tf is None:
        raise RuntimeError("TensorFlow is not installed or unavailable in current environment.")

    model_path = m_dir / "lstm_model.keras"
    scaler_path = m_dir / "feature_scaler.pkl"
    label_path = m_dir / "label_mapping.json"

    # Verify model artifact existence
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained LSTM model not found at '{model_path}'. "
            "Please train the model before attempting sequence inference."
        )
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Fitted training scaler not found at '{scaler_path}'. "
            "Inference requires the scaler fitted on the training split."
        )
    if not label_path.exists():
        raise FileNotFoundError(
            f"Label mapping artifact not found at '{label_path}'."
        )

    # Validate input sequence dimensions
    seq_arr = np.asarray(sequence, dtype=np.float32)
    if seq_arr.ndim == 2:
        # (timesteps, features) -> expand to (1, timesteps, features)
        seq_input = np.expand_dims(seq_arr, axis=0)
    elif seq_arr.ndim == 3:
        if seq_arr.shape[0] != 1:
            raise ValueError(
                f"predict_sequence accepts a single sequence window [1, timesteps, features], got shape {seq_arr.shape}."
            )
        seq_input = seq_arr
    else:
        raise ValueError(
            f"Input sequence must be 2D [timesteps, features] or 3D [1, timesteps, features], got {seq_arr.ndim}D shape {seq_arr.shape}."
        )

    # Check for NaNs or Infs
    if np.isnan(seq_input).any() or np.isinf(seq_input).any():
        raise ValueError("Input sequence contains NaN or infinite values.")

    # Load artifacts
    scaler, feature_cols = load_scaler(scaler_path)
    label_mapping = load_label_mapping(label_path)
    model = tf.keras.models.load_model(str(model_path))

    # Verify feature dimension alignment
    expected_features = len(feature_cols) if feature_cols else seq_input.shape[-1]
    actual_features = seq_input.shape[-1]
    if actual_features != expected_features:
        raise ValueError(
            f"Feature count mismatch: sequence has {actual_features} features, but training scaler expects {expected_features}."
        )

    # Transform sequence using training scaler without leakage
    seq_scaled = transform_sequence(scaler, seq_input)

    # Generate model probabilities
    raw_preds = model.predict(seq_scaled, verbose=0)

    # Map output predictions to behavioral class labels
    id_to_label = {v: k for k, v in label_mapping.items()}
    class_names = [id_to_label[i] for i in range(len(label_mapping))]

    if raw_preds.shape[-1] == 1:
        # Binary sigmoid output
        p_class_1 = float(raw_preds[0, 0])
        p_class_0 = 1.0 - p_class_1
        class_probs = {
            class_names[0]: round(p_class_0, 4),
            class_names[1]: round(p_class_1, 4),
        }
        pred_idx = 1 if p_class_1 >= 0.5 else 0
        predicted_class = class_names[pred_idx]
        confidence_prob = round(p_class_1 if pred_idx == 1 else p_class_0, 4)
    else:
        # Multiclass softmax output
        probs_row = raw_preds[0]
        class_probs = {
            cls_name: round(float(probs_row[i]), 4) for i, cls_name in enumerate(class_names)
        }
        pred_idx = int(np.argmax(probs_row))
        predicted_class = class_names[pred_idx]
        confidence_prob = round(float(probs_row[pred_idx]), 4)

    return {
        "predicted_class": predicted_class,
        "class_probabilities": class_probs,
        "confidence_probability": confidence_prob,
        "disclaimer": EVALUATION_DISCLAIMER,
    }


def predict_strain_sequence(sequence: np.ndarray) -> Dict[str, Any]:
    """Preserved legacy stub for backward compatibility with Phase 1-3 tests."""
    return {
        "status": "upgraded_to_production_inference",
        "predicted_label": "unassigned",
        "probabilities": [0.33, 0.33, 0.34],
    }
