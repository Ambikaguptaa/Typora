"""Deep learning package for sequence modeling of typing dynamics."""

from src.deep_learning.baseline_classifier import (
    aggregate_sequence_features,
    train_baseline_classifier,
)
from src.deep_learning.data_split import (
    create_grouped_train_test_split,
    create_grouped_train_val_test_split,
    fit_feature_scaler,
    fit_sequence_scaler,
    load_scaler,
    save_scaler,
    split_sequences_by_group,
    transform_features,
    transform_sequence,
)
from src.deep_learning.evaluate import (
    EVALUATION_DISCLAIMER,
    assess_overfitting,
    evaluate_classification_model,
    save_confusion_matrix_artifacts,
)
from src.deep_learning.label_encoder import (
    decode_labels,
    encode_labels,
    inspect_labels,
    load_label_mapping,
    save_label_mapping,
)
try:
    from src.deep_learning.model import (
        ModelConfig,
        build_lstm_classifier,
        build_placeholder_model,
    )
except ImportError:
    ModelConfig = None
    def _model_stub(*args, **kwargs):
        raise ImportError("Model operations require the 'tensorflow' package. Install it with: pip install tensorflow")
    build_lstm_classifier = build_placeholder_model = _model_stub

try:
    from src.deep_learning.predict import (
        predict_sequence,
        predict_strain_sequence,
    )
except ImportError:
    def _predict_stub(*args, **kwargs):
        raise ImportError("Prediction requires the 'tensorflow' package. Install it with: pip install tensorflow")
    predict_sequence = predict_strain_sequence = _predict_stub

from src.deep_learning.preprocessing import (
    prepare_grouped_sequences,
    prepare_sequences,
)
try:
    from src.deep_learning.train import (
        inspect_real_dataset_availability,
        set_random_seed,
        train_lstm_model,
        train_pipeline_stub,
    )
except ImportError:
    def _train_stub(*args, **kwargs):
        raise ImportError("Model training requires the 'tensorflow' package. Install it with: pip install tensorflow")
    set_random_seed = train_lstm_model = train_pipeline_stub = _train_stub
    inspect_real_dataset_availability = lambda: (False, "tensorflow not installed", {})

from src.deep_learning.training_validation import validate_training_data

__all__ = [
    "validate_training_data",
    "prepare_sequences",
    "prepare_grouped_sequences",
    "create_grouped_train_test_split",
    "create_grouped_train_val_test_split",
    "split_sequences_by_group",
    "fit_feature_scaler",
    "fit_sequence_scaler",
    "transform_features",
    "transform_sequence",
    "save_scaler",
    "load_scaler",
    "encode_labels",
    "decode_labels",
    "inspect_labels",
    "save_label_mapping",
    "load_label_mapping",
    "ModelConfig",
    "build_lstm_classifier",
    "build_placeholder_model",
    "set_random_seed",
    "train_lstm_model",
    "train_pipeline_stub",
    "inspect_real_dataset_availability",
    "predict_sequence",
    "predict_strain_sequence",
    "evaluate_classification_model",
    "save_confusion_matrix_artifacts",
    "assess_overfitting",
    "EVALUATION_DISCLAIMER",
    "aggregate_sequence_features",
    "train_baseline_classifier",
]
