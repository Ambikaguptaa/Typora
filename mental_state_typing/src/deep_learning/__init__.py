"""Deep learning package for sequence modeling of typing dynamics.

Note: In this foundational phase, model architectures (LSTM) and training
pipelines are stubbed and will be fully developed in Phase 4.
"""

from src.deep_learning.preprocessing import prepare_sequences
from src.deep_learning.model import build_placeholder_model
from src.deep_learning.train import train_pipeline_stub
from src.deep_learning.predict import predict_strain_sequence

__all__ = [
    "prepare_sequences",
    "build_placeholder_model",
    "train_pipeline_stub",
    "predict_strain_sequence",
]
