"""Preprocessing utilities for sequence models.

Prepares sliding-window keystroke timing sequences (hold times, flight times)
for sequential neural network inputs.
"""

from typing import List, Tuple
import numpy as np


def prepare_sequences(
    feature_matrix: np.ndarray,
    sequence_length: int = 50,
    step_size: int = 10,
) -> np.ndarray:
    """Slice 2D timing feature records into 3D sliding-window sequences for recurrent models.

    Args:
        feature_matrix: 2D array of shape (num_samples, num_features).
        sequence_length: Window size for temporal keystroke sequence.
        step_size: Stride between successive sliding windows.

    Returns:
        np.ndarray: 3D sequence array of shape (num_sequences, sequence_length, num_features).
    """
    if len(feature_matrix) < sequence_length:
        return np.empty((0, sequence_length, feature_matrix.shape[-1]))

    sequences: List[np.ndarray] = []
    num_samples = len(feature_matrix)

    for start_idx in range(0, num_samples - sequence_length + 1, step_size):
        end_idx = start_idx + sequence_length
        sequences.append(feature_matrix[start_idx:end_idx])

    return np.array(sequences)
