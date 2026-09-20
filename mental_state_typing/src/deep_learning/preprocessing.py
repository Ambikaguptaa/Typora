"""Preprocessing utilities for sequential deep learning models.

Prepares sliding-window keystroke timing sequences (dwell times, flight times,
variability features) for recurrent models (e.g. LSTM / GRU).

Guarantees:
- Strict chronological event ordering.
- Non-mixing of events across users or sessions (zero cross-session sequence bleed).
- Configurable sequence lengths and strides.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def prepare_sequences(
    feature_matrix: np.ndarray,
    sequence_length: int = 50,
    step_size: int = 10,
) -> np.ndarray:
    """Slice 2D timing feature records into 3D sliding-window sequences.

    Legacy compatibility function.

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


def prepare_grouped_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    group_cols: Optional[List[str]] = None,
    timestamp_col: Optional[str] = "timestamp",
    label_col: Optional[str] = "state",
    sequence_length: int = 30,
    sequence_stride: int = 10,
) -> Tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame]:
    """Slice keystroke event records into 3D temporal sequences respecting session boundaries.

    CRITICAL TEMPORAL INTEGRITY GUARANTEE:
    - Events are sorted chronologically.
    - Sequences NEVER cross user or session boundaries.
    - Each (user, session) is windowed independently.

    Args:
        df: Input DataFrame containing keystroke records.
        feature_cols: List of numeric feature column names to include in each timestep.
        group_cols: Grouping identifiers to partition sessions (e.g. ['user_id', 'session_id']).
        timestamp_col: Column used for chronological sorting.
        label_col: Target label column name (if available).
        sequence_length: Number of consecutive timesteps per sequence (default: 30).
        sequence_stride: Stride between successive sliding windows (default: 10).

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame]:
            - X: 3D sequence array of shape (num_sequences, sequence_length, num_features).
            - y: 1D array of target labels corresponding to each sequence (or None).
            - metadata_df: DataFrame tracking origin user_id, session_id, and timestamps for each sequence.
    """
    if group_cols is None:
        group_cols = [c for c in ["user_id", "session_id"] if c in df.columns]

    X_list: List[np.ndarray] = []
    y_list: List[Any] = []
    meta_records: List[Dict[str, Any]] = []

    # Sort dataset chronologically per group
    sort_keys = [c for c in group_cols + ([timestamp_col] if timestamp_col in df.columns else [])]
    df_sorted = df.sort_values(by=sort_keys).copy() if sort_keys else df.copy()

    # If no grouping columns exist, treat whole dataset as single sequence stream
    if not group_cols:
        groups = [(("all_data",), df_sorted)]
    else:
        groups = df_sorted.groupby(group_cols, sort=False)

    for group_key, group_df in groups:
        # User & session identifiers
        if isinstance(group_key, tuple):
            uid = str(group_key[0]) if len(group_key) > 0 else "unknown_user"
            sid = str(group_key[1]) if len(group_key) > 1 else "unknown_session"
        else:
            uid = str(group_key)
            sid = str(group_key)

        group_records = len(group_df)
        if group_records < sequence_length:
            # Insufficient events in this session to form a complete window
            continue

        # Extract numeric feature matrix for this group
        feature_matrix = group_df[feature_cols].to_numpy(dtype=np.float32)

        # Extract label for this group (take mode or first value)
        group_label = None
        if label_col and label_col in group_df.columns:
            group_label = group_df[label_col].iloc[0]

        # Slide windows across group_records
        for start_idx in range(0, group_records - sequence_length + 1, sequence_stride):
            end_idx = start_idx + sequence_length
            seq_window = feature_matrix[start_idx:end_idx]

            X_list.append(seq_window)
            if group_label is not None:
                y_list.append(group_label)

            # Metadata tracking
            meta_record: Dict[str, Any] = {
                "user_id": uid,
                "session_id": sid,
                "window_start_idx": start_idx,
                "window_end_idx": end_idx,
            }
            if timestamp_col and timestamp_col in group_df.columns:
                meta_record["start_timestamp"] = group_df[timestamp_col].iloc[start_idx]
                meta_record["end_timestamp"] = group_df[timestamp_col].iloc[end_idx - 1]

            meta_records.append(meta_record)

    num_features = len(feature_cols)
    if not X_list:
        X = np.empty((0, sequence_length, num_features), dtype=np.float32)
        y = np.array([]) if label_col and label_col in df.columns else None
        metadata_df = pd.DataFrame(columns=["user_id", "session_id", "window_start_idx", "window_end_idx"])
    else:
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list) if y_list else None
        metadata_df = pd.DataFrame(meta_records)

    return X, y, metadata_df
