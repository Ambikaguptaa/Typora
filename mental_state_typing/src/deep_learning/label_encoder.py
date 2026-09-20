"""Categorical Label Encoding and Class Balance Module.

Inspects target behavioral / psychological state labels, maps categorical classes
to numeric indices without hardcoding specific class names, and serializes
the mapping for future inference pipelines.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def inspect_labels(
    series: pd.Series,
) -> Dict[str, Any]:
    """Inspect unique values, missingness, and frequency breakdown of target labels.

    Args:
        series: Target label series.

    Returns:
        Dict[str, Any]: Class inspection report.
    """
    valid = series.dropna()
    total_valid = len(valid)
    unique_vals = list(valid.unique())
    counts = valid.value_counts().to_dict()
    proportions = {
        str(k): round(float(v) / max(total_valid, 1), 4) for k, v in counts.items()
    }

    return {
        "unique_labels": [str(x) for x in unique_vals],
        "num_classes": len(unique_vals),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "class_proportions": proportions,
        "null_count": int(series.isnull().sum()),
        "is_numeric": pd.api.types.is_numeric_dtype(series),
    }


def encode_labels(
    label_series: pd.Series,
    custom_mapping: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Deterministically map categorical labels to zero-indexed integer IDs.

    Args:
        label_series: Series of categorical or integer labels.
        custom_mapping: Optional predefined mapping dictionary.

    Returns:
        Tuple[np.ndarray, Dict[str, int]]: (encoded_array, label_to_id_mapping).
    """
    valid = label_series.dropna()
    if custom_mapping is None:
        # Sort uniquely for deterministic, reproducible assignment
        sorted_classes = sorted([str(x) for x in valid.unique()])
        mapping = {cls_name: idx for idx, cls_name in enumerate(sorted_classes)}
    else:
        mapping = custom_mapping

    # Map series values
    encoded_list: List[int] = []
    for val in label_series:
        if pd.isnull(val):
            encoded_list.append(-1)
        else:
            val_str = str(val)
            if val_str in mapping:
                encoded_list.append(mapping[val_str])
            else:
                encoded_list.append(-1)

    return np.array(encoded_list, dtype=np.int32), mapping


def decode_labels(
    encoded_array: np.ndarray,
    mapping: Dict[str, int],
) -> List[str]:
    """Decode integer IDs back to original behavioral label strings."""
    id_to_label = {v: k for k, v in mapping.items()}
    return [id_to_label.get(int(idx), "unassigned") for idx in encoded_array]


def save_label_mapping(
    mapping: Dict[str, int],
    output_path: Union[str, Path],
) -> None:
    """Serialize the label mapping as a JSON artifact."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)


def load_label_mapping(
    input_path: Union[str, Path],
) -> Dict[str, int]:
    """Load a serialized label mapping."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Label mapping not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
