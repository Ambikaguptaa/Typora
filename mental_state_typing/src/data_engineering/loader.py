"""Data loader module for keystroke timing datasets.

Responsible for ingesting raw timing metadata files (CSV/JSON).
Ensures no text or key character values are processed.
"""

from pathlib import Path
from typing import Union
import pandas as pd


def load_keystroke_dataset(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load a keystroke timing dataset from a CSV or JSON file.

    Expected columns in subsequent phases include:
    - press_time: timestamp (ms or s) when key was pressed down
    - release_time: timestamp (ms or s) when key was released

    Args:
        file_path: Path to dataset file.

    Returns:
        pd.DataFrame: Loaded timing data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file format is unsupported.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".json", ".jsonl"):
        df = pd.read_json(path)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. Supported formats: .csv, .json"
        )

    return df
