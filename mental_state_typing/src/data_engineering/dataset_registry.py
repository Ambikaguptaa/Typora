"""Dataset Registry Module.

Provides configuration structures and a registry for supported keystroke datasets.
Allows swapping datasets easily without rewriting the downstream ML/engineering pipelines.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.config.settings import settings


@dataclass
class DatasetConfig:
    """Configuration mapping for a specific keystroke dataset."""

    name: str
    file_path: Path
    user_column: Optional[str] = None
    session_column: Optional[str] = None
    timestamp_column: Optional[str] = None
    label_column: Optional[str] = None
    feature_columns: List[str] = field(default_factory=list)
    description: str = ""
    is_synthetic: bool = False


# In-memory registry mapping dataset names to configurations
_REGISTRY: Dict[str, DatasetConfig] = {}


def register_dataset(config: DatasetConfig) -> None:
    """Register a new dataset configuration."""
    _REGISTRY[config.name.lower()] = config


def get_dataset_config(name: str) -> Optional[DatasetConfig]:
    """Retrieve a registered dataset configuration by name."""
    return _REGISTRY.get(name.lower())


def list_registered_datasets() -> List[str]:
    """Return names of all currently registered datasets."""
    return list(_REGISTRY.keys())


# Pre-register default sample synthetic dataset
DEFAULT_SAMPLE_CONFIG = DatasetConfig(
    name="sample_keystrokes",
    file_path=settings.sample_data_path / "sample_keystrokes.csv",
    user_column="user_id",
    session_column="session_id",
    timestamp_column="timestamp",
    label_column="state",
    feature_columns=[
        "dwell_time",
        "flight_time",
        "pause_duration",
        "backspace",
        "error_flag",
        "typing_speed",
    ],
    description="Synthetic benchmark dataset for development, validation, and pipeline tests.",
    is_synthetic=True,
)

register_dataset(DEFAULT_SAMPLE_CONFIG)
