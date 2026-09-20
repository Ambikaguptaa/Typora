"""Dataset Registry Module.

Provides configuration structures, lifecycle tracking, and a registry for supported keystroke datasets.
Allows swapping datasets easily without rewriting the downstream ML/engineering pipelines.
Enforces explicit distinction between:
- AVAILABLE_LOCALLY
- ACCESS_REQUIRED
- NOT_FOUND
- INVALID
- READY_FOR_TRAINING
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config.settings import settings


class DatasetLifecycleStatus(str, Enum):
    """Lifecycle status states for keystroke research datasets."""

    AVAILABLE_LOCALLY = "AVAILABLE_LOCALLY"
    ACCESS_REQUIRED = "ACCESS_REQUIRED"
    NOT_FOUND = "NOT_FOUND"
    INVALID = "INVALID"
    READY_FOR_TRAINING = "READY_FOR_TRAINING"


@dataclass
class DatasetRegistryEntry:
    """Comprehensive registry descriptor for a keystroke dynamics dataset."""

    name: str
    adapter_class: str
    source: str
    access_status: DatasetLifecycleStatus
    expected_schema: Dict[str, Any]
    label_semantics: Dict[str, str]
    participant_field: str
    session_field: str
    timestamp_field: str
    feature_support: List[str]
    raw_text_policy: str = "PROHIBITED_ZERO_RAW_TEXT"
    citation: str = ""
    file_path: Optional[Path] = None
    is_synthetic: bool = False
    description: str = ""


@dataclass
class DatasetConfig:
    """Legacy/compatibility configuration mapping for a specific keystroke dataset."""

    name: str
    file_path: Path
    user_column: Optional[str] = None
    session_column: Optional[str] = None
    timestamp_column: Optional[str] = None
    label_column: Optional[str] = None
    feature_columns: List[str] = field(default_factory=list)
    description: str = ""
    is_synthetic: bool = False


# In-memory registries
_REGISTRY: Dict[str, DatasetConfig] = {}
_ENTRY_REGISTRY: Dict[str, DatasetRegistryEntry] = {}


def register_dataset(config: DatasetConfig) -> None:
    """Register a legacy/compatibility dataset configuration."""
    _REGISTRY[config.name.lower()] = config


def get_dataset_config(name: str) -> Optional[DatasetConfig]:
    """Retrieve a registered dataset configuration by name."""
    return _REGISTRY.get(name.lower())


def list_registered_datasets() -> List[str]:
    """Return names of all currently registered datasets."""
    return list(_REGISTRY.keys())


def register_dataset_entry(entry: DatasetRegistryEntry) -> None:
    """Register an enhanced dataset descriptor."""
    _ENTRY_REGISTRY[entry.name.lower()] = entry


def get_dataset_entry(name: str) -> Optional[DatasetRegistryEntry]:
    """Retrieve an enhanced dataset descriptor by name."""
    return _ENTRY_REGISTRY.get(name.lower())


def list_dataset_entries() -> List[str]:
    """Return names of all enhanced registered datasets."""
    return list(_ENTRY_REGISTRY.keys())


def evaluate_dataset_lifecycle(name: str) -> DatasetLifecycleStatus:
    """Determine the current local lifecycle status of a registered dataset.

    Inspects local file availability and manifest status.
    """
    entry = get_dataset_entry(name)
    if not entry:
        return DatasetLifecycleStatus.NOT_FOUND

    if entry.is_synthetic:
        if entry.file_path and entry.file_path.exists():
            return DatasetLifecycleStatus.AVAILABLE_LOCALLY
        return DatasetLifecycleStatus.NOT_FOUND

    # Real research datasets: check if file exists in data/raw/
    raw_dir = settings.raw_data_path
    if not raw_dir.exists():
        return DatasetLifecycleStatus.ACCESS_REQUIRED

    # Check for specific candidate file
    if entry.file_path and entry.file_path.exists():
        return DatasetLifecycleStatus.AVAILABLE_LOCALLY

    # Check if files matching name exist in raw_dir
    matching_files = [f for f in raw_dir.iterdir() if f.is_file() and entry.name in f.name.lower()]
    if matching_files:
        return DatasetLifecycleStatus.AVAILABLE_LOCALLY

    return entry.access_status


# Pre-register default sample synthetic dataset in legacy registry
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

# Register enhanced entries for real and reference datasets
MOBILESTRESS_ENTRY = DatasetRegistryEntry(
    name="mobilestress",
    adapter_class="MobileStressAdapter",
    source="MobileStress Academic Research Protocol",
    access_status=DatasetLifecycleStatus.ACCESS_REQUIRED,
    expected_schema={
        "participant_id": "string",
        "session_id": "string",
        "timestamp": "float",
        "event_type": "string",
        "key_identifier": "string (hashed)",
        "condition": "string",
    },
    label_semantics={
        "E": "neutral (standard task condition)",
        "H": "stressed (induced cognitive/temporal stress)",
    },
    participant_field="participant_id",
    session_field="session_id",
    timestamp_field="timestamp",
    feature_support=["dwell_time", "flight_time", "pause_duration", "typing_speed", "backspace"],
    raw_text_policy="PROHIBITED_ZERO_RAW_TEXT",
    citation=(
        "MobileStress: Smartphone Keystroke Dynamics Under Induced Cognitive and Temporal Stress. "
        "Academic Behavioral Study."
    ),
    file_path=settings.raw_data_path / "mobilestress.csv",
    is_synthetic=False,
    description="Smartphone keystroke dynamics under induced stress conditions.",
)
register_dataset_entry(MOBILESTRESS_ENTRY)

CMU_STRESS_ENTRY = DatasetRegistryEntry(
    name="cmu_stress",
    adapter_class="CMUStressAdapter",
    source="Carnegie Mellon University (Killourhy & Maxion)",
    access_status=DatasetLifecycleStatus.ACCESS_REQUIRED,
    expected_schema={
        "subject": "string",
        "sessionIndex": "int",
        "rep": "int",
        "hold_times": "float",
        "key_latencies": "float",
    },
    label_semantics={
        "baseline": "neutral (normal typing authentication session)",
    },
    participant_field="subject",
    session_field="sessionIndex",
    timestamp_field="timestamp",
    feature_support=["dwell_time", "flight_time", "latency"],
    raw_text_policy="PROHIBITED_ZERO_RAW_TEXT",
    citation=(
        "Killourhy, K. S., & Maxion, R. A. (2009). "
        "Comparing anomaly-detection algorithms for keystroke dynamics. IEEE/IFIP DSN."
    ),
    file_path=settings.raw_data_path / "cmu_keystroke.csv",
    is_synthetic=False,
    description="Carnegie Mellon University Keystroke Dynamics Benchmark.",
)
register_dataset_entry(CMU_STRESS_ENTRY)
