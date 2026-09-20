"""Unit tests for Dataset Registry and Lifecycle Management."""

from pathlib import Path
import pytest

from src.data_engineering.dataset_registry import (
    DEFAULT_SAMPLE_CONFIG,
    DatasetConfig,
    DatasetLifecycleStatus,
    DatasetRegistryEntry,
    evaluate_dataset_lifecycle,
    get_dataset_config,
    get_dataset_entry,
    list_dataset_entries,
    list_registered_datasets,
    register_dataset,
    register_dataset_entry,
)


def test_dataset_lifecycle_status_enum():
    """Verify standard lifecycle status enum values."""
    assert DatasetLifecycleStatus.AVAILABLE_LOCALLY.value == "AVAILABLE_LOCALLY"
    assert DatasetLifecycleStatus.ACCESS_REQUIRED.value == "ACCESS_REQUIRED"
    assert DatasetLifecycleStatus.NOT_FOUND.value == "NOT_FOUND"
    assert DatasetLifecycleStatus.INVALID.value == "INVALID"
    assert DatasetLifecycleStatus.READY_FOR_TRAINING.value == "READY_FOR_TRAINING"


def test_registered_dataset_entries_exist():
    """Verify built-in registration of research dataset entries."""
    entries = list_dataset_entries()
    assert "mobilestress" in entries
    assert "cmu_stress" in entries

    mob_entry = get_dataset_entry("mobilestress")
    assert mob_entry is not None
    assert mob_entry.name == "mobilestress"
    assert mob_entry.adapter_class == "MobileStressAdapter"
    assert mob_entry.raw_text_policy == "PROHIBITED_ZERO_RAW_TEXT"
    assert mob_entry.access_status == DatasetLifecycleStatus.ACCESS_REQUIRED

    cmu_entry = get_dataset_entry("cmu_stress")
    assert cmu_entry is not None
    assert cmu_entry.name == "cmu_stress"
    assert cmu_entry.adapter_class == "CMUStressAdapter"


def test_evaluate_dataset_lifecycle_states():
    """Verify lifecycle evaluation when dataset is missing from data/raw/."""
    # MobileStress is not locally downloaded in data/raw/
    status = evaluate_dataset_lifecycle("mobilestress")
    assert status in (DatasetLifecycleStatus.ACCESS_REQUIRED, DatasetLifecycleStatus.NOT_FOUND)

    # Unknown dataset
    unknown_status = evaluate_dataset_lifecycle("non_existent_dataset_xyz")
    assert unknown_status == DatasetLifecycleStatus.NOT_FOUND


def test_legacy_dataset_config_backward_compatibility():
    """Verify that legacy DatasetConfig registration and retrieval functions work seamlessly."""
    assert "sample_keystrokes" in list_registered_datasets()

    sample_cfg = get_dataset_config("sample_keystrokes")
    assert sample_cfg is not None
    assert sample_cfg.name == "sample_keystrokes"
    assert sample_cfg.is_synthetic is True

    # Custom registration
    custom_cfg = DatasetConfig(
        name="test_dataset_custom",
        file_path=Path("dummy.csv"),
        user_column="uid",
        session_column="sid",
    )
    register_dataset(custom_cfg)
    retrieved = get_dataset_config("test_dataset_custom")
    assert retrieved is not None
    assert retrieved.user_column == "uid"
