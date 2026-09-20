"""Unit tests for Pluggable Dataset Adapter Framework."""

from pathlib import Path
import pandas as pd
import pytest

from src.data_engineering.adapters.base_adapter import BaseDatasetAdapter
from src.data_engineering.adapters.cmu_adapter import CMUStressAdapter
from src.data_engineering.adapters.mobilestress_adapter import MobileStressAdapter


def test_base_dataset_adapter_cannot_be_instantiated_directly():
    """Verify that BaseDatasetAdapter enforces abstract methods and cannot be instantiated."""
    with pytest.raises(TypeError):
        BaseDatasetAdapter()  # type: ignore


def test_mobilestress_adapter_implements_interface():
    """Verify that MobileStressAdapter implements the complete BaseDatasetAdapter contract."""
    adapter = MobileStressAdapter()
    assert isinstance(adapter, BaseDatasetAdapter)

    # Check metadata
    meta = adapter.get_metadata()
    assert meta["adapter_name"] == "MobileStressAdapter"
    assert meta["canonical_schema_compliant"] is True
    assert meta["access_status"] == "ACCESS_REQUIRED"

    # Check supported features
    features = adapter.get_supported_features()
    assert "dwell_time" in features
    assert "flight_time" in features
    assert "typing_speed" in features

    # Check label mapping
    mapping = adapter.get_label_mapping()
    assert mapping["E"] == "neutral"
    assert mapping["H"] == "stressed"


def test_mobilestress_adapter_detect():
    """Verify detection heuristics for MobileStress formats."""
    adapter = MobileStressAdapter()

    matching_df = pd.DataFrame({
        "participant_id": ["p1"],
        "timestamp": [100.0],
        "condition": ["E"],
        "key": ["a"],
    })
    assert adapter.detect(matching_df) is True

    unrelated_df = pd.DataFrame({
        "temperature": [25.0],
        "humidity": [60.0],
    })
    assert adapter.detect(unrelated_df) is False


def test_mobilestress_adapter_normalize_conforms_to_canonical():
    """Verify that normalize() outputs DataFrame conforming to Canonical Event Schema."""
    adapter = MobileStressAdapter()
    raw_df = pd.DataFrame({
        "sub_id": ["user_101", "user_101"],
        "session": ["s1", "s1"],
        "down_time": [1000.0, 1200.0],
        "type": ["down", "down"],
        "key": ["k", "l"],
        "state": ["E", "H"],
    })

    canonical_df = adapter.normalize(raw_df)
    assert "participant_id" in canonical_df.columns
    assert "session_id" in canonical_df.columns
    assert "key_identifier" in canonical_df.columns
    assert "condition" in canonical_df.columns
    assert canonical_df["condition"].iloc[0] == "neutral"
    assert canonical_df["condition"].iloc[1] == "stressed"
    assert canonical_df["key_identifier"].iloc[0].startswith("k_")
    assert "key" not in canonical_df.columns  # Stripped raw text


def test_cmu_adapter_specification_stub():
    """Verify that CMUStressAdapter serves as a specification stub without column fabrication."""
    adapter = CMUStressAdapter()
    assert isinstance(adapter, BaseDatasetAdapter)

    meta = adapter.get_metadata()
    assert meta["adapter_name"] == "CMUStressAdapter"
    assert meta["status"] == "SPECIFICATION_STUB"
    assert meta["access_status"] == "ACCESS_REQUIRED"

    # Detection
    assert adapter.detect("cmu_keystroke_dataset.csv") is True
    assert adapter.detect("random_file.csv") is False

    # Loading non-existent file raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="CMU dataset file not found"):
        adapter.load("non_existent_cmu.csv")

    # Normalization raises NotImplementedError pending verified local schema
    with pytest.raises(NotImplementedError, match="CMU dataset normalization is pending"):
        adapter.normalize(pd.DataFrame())
