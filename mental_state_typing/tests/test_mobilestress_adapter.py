"""Unit tests for MobileStress research dataset adapter."""

import numpy as np
import pandas as pd
import pytest

from src.data_engineering.adapters.mobilestress_adapter import (
    MobileStressAdapter,
    adapt_mobilestress_dataset,
)
from src.privacy.privacy_utils import is_safe_metadata_only
from src.privacy.pseudonymization import is_valid_pseudonym


@pytest.fixture
def sample_mobilestress_fixture():
    """Small synthetic fixture mirroring the documented MobileStress schema.

    NOTE: This fixture is used strictly for testing adapter translation logic
    and must NEVER be treated as real empirical research data.
    """
    return pd.DataFrame(
        {
            "participant_id": ["P01", "P01", "P01", "P01", "P02", "P02"],
            "session": ["1", "1", "1", "1", "2", "2"],
            "condition": ["E", "E", "H", "H", "E", "E"],
            "time": [1000.0, 1080.0, 1200.0, 1310.0, 2000.0, 2095.0],
            "key": ["a", "a", "b", "b", "c", "c"],
            "event_type": ["down", "up", "down", "up", "down", "up"],
            "pressure": [0.45, 0.0, 0.72, 0.0, 0.50, 0.0],
        }
    )


def test_mobilestress_adapter_schema_mapping(sample_mobilestress_fixture):
    """Verify adapter maps MobileStress schema into standard event schema."""
    adapter = MobileStressAdapter(secret="test_secret")
    norm_df = adapter.adapt(sample_mobilestress_fixture)

    expected_cols = [
        "participant_id",
        "session_id",
        "timestamp",
        "event_type",
        "key_identifier",
        "press_time",
        "release_time",
        "condition",
        "pressure",
    ]

    for col in expected_cols:
        assert col in norm_df.columns

    # 3 down events should be paired with their release times
    assert len(norm_df) == 3


def test_mobilestress_condition_semantics(sample_mobilestress_fixture):
    """Verify experimental condition codes E and H preserve documented semantics."""
    norm_df = adapt_mobilestress_dataset(sample_mobilestress_fixture)

    conditions = norm_df["condition"].tolist()
    assert conditions[0] == "neutral"   # Condition 'E'
    assert conditions[1] == "stressed"  # Condition 'H'
    assert conditions[2] == "neutral"   # Condition 'E'


def test_mobilestress_down_up_pairing(sample_mobilestress_fixture):
    """Verify sequential DOWN and UP events correctly derive dwell time intervals."""
    adapter = MobileStressAdapter()
    norm_df = adapter.adapt(sample_mobilestress_fixture)

    row1 = norm_df.iloc[0]
    assert row1["press_time"] == 1000.0
    assert row1["release_time"] == 1080.0
    assert (row1["release_time"] - row1["press_time"]) == 80.0

    row2 = norm_df.iloc[1]
    assert row2["press_time"] == 1200.0
    assert row2["release_time"] == 1310.0
    assert (row2["release_time"] - row2["press_time"]) == 110.0


def test_mobilestress_zero_raw_text_masking(sample_mobilestress_fixture):
    """Verify raw typed characters are masked to abstract identifiers and text is suppressed."""
    norm_df = adapt_mobilestress_dataset(sample_mobilestress_fixture)

    # Check that raw 'key' column is completely absent
    assert "key" not in norm_df.columns
    assert is_safe_metadata_only(norm_df.columns) is True

    # Check that key_identifier is masked (e.g. k_<hash>)
    for k_id in norm_df["key_identifier"]:
        assert str(k_id).startswith("k_")
        assert k_id not in ["a", "b", "c"]


def test_mobilestress_pseudonymization(sample_mobilestress_fixture):
    """Verify participant IDs are pseudonymized into usr_<16-hex> format."""
    adapter = MobileStressAdapter(pseudonymize_users=True, secret="secret123")
    norm_df = adapter.adapt(sample_mobilestress_fixture)

    for pid in norm_df["participant_id"]:
        assert is_valid_pseudonym(pid) is True
        assert pid not in ["P01", "P02"]


def test_mobilestress_single_event_unpaired():
    """Verify adapter does NOT fabricate release time when up events are missing."""
    single_df = pd.DataFrame(
        {
            "participant_id": ["P01", "P01"],
            "session": ["1", "1"],
            "condition": ["E", "E"],
            "time": [100.0, 300.0],
            "key": ["a", "b"],
            "event_type": ["down", "down"],
        }
    )

    norm_df = adapt_mobilestress_dataset(single_df)
    assert len(norm_df) == 2
    assert norm_df["press_time"].iloc[0] == 100.0
    assert np.isnan(norm_df["release_time"].iloc[0])
