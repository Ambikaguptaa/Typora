"""Unit tests for Canonical Event Schema and validation."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data_engineering.canonical_schema import (
    CANONICAL_REQUIRED_COLUMNS,
    FORBIDDEN_TEXT_COLUMNS,
    assert_canonical_schema,
    validate_canonical_dataframe,
)


@pytest.fixture
def valid_canonical_df():
    """Create a valid canonical DataFrame fixture."""
    return pd.DataFrame({
        "participant_id": ["p_01", "p_01", "p_02", "p_02"],
        "session_id": ["s_1", "s_1", "s_1", "s_1"],
        "timestamp": [100.0, 250.0, 100.0, 240.0],
        "event_type": ["down", "down", "down", "down"],
        "key_identifier": ["k_alpha", "k_beta", "k_alpha", "k_gamma"],
        "press_time": [100.0, 250.0, 100.0, 240.0],
        "release_time": [180.0, 320.0, 175.0, 310.0],
        "condition": ["neutral", "neutral", "stressed", "stressed"],
        "pressure": [0.65, 0.70, 0.85, 0.90],
    })


def test_valid_canonical_dataframe_passes(valid_canonical_df):
    """Verify that a compliant canonical DataFrame passes validation."""
    report = validate_canonical_dataframe(valid_canonical_df)
    assert report["is_valid"] is True
    assert len(report["errors"]) == 0
    assert report["details"]["row_count"] == 4
    assert report["details"]["unique_participants"] == 2
    assert "neutral" in report["details"]["unique_conditions"]
    assert "stressed" in report["details"]["unique_conditions"]

    # Should not raise
    assert_canonical_schema(valid_canonical_df)


def test_missing_required_columns_fails(valid_canonical_df):
    """Verify that dropping any required column triggers validation failure."""
    for col in CANONICAL_REQUIRED_COLUMNS:
        df_invalid = valid_canonical_df.drop(columns=[col])
        report = validate_canonical_dataframe(df_invalid)
        assert report["is_valid"] is False
        assert any(col in err for err in report["errors"])

        with pytest.raises(ValueError, match="Canonical schema validation failed"):
            assert_canonical_schema(df_invalid)


def test_forbidden_raw_text_column_fails(valid_canonical_df):
    """Verify that including raw text columns violates zero-text invariant."""
    for forbidden in ["key", "char", "text", "typed_text", "password", "user_input"]:
        df_leak = valid_canonical_df.copy()
        df_leak[forbidden] = ["hello", "world", "secret", "text"]
        report = validate_canonical_dataframe(df_leak)
        assert report["is_valid"] is False
        assert any("Zero-Raw-Text violation" in err for err in report["errors"])


def test_raw_single_char_in_key_identifier_fails(valid_canonical_df):
    """Verify that raw typed characters in key_identifier fail validation."""
    df_raw_char = valid_canonical_df.copy()
    df_raw_char.loc[0, "key_identifier"] = "a"  # Raw typed character
    report = validate_canonical_dataframe(df_raw_char)
    assert report["is_valid"] is False
    assert any("raw characters" in err for err in report["errors"])


def test_null_in_identity_or_timestamp_fails(valid_canonical_df):
    """Verify that null/NaN values in critical identity and timing fields fail."""
    for col in ["participant_id", "session_id", "timestamp", "condition"]:
        df_null = valid_canonical_df.copy()
        df_null.loc[0, col] = np.nan
        report = validate_canonical_dataframe(df_null)
        assert report["is_valid"] is False
        assert any(f"Canonical column '{col}' contains" in err for err in report["errors"])


def test_non_numeric_timestamps_fail(valid_canonical_df):
    """Verify that string timestamps fail schema validation."""
    df_bad_time = valid_canonical_df.copy()
    df_bad_time["timestamp"] = ["t1", "t2", "t3", "t4"]
    report = validate_canonical_dataframe(df_bad_time)
    assert report["is_valid"] is False
    assert any("numeric" in err for err in report["errors"])


def test_canonical_test_fixture_file_passes():
    """Verify that the test fixture tests/fixtures/canonical_fixture.csv strictly complies."""
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "canonical_fixture.csv"
    assert fixture_path.exists(), "canonical_fixture.csv should exist under tests/fixtures/"

    df_fixture = pd.read_csv(fixture_path)
    report = validate_canonical_dataframe(df_fixture)
    assert report["is_valid"] is True
    assert len(report["errors"]) == 0
    assert report["details"]["unique_participants"] == 4
    assert set(report["details"]["unique_conditions"]) == {"neutral", "stressed"}
