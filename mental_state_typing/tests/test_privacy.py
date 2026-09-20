"""Unit tests for the privacy and security subsystem."""

import pandas as pd
import pytest

from src.privacy.privacy_utils import is_safe_metadata_only, strip_character_data
from src.privacy.pseudonymization import pseudonymize_user_id


def test_privacy_modules_importable():
    """Verify that privacy modules can be imported cleanly."""
    import src.privacy.pseudonymization as p_pseudo
    import src.privacy.privacy_utils as p_utils
    import src.privacy.encryption as p_enc

    assert p_pseudo is not None
    assert p_utils is not None
    assert p_enc is not None


def test_pseudonymization_deterministic():
    """Verify pseudonymization generates consistent hashes for the same user & salt."""
    user = "student_experiment_001"
    salt = "fixed_test_salt"

    hash_1 = pseudonymize_user_id(user, salt=salt)
    hash_2 = pseudonymize_user_id(user, salt=salt)

    assert hash_1 == hash_2
    assert hash_1.startswith("usr_")
    assert user not in hash_1


def test_pseudonymization_salt_difference():
    """Verify different salts generate different pseudonyms for the same user."""
    user = "student_experiment_001"

    hash_salt_a = pseudonymize_user_id(user, salt="salt_alpha")
    hash_salt_b = pseudonymize_user_id(user, salt="salt_beta")

    assert hash_salt_a != hash_salt_b


def test_strip_character_data():
    """Verify that forbidden text/character columns are dropped."""
    unsafe_df = pd.DataFrame(
        {
            "press_time": [100.0, 200.0],
            "release_time": [180.0, 280.0],
            "key": ["a", "b"],
            "text": ["hello", "world"],
            "character": ["a", "b"],
        }
    )

    safe_df = strip_character_data(unsafe_df)

    assert "press_time" in safe_df.columns
    assert "release_time" in safe_df.columns
    assert "key" not in safe_df.columns
    assert "text" not in safe_df.columns
    assert "character" not in safe_df.columns


def test_is_safe_metadata_only():
    """Verify audit function correctly detects text column violations."""
    valid_cols = ["press_time", "release_time", "hold_time", "flight_time"]
    violating_cols = ["press_time", "release_time", "character"]

    assert is_safe_metadata_only(valid_cols) is True
    assert is_safe_metadata_only(violating_cols) is False
