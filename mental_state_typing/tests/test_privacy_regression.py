"""Privacy regression test suite verifying strict Zero-Raw-Text compliance across live integration."""

import os
from pathlib import Path
import re
import pytest

from src.integration.behavioral_engine import BehavioralEngine
from src.integration.pipeline_state import SessionType
from src.live_typing.privacy_filter import FORBIDDEN_PAYLOAD_FIELDS, PrivacyViolationError


def test_behavioral_engine_rejects_dirty_payloads_in_strict_mode(tmp_path):
    """Verify engine immediately raises PrivacyViolationError when a payload with forbidden keys is fed."""
    engine = BehavioralEngine(
        models_dir=tmp_path / "models",
        baselines_dir=tmp_path / "baselines",
        assessments_dir=tmp_path / "assessments",
    )
    engine.start_session()

    dirty_batch = [
        {"event_type": "down", "timestamp_ms": 100.0, "key_token": "k_alpha", "key": "A"},
    ]
    with pytest.raises(PrivacyViolationError):
        engine.ingest_raw_events(dirty_batch, strict_privacy=True)


def test_codebase_zero_text_leakage_audit():
    """Verify integration and live typing source files do not define raw text storage variables."""
    src_dirs = [
        Path("src/integration"),
        Path("src/live_typing"),
    ]

    suspicious_patterns = [
        re.compile(r"\bself\._typed_text\b"),
        re.compile(r"\bself\._raw_characters\b"),
        re.compile(r"\bself\._input_value\b"),
        re.compile(r"\bself\._stored_words\b"),
        re.compile(r"\bcaptured_string\b"),
    ]

    for d in src_dirs:
        if not d.exists():
            continue
        for py_file in d.glob("*.py"):
            code = py_file.read_text(encoding="utf-8")
            for pattern in suspicious_patterns:
                match = pattern.search(code)
                assert match is None, f"Suspicious text capture pattern {pattern} found in {py_file}!"
