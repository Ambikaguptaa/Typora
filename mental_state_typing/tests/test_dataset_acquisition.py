"""Unit tests for dataset acquisition and discovery status module."""

import json
from pathlib import Path
import pytest

from src.data_engineering.dataset_acquisition import DatasetStatus, get_dataset_status


def test_dataset_status_missing_directory(tmp_path):
    """Verify that a non-existent directory reports status MISSING."""
    missing_dir = tmp_path / "non_existent_raw"
    status = get_dataset_status(data_dir=missing_dir)

    assert status["status"] == DatasetStatus.MISSING.value
    assert status["labeled_dataset_available"] is False
    assert status["file_count"] == 0
    assert len(status["candidate_files"]) == 0


def test_dataset_status_empty_directory(tmp_path):
    """Verify that an empty directory (or with only .gitkeep) reports status MISSING."""
    empty_dir = tmp_path / "raw_empty"
    empty_dir.mkdir()
    (empty_dir / ".gitkeep").write_text("", encoding="utf-8")

    status = get_dataset_status(data_dir=empty_dir)
    assert status["status"] == DatasetStatus.MISSING.value
    assert status["labeled_dataset_available"] is False
    assert status["file_count"] == 0


def test_dataset_status_candidate_found(tmp_path):
    """Verify that placing a research CSV marks status CANDIDATE_FOUND."""
    raw_dir = tmp_path / "raw_data"
    raw_dir.mkdir()
    csv_file = raw_dir / "research_keystrokes.csv"
    csv_file.write_text("participant_id,session_id,press_time,release_time,condition\n", encoding="utf-8")

    status = get_dataset_status(data_dir=raw_dir)
    assert status["status"] == DatasetStatus.CANDIDATE_FOUND.value
    assert status["labeled_dataset_available"] is True
    assert status["file_count"] == 1
    assert "research_keystrokes.csv" in status["candidate_files"]


def test_dataset_status_access_required(tmp_path):
    """Verify that manifest with ACCESS REQUIRED reports BLOCKED and access_status."""
    raw_dir = tmp_path / "raw_protected"
    raw_dir.mkdir()
    (raw_dir / "dataset.csv").write_text("dummy", encoding="utf-8")

    manifest = {
        "dataset_name": "MobileStress Candidate",
        "access_type": "ACCESS REQUIRED",
        "status": "blocked",
    }
    (raw_dir / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    status = get_dataset_status(data_dir=raw_dir)
    assert status["status"] == DatasetStatus.BLOCKED.value
    assert status["access_status"] == "ACCESS REQUIRED"
    assert status["labeled_dataset_available"] is False
