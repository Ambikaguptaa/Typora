"""Tests for Dashboard Readiness reporting (Dataset, Model, Baseline, and Training Gates)."""

from pathlib import Path
import pytest

from src.integration.behavioral_engine import BehavioralEngine
from src.integration.system_status import get_system_overview_status, get_training_gate_matrix


def test_dashboard_dataset_missing_state():
    """Verify that a missing dataset reports REAL DATASET REQUIRED with BLOCKED status."""
    engine = BehavioralEngine()
    overview = get_system_overview_status(engine)

    assert "dataset" in overview
    assert overview["dataset"]["raw_status"] in ("missing", "candidate_found")
    if overview["dataset"]["raw_status"] == "missing":
        assert overview["dataset"]["badge"] == "BLOCKED"
        assert "REAL DATASET REQUIRED" in overview["dataset"]["label"]


def test_dashboard_model_blocked_state():
    """Verify that un-trained production model reports NOT READY without fabricated accuracy."""
    engine = BehavioralEngine()
    overview = get_system_overview_status(engine)

    assert overview["model"]["is_ready"] is False
    assert overview["model"]["badge"] == "NOT READY"
    assert "MODEL NOT READY" in overview["model"]["label"]
    assert "real research dataset" in overview["model"]["reason"].lower() or "verified" in overview["model"]["reason"].lower()


def test_dashboard_baseline_calibration_progression(tmp_path: Path):
    """Verify baseline readiness transitions from NOT READY (<5) to READY (>=5)."""
    engine = BehavioralEngine(baselines_dir=tmp_path)

    # 0 sessions
    ready, msg, count, req, _ = engine.check_baseline_readiness()
    assert ready is False
    assert count == 0
    assert req == 5

    # Add 4 calibration records (< 5)
    for i in range(4):
        engine.user_calibration_history.append(
            {
                "mean_dwell_ms": 90.0 + i,
                "mean_flight_ms": 110.0 + i,
                "pause_rate": 0.05,
                "estimated_wpm": 50.0,
                "backspace_count": 1,
            }
        )
    ready, msg, count, req, _ = engine.check_baseline_readiness()
    assert ready is False
    assert count == 4

    # Add 5th calibration record (>= 5)
    engine.user_calibration_history.append(
        {
            "mean_dwell_ms": 95.0,
            "mean_flight_ms": 115.0,
            "pause_rate": 0.06,
            "estimated_wpm": 48.0,
            "backspace_count": 2,
        }
    )
    ready, msg, count, req, profile = engine.check_baseline_readiness()
    assert ready is True
    assert count == 5
    assert profile is not None
    assert "means" in profile


def test_training_gate_matrix_truthfulness():
    """Verify training gate matrix reports truth for each criterion without fake passes."""
    matrix = get_training_gate_matrix()
    matrix_dict = {m["criterion"]: m["status"] for m in matrix}

    # Since no real dataset is installed, dataset-dependent criteria must be BLOCKED
    assert matrix_dict["Dataset Available"] in ("BLOCKED", "PENDING")
    assert matrix_dict["Schema Valid"] == "BLOCKED"
    assert matrix_dict["Labels Valid"] == "BLOCKED"

    # Static criteria must report PASS
    assert matrix_dict["Feature Compatibility"] == "PASS"
    assert matrix_dict["Privacy Audit"] == "PASS"
    assert matrix_dict["Scaler Readiness"] == "PASS"
