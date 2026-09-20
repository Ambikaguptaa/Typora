"""Unit tests for the Personal Typing Baseline Analysis engine.

Validates:
- Statistical baseline metrics (Mean, Median, Std, MAD, IQR)
- Insufficient history and cold-start handling
- Absolute, percentage, and standardized (z-score) deviations
- Safe handling of zero baseline and zero standard deviation
- Controlled baseline updating
- Non-diagnostic Typing Deviation Index (TDI)
- Privacy enforcement (zero text, pseudonymous IDs)
- Reproducibility
- Step 15 exact numerical validation example
- Legacy heuristic strain model backwards compatibility
"""

import numpy as np
import pandas as pd
import pytest

from src.data_engineering.baseline import (
    DEFAULT_BASELINE_FEATURES,
    build_all_user_baselines,
    build_user_baseline,
    calculate_baseline_deviation,
    compute_baseline_strain,
    generate_baseline_report,
    load_user_baselines,
    save_user_baselines,
    update_user_baseline,
)


# ============================================================================
# Legacy Heuristic Strain Model Tests (Backwards Compatibility)
# ============================================================================

def test_baseline_module_importable():
    """Verify that baseline module and helper functions import cleanly."""
    import src.data_engineering.baseline as baseline

    assert hasattr(baseline, "compute_baseline_strain")
    assert hasattr(baseline, "build_user_baseline")
    assert hasattr(baseline, "calculate_baseline_deviation")


def test_compute_baseline_strain_low_data():
    """Verify legacy handling when keystroke count is insufficient."""
    low_data_features = {
        "keystroke_count": 3,
        "mean_hold_time_ms": 100.0,
        "pause_rate": 0.1,
    }
    result = compute_baseline_strain(low_data_features)
    assert result["strain_score"] == 0.0
    assert result["strain_category"] == "Insufficient Data"
    assert result["is_medical_diagnosis"] is False


def test_compute_baseline_strain_normal_range():
    """Verify strain scoring output with realistic feature parameters."""
    features = {
        "keystroke_count": 25,
        "mean_hold_time_ms": 120.0,
        "std_flight_time_ms": 50.0,
        "pause_rate": 0.05,
    }
    result = compute_baseline_strain(features)
    assert 0.0 <= result["strain_score"] <= 100.0
    assert "strain_category" in result
    assert result["is_medical_diagnosis"] is False


def test_compute_baseline_strain_high_hesitation():
    """Verify that elevated pauses and high variance yield a higher strain score."""
    calm_features = {
        "keystroke_count": 30,
        "mean_hold_time_ms": 90.0,
        "std_flight_time_ms": 30.0,
        "pause_rate": 0.02,
    }
    strained_features = {
        "keystroke_count": 30,
        "mean_hold_time_ms": 220.0,
        "std_flight_time_ms": 280.0,
        "pause_rate": 0.35,
    }
    calm_result = compute_baseline_strain(calm_features)
    strained_result = compute_baseline_strain(strained_features)
    assert strained_result["strain_score"] > calm_result["strain_score"]


# ============================================================================
# Personal Typing Baseline Tests
# ============================================================================

def test_build_user_baseline_statistics():
    """Verify calculation of mean, median, std, min, max, MAD, and IQR."""
    # 5 sessions for user_test
    df = pd.DataFrame(
        {
            "user_id": ["user_test"] * 5,
            "mean_typing_speed_wpm": [50.0, 52.0, 48.0, 54.0, 46.0],
            "mean_dwell_time": [100.0, 105.0, 95.0, 110.0, 90.0],
        }
    )

    baseline = build_user_baseline(df, user_id="user_test", min_sessions=5)

    assert baseline["baseline_status"] == "available"
    assert baseline["session_count"] == 5

    speed_stats = baseline["features"]["mean_typing_speed_wpm"]
    assert pytest.approx(speed_stats["mean"], abs=1e-2) == 50.0
    assert pytest.approx(speed_stats["median"], abs=1e-2) == 50.0
    assert speed_stats["min"] == 46.0
    assert speed_stats["max"] == 54.0
    assert speed_stats["std"] > 0.0
    assert speed_stats["mad"] > 0.0
    assert speed_stats["iqr"] > 0.0


def test_insufficient_history_detection_cold_start():
    """Verify cold-start handling when a user has fewer than min_sessions."""
    # Only 3 sessions
    df = pd.DataFrame(
        {
            "user_id": ["user_new"] * 3,
            "mean_typing_speed_wpm": [50.0, 52.0, 51.0],
        }
    )

    baseline = build_user_baseline(df, user_id="user_new", min_sessions=5)

    assert baseline["baseline_status"] == "insufficient_history"
    assert baseline["session_count"] == 3
    assert baseline["min_required_sessions"] == 5
    assert baseline["features"] == {}  # No fabricated features!


def test_calculate_baseline_deviation_percentage_and_zscore():
    """Verify absolute deviation, percentage deviation, and z-score."""
    baseline = {
        "user_id": "u1",
        "baseline_status": "available",
        "session_count": 10,
        "features": {
            "mean_typing_speed_wpm": {
                "mean": 50.0,
                "std": 5.0,
                "median": 50.0,
                "mad": 3.0,
                "unit": "WPM",
                "interpretation": "typing velocity",
            }
        },
    }

    # Observed session with 60 WPM (+10 absolute, +20%, z = +2.0)
    current_session = {"mean_typing_speed_wpm": 60.0}
    dev = calculate_baseline_deviation(current_session, baseline, tolerance=0.5)

    speed_dev = dev["features"]["mean_typing_speed_wpm"]
    assert speed_dev["absolute_deviation"] == 10.0
    assert speed_dev["percentage_deviation"] == 20.0
    assert pytest.approx(speed_dev["z_score"], abs=1e-2) == 2.0
    assert speed_dev["direction"] == "higher"
    assert speed_dev["status"] == "deviated"


def test_zero_baseline_mean_handling():
    """Verify percentage deviation does not divide by zero when baseline mean is 0."""
    baseline = {
        "user_id": "u1",
        "baseline_status": "available",
        "session_count": 5,
        "features": {
            "error_rate": {
                "mean": 0.0,
                "std": 0.01,
                "median": 0.0,
                "mad": 0.0,
            }
        },
    }

    current = {"error_rate": 0.05}
    dev = calculate_baseline_deviation(current, baseline)

    err_dev = dev["features"]["error_rate"]
    assert not np.isnan(err_dev["percentage_deviation"])
    assert not np.isinf(err_dev["percentage_deviation"])
    assert err_dev["percentage_deviation"] == 0.0


def test_zero_standard_deviation_handling():
    """Verify z-score calculation does not crash when baseline std is zero."""
    baseline = {
        "user_id": "u1",
        "baseline_status": "available",
        "session_count": 5,
        "features": {
            "pause_rate": {
                "mean": 0.10,
                "std": 0.0,  # Zero variance
                "median": 0.10,
                "mad": 0.0,
            }
        },
    }

    current = {"pause_rate": 0.10}
    dev = calculate_baseline_deviation(current, baseline)

    pause_dev = dev["features"]["pause_rate"]
    assert pause_dev["has_zero_variance"] is True
    assert pause_dev["z_score"] == 0.0
    assert pause_dev["direction"] == "within_baseline"


def test_direction_of_change_tolerance():
    """Verify direction classifies 'within_baseline' when within tolerance."""
    baseline = {
        "user_id": "u1",
        "baseline_status": "available",
        "session_count": 10,
        "features": {
            "mean_dwell_time": {
                "mean": 100.0,
                "std": 10.0,
                "median": 100.0,
                "mad": 7.0,
            }
        },
    }

    # Current = 103 (z = 0.3, within tolerance 0.5)
    curr_within = {"mean_dwell_time": 103.0}
    res_within = calculate_baseline_deviation(curr_within, baseline, tolerance=0.5)
    assert res_within["features"]["mean_dwell_time"]["direction"] == "within_baseline"

    # Current = 115 (z = 1.5, above tolerance 0.5)
    curr_higher = {"mean_dwell_time": 115.0}
    res_higher = calculate_baseline_deviation(curr_higher, baseline, tolerance=0.5)
    assert res_higher["features"]["mean_dwell_time"]["direction"] == "higher"


def test_typing_deviation_index_normalized_range():
    """Verify TDI is normalized between 0.0 and 100.0."""
    baseline = {
        "user_id": "u1",
        "baseline_status": "available",
        "session_count": 5,
        "features": {
            "f1": {"mean": 10.0, "std": 1.0, "median": 10.0, "mad": 0.8},
            "f2": {"mean": 20.0, "std": 2.0, "median": 20.0, "mad": 1.5},
        },
    }

    # Identical to baseline -> TDI = 0.0
    curr_zero = {"f1": 10.0, "f2": 20.0}
    dev_zero = calculate_baseline_deviation(curr_zero, baseline)
    assert dev_zero["typing_deviation_index"] == 0.0

    # Massive deviation -> TDI clamped to 100.0
    curr_huge = {"f1": 100.0, "f2": 200.0}
    dev_huge = calculate_baseline_deviation(curr_huge, baseline)
    assert dev_huge["typing_deviation_index"] == 100.0


def test_baseline_update():
    """Verify incorporating new session records into an existing baseline."""
    df_initial = pd.DataFrame(
        {
            "user_id": ["u1"] * 4,
            "mean_typing_speed_wpm": [50.0, 52.0, 51.0, 53.0],
        }
    )
    b_initial = build_user_baseline(df_initial, user_id="u1", min_sessions=5)
    assert b_initial["baseline_status"] == "insufficient_history"

    # Add 2 more sessions to reach 6 total
    df_new = pd.DataFrame(
        {
            "user_id": ["u1"] * 6,
            "mean_typing_speed_wpm": [50.0, 52.0, 51.0, 53.0, 50.0, 52.0],
        }
    )
    b_updated = update_user_baseline(b_initial, df_new, min_sessions=5)
    assert b_updated["baseline_status"] == "available"
    assert b_updated["session_count"] == 6


def test_zero_raw_text_in_baseline_storage(tmp_path):
    """Verify baseline storage excludes raw text and respects privacy."""
    df = pd.DataFrame(
        {
            "user_id": ["usr_anon_1"] * 5,
            "mean_dwell_time": [100.0, 102.0, 99.0, 101.0, 103.0],
            "text": ["secret sentence 1", "secret sentence 2", "s3", "s4", "s5"],
            "message": ["private msg", "private msg 2", "m3", "m4", "m5"],
        }
    )

    baselines_map = build_all_user_baselines(df, user_column="user_id", min_sessions=5)
    csv_p, json_p = save_user_baselines(baselines_map, output_dir=tmp_path)

    csv_text = csv_p.read_text(encoding="utf-8")
    json_text = json_p.read_text(encoding="utf-8")

    assert "secret sentence" not in csv_text
    assert "secret sentence" not in json_text
    assert "private msg" not in csv_text
    assert "private msg" not in json_text


def test_reproducibility_of_baselines():
    """Verify baseline construction is deterministic given the same data."""
    df = pd.DataFrame(
        {
            "user_id": ["u_rep"] * 6,
            "mean_typing_speed_wpm": [45.0, 48.0, 47.0, 46.0, 49.0, 47.5],
        }
    )
    b1 = build_user_baseline(df, user_id="u_rep", min_sessions=5)
    b2 = build_user_baseline(df, user_id="u_rep", min_sessions=5)

    assert b1 == b2


def test_step_15_exact_numerical_example():
    """STEP 15 VALIDATION: Test the exact numerical values specified in user prompt.

    Historical Sessions (1 to 5):
    - Session 1: WPM = 50, Dwell = 100, Flight = 80, Backspace rate = 0.03
    - Session 2: WPM = 52, Dwell = 105, Flight = 82, Backspace rate = 0.04
    - Session 3: WPM = 49, Dwell = 102, Flight = 79, Backspace rate = 0.03
    - Session 4: WPM = 51, Dwell = 101, Flight = 81, Backspace rate = 0.03
    - Session 5: WPM = 53, Dwell = 104, Flight = 84, Backspace rate = 0.04

    New Session (Session 6):
    - WPM = 42, Dwell = 135, Flight = 110, Backspace rate = 0.08
    """
    historical_sessions = pd.DataFrame(
        {
            "user_id": ["user_step15"] * 5,
            "session_id": [f"sess_{i}" for i in range(1, 6)],
            "mean_typing_speed_wpm": [50.0, 52.0, 49.0, 51.0, 53.0],
            "mean_dwell_time": [100.0, 105.0, 102.0, 101.0, 104.0],
            "mean_flight_time": [80.0, 82.0, 79.0, 81.0, 84.0],
            "backspace_rate": [0.03, 0.04, 0.03, 0.03, 0.04],
        }
    )

    baseline = build_user_baseline(
        historical_sessions,
        user_id="user_step15",
        min_sessions=5,
        baseline_features=[
            "mean_typing_speed_wpm",
            "mean_dwell_time",
            "mean_flight_time",
            "backspace_rate",
        ],
    )

    assert baseline["baseline_status"] == "available"
    assert baseline["session_count"] == 5

    # Verify Baseline Means:
    # WPM: (50 + 52 + 49 + 51 + 53) / 5 = 51.0
    assert pytest.approx(baseline["features"]["mean_typing_speed_wpm"]["mean"], abs=1e-2) == 51.0
    # Dwell: (100 + 105 + 102 + 101 + 104) / 5 = 102.4
    assert pytest.approx(baseline["features"]["mean_dwell_time"]["mean"], abs=1e-2) == 102.4
    # Flight: (80 + 82 + 79 + 81 + 84) / 5 = 81.2
    assert pytest.approx(baseline["features"]["mean_flight_time"]["mean"], abs=1e-2) == 81.2
    # Backspace: (0.03 + 0.04 + 0.03 + 0.03 + 0.04) / 5 = 0.034
    assert pytest.approx(baseline["features"]["backspace_rate"]["mean"], abs=1e-3) == 0.034

    # Evaluate New Session
    new_session = {
        "session_id": "sess_06",
        "mean_typing_speed_wpm": 42.0,
        "mean_dwell_time": 135.0,
        "mean_flight_time": 110.0,
        "backspace_rate": 0.08,
    }

    report = generate_baseline_report(new_session, baseline, tolerance=0.5)

    assert report["baseline_status"] == "available"
    assert report["historical_sessions"] == 5
    assert report["typing_deviation_index"] is not None

    devs = {d["feature"]: d for d in report["all_deviations"]}

    # WPM: 42 vs 51.0 -> absolute dev = -9.0, direction = 'lower'
    assert pytest.approx(devs["mean_typing_speed_wpm"]["absolute_deviation"], abs=1e-2) == -9.0
    assert pytest.approx(devs["mean_typing_speed_wpm"]["percentage_deviation"], abs=1e-1) == -17.65
    assert devs["mean_typing_speed_wpm"]["direction"] == "lower"

    # Dwell: 135 vs 102.4 -> absolute dev = +32.6, direction = 'higher'
    assert pytest.approx(devs["mean_dwell_time"]["absolute_deviation"], abs=1e-2) == 32.6
    assert pytest.approx(devs["mean_dwell_time"]["percentage_deviation"], abs=1e-1) == 31.84
    assert devs["mean_dwell_time"]["direction"] == "higher"

    # Flight: 110 vs 81.2 -> absolute dev = +28.8, direction = 'higher'
    assert pytest.approx(devs["mean_flight_time"]["absolute_deviation"], abs=1e-2) == 28.8
    assert pytest.approx(devs["mean_flight_time"]["percentage_deviation"], abs=1e-1) == 35.47
    assert devs["mean_flight_time"]["direction"] == "higher"

    # Backspace: 0.08 vs 0.034 -> absolute dev = +0.046, direction = 'higher'
    assert pytest.approx(devs["backspace_rate"]["absolute_deviation"], abs=1e-3) == 0.046
    assert pytest.approx(devs["backspace_rate"]["percentage_deviation"], abs=1e-1) == 135.29
    assert devs["backspace_rate"]["direction"] == "higher"

    # Verify report limitations emphasize non-medical interpretation
    assert any("medical" in lim.lower() for lim in report["limitations"])
