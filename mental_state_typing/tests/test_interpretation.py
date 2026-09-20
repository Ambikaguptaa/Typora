"""Unit tests for model output and baseline deviation interpretation (interpretation.py)."""

import pytest

from src.assessment.interpretation import (
    DEVIATION_LEVEL_EXPECTED,
    DEVIATION_LEVEL_HIGHER,
    DEVIATION_LEVEL_MODERATE,
    RELIABILITY_HIGH_SEPARATION,
    RELIABILITY_LOW_SEPARATION,
    RELIABILITY_MODERATE_SEPARATION,
    interpret_baseline_deviation,
    interpret_model_prediction,
    rank_feature_deviations,
)


def test_interpret_model_prediction_multiclass_high_separation():
    """Verify multiclass interpretation with high class separation."""
    probs = {"Calm": 0.75, "Fatigued": 0.15, "High_Workload": 0.10}

    result = interpret_model_prediction(probs)

    assert result["status"] == "interpreted"
    assert result["predicted_class"] == "Calm"
    assert result["top_probability"] == 0.75
    assert result["second_class"] == "Fatigued"
    assert result["second_probability"] == 0.15
    assert pytest.approx(result["probability_margin"], abs=1e-3) == 0.60
    assert result["reliability"] == RELIABILITY_HIGH_SEPARATION
    assert result["entropy"] > 0.0
    assert 0.0 <= result["normalized_entropy"] <= 1.0
    assert "disclaimer" in result


def test_interpret_model_prediction_low_separation():
    """Verify prediction uncertainty and low separation reliability."""
    probs = {"Calm": 0.36, "Fatigued": 0.34, "High_Workload": 0.30}

    result = interpret_model_prediction(probs)

    assert result["predicted_class"] == "Calm"
    assert pytest.approx(result["probability_margin"], abs=1e-3) == 0.02
    assert result["reliability"] == RELIABILITY_LOW_SEPARATION
    # High entropy because probabilities are almost uniformly distributed
    assert result["normalized_entropy"] > 0.90


def test_interpret_model_prediction_moderate_separation():
    """Verify moderate separation categorization."""
    probs = {"ClassA": 0.55, "ClassB": 0.35, "ClassC": 0.10}

    result = interpret_model_prediction(probs)

    assert pytest.approx(result["probability_margin"], abs=1e-3) == 0.20
    assert result["reliability"] == RELIABILITY_MODERATE_SEPARATION


def test_interpret_model_prediction_binary():
    """Verify binary probability interpretation."""
    probs = {"Baseline": 0.85, "Elevated": 0.15}

    result = interpret_model_prediction(probs)

    assert result["predicted_class"] == "Baseline"
    assert result["top_probability"] == 0.85
    assert pytest.approx(result["probability_margin"], abs=1e-3) == 0.70
    assert result["reliability"] == RELIABILITY_HIGH_SEPARATION


def test_interpret_model_prediction_normalization():
    """Verify that unnormalized probabilities (e.g. summing to 100) are normalized."""
    raw_probs = {"A": 70.0, "B": 20.0, "C": 10.0}

    result = interpret_model_prediction(raw_probs)

    assert pytest.approx(result["top_probability"], abs=1e-3) == 0.70
    assert pytest.approx(result["second_probability"], abs=1e-3) == 0.20
    assert pytest.approx(sum(result["class_probabilities"].values()), abs=1e-4) == 1.0


def test_rank_feature_deviations():
    """Verify ranking of features by absolute standardized deviation (|z-score|)."""
    deviations = {
        "mean_dwell_time": {"z_score": 1.5, "direction": "higher", "absolute_deviation": 15.0},
        "mean_flight_time": {"z_score": -2.8, "direction": "lower", "absolute_deviation": -30.0},
        "pause_rate": {"z_score": 0.4, "direction": "within_baseline", "absolute_deviation": 0.02},
        "backspace_rate": {"z_score": 2.1, "direction": "higher", "absolute_deviation": 0.08},
    }

    ranked = rank_feature_deviations(deviations, top_n=3)

    assert len(ranked) == 3
    # Top rank should be mean_flight_time (|z|=2.8)
    assert ranked[0]["feature"] == "mean_flight_time"
    assert ranked[0]["abs_z_score"] == 2.8
    assert ranked[0]["direction"] == "lower"
    # Second rank should be backspace_rate (|z|=2.1)
    assert ranked[1]["feature"] == "backspace_rate"
    # Third rank should be mean_dwell_time (|z|=1.5)
    assert ranked[2]["feature"] == "mean_dwell_time"


def test_interpret_baseline_deviation_available():
    """Verify personal baseline interpretation when baseline is available."""
    baseline_res = {
        "baseline_status": "available",
        "typing_deviation_index": 68.5,
        "features": {
            "mean_dwell_time": {"z_score": 2.2, "direction": "higher"},
            "mean_flight_time": {"z_score": 1.8, "direction": "higher"},
        },
    }

    interp = interpret_baseline_deviation(baseline_res)

    assert interp["baseline_status"] == "available"
    assert interp["typing_deviation_index"] == 68.5
    assert interp["deviation_level"] == DEVIATION_LEVEL_HIGHER
    assert len(interp["largest_deviations"]) == 2
    assert "disclaimer" in interp


def test_interpret_baseline_deviation_levels():
    """Verify expected, moderate, and higher deviation level thresholds."""
    # Expected variance: TDI < 30
    res_low = {"baseline_status": "available", "typing_deviation_index": 18.0, "features": {}}
    assert interpret_baseline_deviation(res_low)["deviation_level"] == DEVIATION_LEVEL_EXPECTED

    # Moderate deviation: 30 <= TDI < 60
    res_mod = {"baseline_status": "available", "typing_deviation_index": 45.0, "features": {}}
    assert interpret_baseline_deviation(res_mod)["deviation_level"] == DEVIATION_LEVEL_MODERATE

    # Higher deviation: TDI >= 60
    res_high = {"baseline_status": "available", "typing_deviation_index": 75.0, "features": {}}
    assert interpret_baseline_deviation(res_high)["deviation_level"] == DEVIATION_LEVEL_HIGHER


def test_interpret_baseline_deviation_unavailable():
    """Verify graceful handling when personal baseline is unavailable or cold-start."""
    baseline_res = {
        "baseline_status": "insufficient_history",
        "session_count": 2,
        "message": "Requires at least 5 sessions.",
    }

    interp = interpret_baseline_deviation(baseline_res)

    assert interp["baseline_status"] == "insufficient_history"
    assert interp["typing_deviation_index"] is None
    assert interp["deviation_level"] == "unavailable"
    assert len(interp["largest_deviations"]) == 0
