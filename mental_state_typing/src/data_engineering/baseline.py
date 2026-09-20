"""Baseline heuristic model for estimating mental strain.

Provides simple, interpretable statistical baseline metrics of typing strain.
This serves as a benchmark before introducing deep learning models in Phase 4.

DISCLAIMER: This provides behavioral strain estimation only, not a medical diagnosis.
"""

from typing import Any, Dict


def compute_baseline_strain(features: Dict[str, float]) -> Dict[str, Any]:
    """Compute a heuristic baseline strain score from extracted typing features.

    The score ranges from 0.0 (minimal observed strain) to 100.0 (elevated strain indicators).
    Features utilized:
    - pause_rate: Higher frequency of pauses often correlates with cognitive hesitation.
    - std_flight_time_ms: Greater rhythm irregularity indicates fragmented motor flow.
    - mean_hold_time_ms: Significantly elongated hold times may indicate fatigue.

    Args:
        features: Dictionary of typing features from feature_engineering module.

    Returns:
        Dict[str, Any]: Baseline score, categorical band, and explanatory note.
    """
    if features.get("keystroke_count", 0) < 5:
        return {
            "strain_score": 0.0,
            "strain_category": "Insufficient Data",
            "confidence": 0.0,
            "is_medical_diagnosis": False,
        }

    pause_rate = features.get("pause_rate", 0.0)
    std_flight = features.get("std_flight_time_ms", 0.0)
    mean_hold = features.get("mean_hold_time_ms", 100.0)

    # Simple normalized heuristic components (clamped 0 to 1)
    pause_component = min(pause_rate / 0.4, 1.0)
    variability_component = min(std_flight / 300.0, 1.0)
    hold_component = min(max((mean_hold - 80.0) / 150.0, 0.0), 1.0)

    # Weighted baseline score (0 - 100 scale)
    raw_score = (
        0.45 * pause_component
        + 0.35 * variability_component
        + 0.20 * hold_component
    ) * 100.0

    strain_score = round(float(raw_score), 1)

    if strain_score < 35.0:
        category = "Low Strain (Fluent Rhythm)"
    elif strain_score < 70.0:
        category = "Moderate Strain (Occasional Hesitation)"
    else:
        category = "High Strain (Fragmented Rhythm / Fatigue Indicators)"

    return {
        "strain_score": strain_score,
        "strain_category": category,
        "confidence": 0.75,
        "is_medical_diagnosis": False,
    }
