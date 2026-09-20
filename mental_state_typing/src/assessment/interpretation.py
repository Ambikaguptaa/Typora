"""Model Output Interpretation and Baseline Deviation Diagnostics Module.

Interprets deep learning class probabilities, evaluates prediction uncertainty
via Shannon entropy and probability separation margins, categorizes prediction
reliability neutrally, ranks feature deviations against personal baselines, and
enforces clear non-diagnostic boundaries.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Neutral reliability separation levels (model-oriented, never medical)
RELIABILITY_HIGH_SEPARATION = "HIGH_SEPARATION"
RELIABILITY_MODERATE_SEPARATION = "MODERATE_SEPARATION"
RELIABILITY_LOW_SEPARATION = "LOW_SEPARATION"

# Neutral baseline deviation levels
DEVIATION_LEVEL_EXPECTED = "expected_variance"
DEVIATION_LEVEL_MODERATE = "moderate_deviation"
DEVIATION_LEVEL_HIGHER = "higher_deviation"

# Academic disclaimer
ASSESSMENT_DISCLAIMER = (
    "Assessments represent typing rhythm patterns and statistical deviations. "
    "This system does not provide psychological, psychiatric, or medical diagnoses."
)


def interpret_model_prediction(
    class_probabilities: Dict[str, float],
    predicted_class: Optional[str] = None,
    high_separation_threshold: float = 0.30,
    moderate_separation_threshold: float = 0.15,
) -> Dict[str, Any]:
    """Interpret model class probabilities, calculate uncertainty, and assess separation reliability.

    Uncertainty Metrics:
    1. top_probability: Maximum class probability.
    2. probability_margin: Difference between top and second class probability (p_top - p_second).
    3. Shannon entropy: H(p) = -Σ p_i log2(p_i), reflecting prediction dispersion across classes.
    4. normalized_entropy: H(p) / log2(K), scaled from 0.0 (certainty) to 1.0 (uniform uncertainty).

    Reliability Categorization:
    - HIGH_SEPARATION: margin >= high_separation_threshold (default: 0.30)
    - MODERATE_SEPARATION: moderate_separation_threshold <= margin < high_separation_threshold
    - LOW_SEPARATION: margin < moderate_separation_threshold (competing classes are close)

    Args:
        class_probabilities: Dictionary mapping class name to model probability.
        predicted_class: Optional explicit predicted class (inferred from max prob if None).
        high_separation_threshold: Probability margin threshold for high separation.
        moderate_separation_threshold: Probability margin threshold for moderate separation.

    Returns:
        Dict[str, Any]: Model interpretation report.
    """
    if not class_probabilities:
        return {
            "status": "unavailable",
            "message": "No model class probabilities provided.",
            "disclaimer": ASSESSMENT_DISCLAIMER,
        }

    # Normalize probabilities if sum deviates from 1.0
    raw_sum = sum(class_probabilities.values())
    if raw_sum > 1e-6:
        probs = {k: float(v) / raw_sum for k, v in class_probabilities.items()}
    else:
        probs = {k: 1.0 / len(class_probabilities) for k in class_probabilities}

    # Sort classes by probability descending
    sorted_classes = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    top_cls, top_p = sorted_classes[0]
    pred_cls = predicted_class if predicted_class is not None else top_cls

    if len(sorted_classes) > 1:
        second_cls, second_p = sorted_classes[1]
        margin = top_p - second_p
    else:
        second_cls, second_p = None, 0.0
        margin = top_p

    # Shannon Entropy H(p) = -Σ p_i log2(p_i)
    k_classes = len(probs)
    entropy = 0.0
    for p in probs.values():
        if p > 1e-9:
            entropy -= p * np.log2(p)

    # Normalized entropy in range [0, 1]
    max_entropy = np.log2(k_classes) if k_classes > 1 else 1.0
    norm_entropy = float(entropy / max_entropy) if max_entropy > 1e-9 else 0.0

    # Categorize reliability based on probability separation
    if margin >= high_separation_threshold:
        reliability = RELIABILITY_HIGH_SEPARATION
        reliability_desc = "The model clearly separates the top predicted class from alternatives."
    elif margin >= moderate_separation_threshold:
        reliability = RELIABILITY_MODERATE_SEPARATION
        reliability_desc = "The model exhibits moderate separation between competing classes."
    else:
        reliability = RELIABILITY_LOW_SEPARATION
        reliability_desc = "The model exhibits low separation between competing classes; alternative states have similar probabilities."

    return {
        "status": "interpreted",
        "predicted_class": pred_cls,
        "top_probability": round(float(top_p), 4),
        "second_class": second_cls,
        "second_probability": round(float(second_p), 4) if second_cls else None,
        "probability_margin": round(float(margin), 4),
        "entropy": round(float(entropy), 4),
        "normalized_entropy": round(float(norm_entropy), 4),
        "reliability": reliability,
        "reliability_description": reliability_desc,
        "class_probabilities": {k: round(float(v), 4) for k, v in probs.items()},
        "disclaimer": ASSESSMENT_DISCLAIMER,
    }


def rank_feature_deviations(
    feature_deviations: Dict[str, Any],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Rank feature-level deviations by absolute standardized deviation (|z-score|).

    Args:
        feature_deviations: Feature deviation dictionary from calculate_baseline_deviation().
        top_n: Number of largest deviations to return (default: 5).

    Returns:
        List[Dict[str, Any]]: Ranked list of feature deviations.
    """
    ranked: List[Dict[str, Any]] = []

    for feat, stats in feature_deviations.items():
        if not isinstance(stats, dict):
            continue

        z = stats.get("z_score")
        if z is None:
            continue

        z_float = float(z)
        abs_z = abs(z_float)
        direction = stats.get("direction", "within_baseline")
        unit = stats.get("unit", "")
        interpretation = stats.get("interpretation", feat)

        # Descriptive representation (e.g. "+2.4 SD higher than baseline")
        sign = "+" if z_float > 0 else ""
        descriptor = f"{sign}{z_float:.1f} SD {direction} than baseline"

        ranked.append({
            "feature": feat,
            "z_score": round(z_float, 2),
            "abs_z_score": round(abs_z, 2),
            "direction": direction,
            "absolute_deviation": stats.get("absolute_deviation"),
            "percentage_deviation": stats.get("percentage_deviation"),
            "descriptor": descriptor,
            "unit": unit,
            "interpretation": interpretation,
        })

    # Sort descending by absolute z-score
    ranked.sort(key=lambda item: item["abs_z_score"], reverse=True)
    return ranked[:top_n]


def interpret_baseline_deviation(
    baseline_deviation_result: Dict[str, Any],
    higher_tdi_threshold: float = 60.0,
    moderate_tdi_threshold: float = 30.0,
    top_features_count: int = 5,
) -> Dict[str, Any]:
    """Interpret personal baseline deviation and the Typing Deviation Index (TDI).

    CRITICAL INTERPRETATION RULE:
    'higher_deviation' represents greater statistical divergence from an individual's
    historical baseline. It MUST NOT be translated into 'higher stress' or clinical diagnoses.

    Args:
        baseline_deviation_result: Dictionary returned by calculate_baseline_deviation().
        higher_tdi_threshold: TDI threshold for 'higher_deviation' (default: 60.0).
        moderate_tdi_threshold: TDI threshold for 'moderate_deviation' (default: 30.0).
        top_features_count: Number of largest deviating features to include.

    Returns:
        Dict[str, Any]: Structured baseline interpretation.
    """
    status = baseline_deviation_result.get("baseline_status", "unavailable")

    if status != "available":
        return {
            "baseline_status": status,
            "typing_deviation_index": None,
            "deviation_level": "unavailable",
            "largest_deviations": [],
            "message": baseline_deviation_result.get(
                "message", "Personal typing baseline is unavailable or requires more sessions."
            ),
            "disclaimer": ASSESSMENT_DISCLAIMER,
        }

    raw_tdi = baseline_deviation_result.get("typing_deviation_index")
    tdi = round(float(raw_tdi), 2) if raw_tdi is not None else 0.0

    # Classify deviation level neutrally
    if tdi >= higher_tdi_threshold:
        deviation_level = DEVIATION_LEVEL_HIGHER
        deviation_desc = "Current typing cadence shows substantial statistical deviation from historical baseline."
    elif tdi >= moderate_tdi_threshold:
        deviation_level = DEVIATION_LEVEL_MODERATE
        deviation_desc = "Current typing cadence shows moderate statistical deviation from historical baseline."
    else:
        deviation_level = DEVIATION_LEVEL_EXPECTED
        deviation_desc = "Current typing cadence is within expected historical variance."

    # Rank feature deviations
    raw_feats = baseline_deviation_result.get("features", {})
    ranked_feats = rank_feature_deviations(raw_feats, top_n=top_features_count)

    return {
        "baseline_status": "available",
        "typing_deviation_index": tdi,
        "deviation_level": deviation_level,
        "deviation_description": deviation_desc,
        "largest_deviations": ranked_feats,
        "features_evaluated_count": len(raw_feats),
        "disclaimer": ASSESSMENT_DISCLAIMER,
    }
