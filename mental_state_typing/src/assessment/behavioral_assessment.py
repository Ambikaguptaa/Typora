"""Behavioral Assessment Engine and Model Interpretation Layer.

Combines independent signals:
1. Deep learning sequence model prediction and uncertainty metrics.
2. Personal baseline statistical deviation (Typing Deviation Index).
3. Data-quality integrity verification.

STRICT DESIGN GUARANTEE:
Model predictions and personal baseline deviations are NEVER mathematically merged
into a composite 'mental health score'. They are evaluated as independent behavioral
indicators accompanied by explicit scientific limitations.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from src.config.settings import settings
from src.assessment.interpretation import (
    ASSESSMENT_DISCLAIMER,
    interpret_baseline_deviation,
    interpret_model_prediction,
)
from src.assessment.quality_checks import (
    STATE_BASELINE_AVAILABLE,
    STATE_INSUFFICIENT_DATA,
    STATE_MODEL_UNAVAILABLE,
    STATE_NOT_READY,
    STATE_PREDICTION_AVAILABLE,
    STATE_READY,
    validate_assessment_data,
)


@dataclass
class AssessmentInput:
    """Structured input container for behavioral session assessment."""

    user_id: str
    session_id: str
    keystroke_count: int
    predicted_class: Optional[str] = None
    class_probabilities: Optional[Dict[str, float]] = None
    baseline_deviation_result: Optional[Dict[str, Any]] = None
    sequence: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None
    model_available: bool = False
    model_version: str = "lstm-v0.4.0"
    feature_version: str = "manifest-v0.3.0"
    baseline_version: str = "baseline-v0.3.0"
    timestamp: Optional[str] = None
    is_mock: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert container to serializable dictionary."""
        data = asdict(self)
        if self.sequence is not None:
            data["sequence_shape"] = list(self.sequence.shape)
            data.pop("sequence", None)  # Don't serialize large raw tensors in metadata
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssessmentInput":
        """Instantiate from dictionary."""
        fields = cls.__dataclass_fields__
        filtered = {k: v for k, v in data.items() if k in fields}
        return cls(**filtered)


def generate_behavioral_assessment(
    input_data: AssessmentInput,
    expected_features: Optional[List[str]] = None,
    expected_sequence_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate a structured, non-diagnostic behavioral assessment from session data.

    Evaluates:
    - Data quality and observation count.
    - Model prediction, entropy, and reliability margin (if model available).
    - Personal baseline deviation, TDI, and ranked feature changes (if baseline available).
    - Deterministic, rule-based interpretation narrative.
    - Technical and scientific limitations.

    Args:
        input_data: AssessmentInput dataclass instance.
        expected_features: Optional list of expected feature names from model manifest.
        expected_sequence_length: Optional sequence window length expected by model.

    Returns:
        Dict[str, Any]: Comprehensive behavioral assessment dictionary.
    """
    ts = input_data.timestamp or datetime.now(timezone.utc).isoformat()
    baseline_status = (
        input_data.baseline_deviation_result.get("baseline_status", "unavailable")
        if input_data.baseline_deviation_result
        else "unavailable"
    )

    # 1. Quality Gate Verification
    quality_report = validate_assessment_data(
        keystroke_count=input_data.keystroke_count,
        sequence=input_data.sequence,
        feature_names=input_data.feature_names,
        expected_features=expected_features,
        expected_sequence_length=expected_sequence_length,
        model_available=input_data.model_available,
        baseline_status=baseline_status,
    )

    # 2. Model Prediction Interpretation (if available and valid)
    model_interp: Optional[Dict[str, Any]] = None
    if input_data.model_available and input_data.class_probabilities:
        model_interp = interpret_model_prediction(
            class_probabilities=input_data.class_probabilities,
            predicted_class=input_data.predicted_class,
        )
    elif not input_data.model_available:
        model_interp = {
            "status": "model_unavailable",
            "message": "Behavioral-state prediction is unavailable because a trained model has not been loaded.",
            "disclaimer": ASSESSMENT_DISCLAIMER,
        }

    # 3. Personal Baseline Interpretation (if available)
    baseline_interp: Optional[Dict[str, Any]] = None
    if input_data.baseline_deviation_result:
        baseline_interp = interpret_baseline_deviation(input_data.baseline_deviation_result)
    else:
        baseline_interp = {
            "baseline_status": "unavailable",
            "typing_deviation_index": None,
            "deviation_level": "unavailable",
            "largest_deviations": [],
            "message": "Personal typing baseline is not established for this session.",
            "disclaimer": ASSESSMENT_DISCLAIMER,
        }

    # 4. Rule-Based Deterministic Narrative Synthesis
    narrative_points: List[str] = []

    # Quality Gate State Handling
    state = quality_report["assessment_state"]
    if state == STATE_INSUFFICIENT_DATA:
        narrative_points.append(
            f"Session contains insufficient keystroke observations ({input_data.keystroke_count}). "
            "Reliable behavioral timing assessment requires at least 5 keystrokes."
        )
    elif state == STATE_NOT_READY:
        narrative_points.append(
            f"Assessment could not be conducted due to input data validation errors: {', '.join(quality_report['blocking_reasons'])}."
        )
    else:
        # Interpret Personal Baseline Signal
        if baseline_interp.get("baseline_status") == "available":
            tdi_val = baseline_interp.get("typing_deviation_index", 0.0)
            dev_level = baseline_interp.get("deviation_level")
            if dev_level == "higher_deviation":
                narrative_points.append(
                    f"Current typing cadence shows substantial statistical deviation from the user's historical baseline (TDI: {tdi_val}/100)."
                )
            elif dev_level == "moderate_deviation":
                narrative_points.append(
                    f"Current typing cadence exhibits moderate statistical variation from historical baseline (TDI: {tdi_val}/100)."
                )
            else:
                narrative_points.append(
                    f"Current typing cadence is consistent with the user's expected historical pattern (TDI: {tdi_val}/100)."
                )
        else:
            narrative_points.append(
                f"Personal baseline is not yet established ({baseline_interp.get('message')}). Continue logging sessions to build an individual profile."
            )

        # Interpret Model Prediction Signal (Separate from Baseline)
        if model_interp and model_interp.get("status") == "interpreted":
            pred_c = model_interp.get("predicted_class")
            top_p = int(round(model_interp.get("top_probability", 0.0) * 100))
            rel = model_interp.get("reliability")

            if rel == "LOW_SEPARATION":
                narrative_points.append(
                    f"The temporal sequence model assigns the highest probability ({top_p}%) to class '{pred_c}', "
                    "but exhibits low separation between competing classes."
                )
            else:
                narrative_points.append(
                    f"The temporal sequence model identifies a cadence resembling class '{pred_c}' "
                    f"with {top_p}% model probability ({rel.replace('_', ' ').lower()})."
                )
        elif not input_data.model_available:
            narrative_points.append(
                "Behavioral-state prediction is unavailable because a trained model has not been loaded."
            )

    overall_interpretation = " ".join(narrative_points)

    # 5. Scientific & Technical Limitations
    limitations = [
        "These results reflect statistical keystroke timing patterns and model predictions; they do NOT diagnose medical, neurological, or psychological conditions.",
        "Model probability reflects classification alignment with research dataset annotations, not diagnostic certainty.",
        "Personal baseline statistics reflect motor typing speed and rhythm, which can be influenced by hardware, ergonomics, fatigue, or ambient distractions.",
        "The system strictly adheres to a Zero-Text policy: no character identities, typed content, or message context are evaluated.",
    ]
    if input_data.is_mock:
        limitations.append(
            "DEVELOPMENT NOTICE: This assessment was generated using mock model outputs for pipeline validation only."
        )

    # 6. Assemble Full Assessment Document
    return {
        "assessment_id": f"asmt_{input_data.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": ts,
        "assessment_state": quality_report["assessment_state"],
        "assessment_ready": quality_report["assessment_ready"],
        "user_id": input_data.user_id,
        "session_id": input_data.session_id,
        "keystroke_count": input_data.keystroke_count,
        "is_mock": input_data.is_mock,
        "data_quality": quality_report,
        "model_prediction": model_interp,
        "personal_baseline": baseline_interp,
        "overall_interpretation": overall_interpretation,
        "limitations": limitations,
        "version_metadata": {
            "model_version": input_data.model_version,
            "feature_version": input_data.feature_version,
            "baseline_version": input_data.baseline_version,
            "system_disclaimer": settings.system_disclaimer,
        },
        "privacy": {
            "pseudonymized_user_id": input_data.user_id,
            "zero_text_guaranteed": True,
            "raw_text_stored": False,
        },
    }


def generate_assessment_summary(assessment: Dict[str, Any]) -> str:
    """Generate a clean, structured human-readable text summary of the assessment."""
    lines: List[str] = []
    lines.append("================================================================================")
    lines.append("BEHAVIORAL ASSESSMENT SUMMARY")
    lines.append("Academic Motor Cadence Analysis (Non-Diagnostic)")
    lines.append("================================================================================")
    lines.append(f"Session ID  : {assessment.get('session_id')}")
    lines.append(f"User ID     : {assessment.get('user_id')}")
    lines.append(f"State       : {assessment.get('assessment_state')}")
    lines.append(f"Observations: {assessment.get('keystroke_count')} keystrokes")
    lines.append("--------------------------------------------------------------------------------")

    # MODEL OUTPUT SECTION
    lines.append("\nMODEL OUTPUT")
    lines.append("------------")
    m_pred = assessment.get("model_prediction", {})
    if m_pred and m_pred.get("status") == "interpreted":
        lines.append(f"Predicted Class       : {m_pred.get('predicted_class')}")
        top_pct = int(round(m_pred.get('top_probability', 0.0) * 100))
        lines.append(f"Model Probability     : {top_pct}%")
        lines.append(f"Separation Margin     : {m_pred.get('probability_margin', 0.0):.2f}")
        lines.append(f"Separation Reliability: {m_pred.get('reliability')}")
        lines.append("Class Probabilities   :")
        for cls, prob in m_pred.get("class_probabilities", {}).items():
            lines.append(f"  • {cls}: {int(round(prob * 100))}%")
    else:
        lines.append("Status : " + str(m_pred.get("message", "Model prediction unavailable.")))

    # PERSONAL TYPING BASELINE SECTION
    lines.append("\nPERSONAL TYPING BASELINE")
    lines.append("------------------------")
    b_stat = assessment.get("personal_baseline", {})
    status = b_stat.get("baseline_status", "unavailable")
    lines.append(f"Baseline Status       : {status.replace('_', ' ').title()}")
    tdi = b_stat.get("typing_deviation_index")
    if tdi is not None:
        lines.append(f"Typing Deviation Index: {tdi} / 100")
        lines.append(f"Deviation Level       : {b_stat.get('deviation_level', '').replace('_', ' ').title()}")
    else:
        lines.append(f"Notice                : {b_stat.get('message', 'Baseline not established.')}")

    # LARGEST OBSERVED DEVIATIONS
    lines.append("\nLARGEST OBSERVED DEVIATIONS")
    lines.append("---------------------------")
    largest_devs = b_stat.get("largest_deviations", [])
    if largest_devs:
        for dev in largest_devs:
            feat_name = dev.get("feature", "").replace("_", " ").title()
            descriptor = dev.get("descriptor", "")
            lines.append(f"  • {feat_name}: {descriptor}")
    else:
        lines.append("  • None recorded or personal baseline pending.")

    # INTERPRETATION SECTION
    lines.append("\nINTERPRETATION")
    lines.append("--------------")
    lines.append(assessment.get("overall_interpretation", "No interpretation generated."))

    # LIMITATION SECTION
    lines.append("\nIMPORTANT LIMITATIONS")
    lines.append("---------------------")
    for lim in assessment.get("limitations", []):
        lines.append(f"  • {lim}")

    lines.append("================================================================================")
    return "\n".join(lines)


def save_assessment_report(
    assessment: Dict[str, Any],
    output_dir: Optional[Union[str, Path]] = None,
) -> str:
    """Serialize assessment report as machine-readable JSON in data/processed/assessments/.

    Guarantees Zero-Text policy: only pseudonymous IDs and timing statistics are written.

    Args:
        assessment: Assessment dictionary from generate_behavioral_assessment().
        output_dir: Optional destination directory (defaults to settings.assessments_path).

    Returns:
        str: Absolute path of the serialized JSON file.
    """
    out_dir = Path(output_dir) if output_dir else settings.assessments_path
    out_dir.mkdir(parents=True, exist_ok=True)

    asmt_id = assessment.get("assessment_id", f"asmt_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_file = out_dir / f"{asmt_id}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(assessment, f, indent=2)

    return str(output_file)
