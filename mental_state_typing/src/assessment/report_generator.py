"""Structured Report Generation Module for Behavioral Typing Intelligence.

Generates human-readable, privacy-safe text and Markdown session reports:
- Decouples Data Quality, Personal Baseline, Model Assessment, and Uncertainty.
- Enforces strict Zero-Raw-Text invariant: never includes character identities or typed text.
- Formulates findings strictly in behavioral and non-diagnostic research terminology.
"""

from typing import Any, Dict, Optional

from src.deep_learning.evaluate import EVALUATION_DISCLAIMER
from src.integration.result_schema import BehavioralAssessmentResult


def generate_markdown_report(result: BehavioralAssessmentResult) -> str:
    """Generate a clean, structured Markdown report from a BehavioralAssessmentResult.

    Args:
        result: Consolidated BehavioralAssessmentResult dataclass.

    Returns:
        str: GitHub-flavored Markdown formatted report.
    """
    lines = []
    lines.append("# Behavioral Session Intelligence Report")
    lines.append("")
    lines.append(f"**Session Identifier**: `{result.session_id}`  ")
    lines.append(f"**Session Mode**: `{result.session_type}`  ")
    lines.append(f"**Timestamp (UTC)**: `{result.timestamp}`  ")
    lines.append(f"**Pipeline State**: `{result.pipeline_status}`  ")
    lines.append(f"**Privacy Protocol**: `{result.privacy_status}`  ")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1. Technical Data Quality
    lines.append("## 1. Technical Data Quality")
    lines.append("")
    q_badge = "🟢 PASS" if result.data_quality.is_valid else "🔴 FAIL"
    lines.append(f"- **Verdict**: {q_badge} (`{result.data_quality.verdict}`)")
    m = result.data_quality.metrics
    lines.append(f"- **Paired Events**: {m.get('total_paired_events', result.feature_summary.event_count)}")
    lines.append(f"- **Active Duration**: {m.get('duration_seconds', result.feature_summary.duration_seconds):.1f} seconds")
    lines.append(f"- **Valid Dwell Observations**: {m.get('valid_dwell_count', 'N/A')}")
    lines.append(f"- **Valid Flight Observations**: {m.get('valid_flight_count', 'N/A')}")
    lines.append(f"- **Temporal Sequence Windows (30-step)**: {m.get('sequence_windows_count', 0)}")
    if result.data_quality.reasons:
        lines.append("- **Quality Observations**:")
        for r in result.data_quality.reasons:
            lines.append(f"  - {r}")
    lines.append("")

    # 2. Personal Baseline Analysis
    lines.append("## 2. Personal Typing Baseline")
    lines.append("")
    b = result.baseline_result
    if b.status == "READY":
        lines.append(f"- **Status**: 🟢 READY (Calibrated from {b.session_count} sessions)")
        if b.typing_deviation_index is not None:
            lines.append(f"- **Typing Deviation Index (TDI)**: `{b.typing_deviation_index:.1f} / 100`")
        lines.append(f"- **Interpretation**: {b.message}")
        if b.deviations:
            lines.append("- **Observed Motor Deviations**:")
            for feat, d in b.deviations.items():
                cur = d.get("current_value")
                mean = d.get("baseline_mean")
                direction = d.get("direction", "within_baseline")
                lines.append(f"  - `{feat}`: current {cur} vs baseline {mean} ({direction})")
    else:
        lines.append(f"- **Status**: 🟡 NOT READY ({b.session_count}/{b.min_required_sessions} Calibration Sessions)")
        lines.append(f"- **Guidance**: {b.message}")
        lines.append("- *Notice: An individual motor baseline requires at least 5 sessions to establish longitudinal validity.*")
    lines.append("")

    # 3. Model Classification & Uncertainty
    lines.append("## 3. Deep Learning Sequence Model")
    lines.append("")
    m_res = result.model_result
    if m_res.status == "MODEL_READY":
        lines.append(f"- **Status**: 🟢 READY (Production inference complete)")
        lines.append(f"- **Predicted Class**: `{m_res.predicted_class}`")
        if m_res.class_probabilities:
            lines.append("- **Model Output Probabilities**:")
            for cls_name, prob in m_res.class_probabilities.items():
                lines.append(f"  - `{cls_name}`: {prob*100:.1f}%")
        if m_res.reliability:
            lines.append(f"- **Separation Reliability**: `{m_res.reliability}`")
        if m_res.probability_margin is not None:
            lines.append(f"- **Probability Margin**: `{m_res.probability_margin:.3f}`")
        if m_res.normalized_entropy is not None:
            lines.append(f"- **Normalized Entropy**: `{m_res.normalized_entropy:.3f}`")
    else:
        lines.append(f"- **Status**: 🔒 MODEL_NOT_READY")
        lines.append(f"- **Reason**: {m_res.reason}")
        lines.append("- *Academic Policy: Production LSTM inference is gated until an approved research dataset is provided in `data/raw/`. No fake predictions are generated.*")
    lines.append("")

    # 4. Behavioral Dynamics Telemetry
    lines.append("## 4. Fine-Motor Dynamics Telemetry")
    lines.append("")
    f = result.feature_summary
    lines.append(f"- **Mean Dwell Time**: {f.mean_dwell_ms:.1f} ms (Median: {f.median_dwell_ms:.1f} ms)")
    lines.append(f"- **Mean Flight Time**: {f.mean_flight_ms:.1f} ms (Median: {f.median_flight_ms:.1f} ms)")
    lines.append(f"- **Estimated Cadence**: {f.estimated_wpm:.1f} WPM")
    lines.append(f"- **Cognitive Pauses**: {f.pause_count} ({f.pause_rate*100:.1f}% of key transitions)")
    lines.append(f"- **Editing & Corrections**: {f.backspace_count} backspaces, {f.correction_count} rapid bursts (Error rate: {f.error_rate*100:.1f}%)")
    lines.append("")

    # 5. Non-Diagnostic Narrative
    if result.assessment_result and "overall_interpretation" in result.assessment_result:
        lines.append("## 5. Behavioral Synthesis")
        lines.append("")
        lines.append(f"> {result.assessment_result['overall_interpretation']}")
        lines.append("")

    # 6. Disclaimers & Ethics
    lines.append("---")
    lines.append("### Research Disclaimers & Privacy Guarantees")
    lines.append(f"- 🔒 **Zero-Text Policy**: Character values and typed text strings were suppressed at intake.")
    lines.append(f"- ⚠️ **Academic Disclaimer**: {result.disclaimer}")

    return "\n".join(lines)


def generate_text_report(result: BehavioralAssessmentResult) -> str:
    """Generate a clean, structured plaintext report."""
    md = generate_markdown_report(result)
    # Strip basic markdown markers for clean ASCII terminal presentation
    text = (
        md.replace("## ", "\n--- ")
        .replace("# ", "=== ")
        .replace("**", "")
        .replace("`", "")
        .replace("> ", "")
    )
    return text
