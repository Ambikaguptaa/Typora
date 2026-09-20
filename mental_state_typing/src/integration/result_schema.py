"""Structured, privacy-safe result schemas for behavioral assessment integration.

ZERO-TEXT INVARIANT:
Schemas strictly encapsulate timing metrics, statistical deviations, model outputs,
and technical quality indicators. No character identities, typed content, or browser
event strings are ever admitted into these structures.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.deep_learning.evaluate import EVALUATION_DISCLAIMER
from src.integration.pipeline_state import PipelineState, SessionType
from src.live_typing.privacy_filter import FORBIDDEN_PAYLOAD_FIELDS, audit_payload_for_sensitive_keys


@dataclass
class DataQualityResult:
    """Technical data-quality validation report for a session."""

    is_valid: bool
    verdict: str  # 'PASS' / 'FAIL' or 'SESSION_VALID' / 'INSUFFICIENT_DATA'
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureSummary:
    """Aggregated physiological and behavioral typing dynamic metrics."""

    event_count: int = 0
    duration_seconds: float = 0.0
    mean_dwell_ms: float = 0.0
    std_dwell_ms: float = 0.0
    median_dwell_ms: float = 0.0
    mean_flight_ms: float = 0.0
    std_flight_ms: float = 0.0
    median_flight_ms: float = 0.0
    pause_count: int = 0
    pause_rate: float = 0.0
    estimated_wpm: float = 0.0
    backspace_count: int = 0
    correction_count: int = 0
    error_rate: float = 0.0
    sequence_windows_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BaselineResult:
    """Personal typing baseline analysis and deviation evaluation."""

    status: str  # 'READY' / 'NOT_READY'
    session_count: int = 0
    min_required_sessions: int = 5
    typing_deviation_index: Optional[float] = None
    deviations: Optional[Dict[str, Any]] = None
    message: str = ""
    disclaimer: str = (
        "Personal typing baseline measures individual motor consistency over time "
        "and does not diagnose medical or psychological conditions."
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelResult:
    """Sequential deep learning model inference and uncertainty evaluation."""

    status: str  # 'MODEL_NOT_READY' / 'MODEL_READY'
    reason: str = ""
    predicted_class: Optional[str] = None
    class_probabilities: Optional[Dict[str, float]] = None
    top_probability: Optional[float] = None
    second_probability: Optional[float] = None
    probability_margin: Optional[float] = None
    normalized_entropy: Optional[float] = None
    reliability: Optional[str] = None  # HIGH_SEPARATION / MODERATE_SEPARATION / LOW_SEPARATION
    is_mock: bool = False
    disclaimer: str = EVALUATION_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BehavioralAssessmentResult:
    """Consolidated session assessment integrating quality, baseline, and model signals.

    STRICT DESIGN GUARANTEE:
    Model classifications and personal baseline deviations remain conceptually decoupled
    and are never mathematically combined into a single composite diagnosis.
    """

    session_id: str
    session_type: str  # 'CALIBRATION' / 'ANALYSIS'
    timestamp: str
    pipeline_status: str  # PipelineState.value
    data_quality: DataQualityResult
    feature_summary: FeatureSummary
    baseline_result: BaselineResult
    model_result: ModelResult
    assessment_result: Optional[Dict[str, Any]] = None
    privacy_status: str = "ENFORCED"
    warnings: List[str] = field(default_factory=list)
    disclaimer: str = (
        "Academic Research Instrument: This system estimates fine-motor behavioral "
        "typing dynamics and does not provide medical or psychiatric diagnoses."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert assessment result to privacy-audited serializable dictionary."""
        d = asdict(self)
        # Deep audit for sensitive raw text keys before export
        sensitive = audit_payload_for_sensitive_keys(d)
        if sensitive:
            raise ValueError(f"Privacy violation: sensitive keys detected in assessment payload: {sensitive}")
        return d

    def save_json(self, file_path: Path) -> Path:
        """Serialize assessment safely to JSON file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return file_path

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BehavioralAssessmentResult":
        """Reconstruct assessment object from dictionary safely."""
        dq_data = data.get("data_quality", {})
        feat_data = data.get("feature_summary", {})
        base_data = data.get("baseline_result", {})
        model_data = data.get("model_result", {})

        return cls(
            session_id=data.get("session_id", "sess_unknown"),
            session_type=data.get("session_type", SessionType.ANALYSIS.value),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            pipeline_status=data.get("pipeline_status", PipelineState.ASSESSMENT_COMPLETE.value),
            data_quality=DataQualityResult(**dq_data) if dq_data else DataQualityResult(is_valid=False, verdict="UNKNOWN"),
            feature_summary=FeatureSummary(**feat_data) if feat_data else FeatureSummary(),
            baseline_result=BaselineResult(**base_data) if base_data else BaselineResult(status="NOT_READY"),
            model_result=ModelResult(**model_data) if model_data else ModelResult(status="MODEL_NOT_READY"),
            assessment_result=data.get("assessment_result"),
            privacy_status=data.get("privacy_status", "ENFORCED"),
            warnings=data.get("warnings", []),
            disclaimer=data.get("disclaimer", cls.disclaimer),
        )
