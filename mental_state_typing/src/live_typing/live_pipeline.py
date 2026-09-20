"""Live Typing Behavior Pipeline Orchestrator.

Coordinates session lifecycle, raw event ingestion, privacy filtration,
micro-timing normalization, feature buffering, quality validation, and
model/baseline readiness gating.

CRITICAL INVARIANTS:
1. Zero Raw Text: No typed characters or sensitive fields are ever accepted or stored.
2. Honest Model Gating: If no production model trained on real research data is available,
   reports MODEL_NOT_READY without fabricating predictions or probabilities.
3. Personal Baseline Gating: If fewer than 5 calibration sessions exist, reports
   BASELINE_NOT_READY without fabricating baselines.
4. Non-Diagnostic Disclaimer: Outputs are strictly research motor measurements,
   never clinical or medical diagnoses.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config.settings import settings
from src.data_engineering.baseline import (
    DEFAULT_BASELINE_FEATURES,
    build_user_baseline,
    calculate_baseline_deviation,
)
from src.data_engineering.feature_engineering import build_feature_table
from src.deep_learning.evaluate import EVALUATION_DISCLAIMER
from src.deep_learning.train import inspect_real_dataset_availability
from src.live_typing.event_types import KeyEventType, RawBrowserEvent, TypingEvent
from src.live_typing.feature_buffer import CANONICAL_SEQUENCE_FEATURES, LiveFeatureBuffer
from src.live_typing.privacy_filter import (
    PrivacyViolationError,
    audit_payload_for_sensitive_keys,
    sanitize_event_batch,
)
from src.live_typing.session import LiveTypingSession, SessionStatus
from src.live_typing.validation import validate_session_quality


class LiveTypingPipeline:
    """End-to-end orchestrator for live typing capture, feature extraction, and model gating."""

    def __init__(
        self,
        user_id: str = "local_participant",
        min_calibration_sessions: int = 5,
        sequence_length: int = 30,
        sequence_stride: int = 10,
        pause_threshold_ms: float = 500.0,
    ):
        self.user_id = user_id
        self.min_calibration_sessions = min_calibration_sessions
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.pause_threshold_ms = pause_threshold_ms

        self.session = LiveTypingSession()
        self.feature_buffer = LiveFeatureBuffer(
            pause_threshold_ms=pause_threshold_ms,
            sequence_length=sequence_length,
            sequence_stride=sequence_stride,
        )
        # Historical session feature rows for this user to compute personal baseline
        self.user_session_history: List[Dict[str, Any]] = []

    def start_session(self) -> str:
        """Start a new active typing session."""
        self.session.start()
        self.feature_buffer.clear()
        return self.session.session_id

    def pause_session(self) -> None:
        """Pause the current session."""
        self.session.pause()

    def resume_session(self) -> None:
        """Resume a paused session."""
        self.session.resume()

    def stop_session(self) -> Dict[str, Any]:
        """Stop current session and perform technical data quality validation.

        Returns:
            Dict[str, Any]: Session quality validation summary.
        """
        self.session.stop()
        is_valid, verdict, details = validate_session_quality(self.session)

        # If valid, compute session-level features and record in historical session history
        if is_valid:
            df = self.feature_buffer.to_dataframe(
                user_id=self.user_id,
                session_id=self.session.session_id,
            )
            if not df.empty:
                session_features_df, _ = build_feature_table(
                    df,
                    user_col="user_id",
                    session_col="session_id",
                    timestamp_col="press_time",
                )
                if not session_features_df.empty:
                    record = session_features_df.iloc[0].to_dict()
                    record["session_id"] = self.session.session_id
                    record["user_id"] = self.user_id
                    self.user_session_history.append(record)

        return details

    def reset_session(self) -> str:
        """Reset session and feature buffer completely."""
        self.session.reset()
        self.feature_buffer.clear()
        return self.session.session_id

    def ingest_event_batch(
        self,
        raw_events: List[Dict[str, Any]],
        strict_privacy: bool = True,
    ) -> int:
        """Ingest, filter, normalize, and buffer raw browser keystroke events.

        Args:
            raw_events: Raw event dictionaries from frontend bridge.
            strict_privacy: If True, raises PrivacyViolationError on forbidden fields.

        Returns:
            int: Number of new paired events buffered.
        """
        initial_count = len(self.session.get_paired_events())
        added = self.session.ingest_browser_batch(
            raw_events=raw_events,
            strict_privacy=strict_privacy,
        )

        # Update feature buffer with newly paired events
        all_paired = self.session.get_paired_events()
        new_events = all_paired[initial_count:]
        self.feature_buffer.add_events(new_events)

        return added

    def get_model_readiness(self) -> Dict[str, Any]:
        """Evaluate whether a verified production model trained on real research data is ready.

        Returns:
            Dict[str, Any]: Model readiness status and explanation.
        """
        dataset_info = inspect_real_dataset_availability()
        real_data_available = dataset_info.get("real_dataset_available", False)

        model_file = settings.models_path / "lstm_model.keras"
        scaler_file = settings.models_path / "feature_scaler.pkl"
        metadata_file = settings.models_path / "training_metadata.json"

        is_production_ready = (
            real_data_available
            and model_file.exists()
            and scaler_file.exists()
        )

        if not is_production_ready:
            return {
                "status": "MODEL_NOT_READY",
                "ready": False,
                "reason": "No verified production model trained on real research data.",
                "details": (
                    "In strict accordance with project data engineering ethics and scientific standards, "
                    "production LSTM sequence inference is gated until an approved, real labeled dataset "
                    "is placed in data/raw/ and the model is trained on real data. "
                    "Fabricating predictions, confidence scores, or mental state labels is strictly prohibited."
                ),
                "model_artifact_exists": model_file.exists(),
                "real_dataset_available": real_data_available,
                "disclaimer": EVALUATION_DISCLAIMER,
            }

        return {
            "status": "MODEL_READY",
            "ready": True,
            "reason": "Production model trained on verified real dataset is available.",
            "model_path": str(model_file),
            "disclaimer": EVALUATION_DISCLAIMER,
        }

    def get_baseline_readiness(self) -> Dict[str, Any]:
        """Evaluate whether the user has accumulated sufficient calibration sessions.

        Returns:
            Dict[str, Any]: Baseline readiness report and calibration progress.
        """
        completed_sessions = len(self.user_session_history)
        min_required = self.min_calibration_sessions

        if completed_sessions < min_required:
            return {
                "status": "BASELINE_NOT_READY",
                "ready": False,
                "completed_sessions": completed_sessions,
                "required_sessions": min_required,
                "remaining_sessions": min_required - completed_sessions,
                "reason": f"Minimum {min_required} calibration sessions required to establish a personal baseline.",
                "message": (
                    f"Participant has completed {completed_sessions}/{min_required} calibration sessions. "
                    "Additional sessions are required before individual motor deviation can be computed."
                ),
                "disclaimer": (
                    "Typing baseline analysis measures individual motor consistency over time "
                    "and does not diagnose or screen for psychological or medical conditions."
                ),
            }

        # Build baseline
        history_df = pd.DataFrame(self.user_session_history)
        baseline_profile = build_user_baseline(
            history_df,
            user_id=self.user_id,
            min_sessions=min_required,
        )

        return {
            "status": "BASELINE_READY",
            "ready": True,
            "completed_sessions": completed_sessions,
            "required_sessions": min_required,
            "baseline_profile": baseline_profile,
            "disclaimer": (
                "Typing baseline analysis measures individual motor consistency over time "
                "and does not diagnose or screen for psychological or medical conditions."
            ),
        }

    def evaluate_live_session(self) -> Dict[str, Any]:
        """Perform comprehensive evaluation of the current live typing session.

        Guarantees:
        - Validates technical session quality.
        - Gates model predictions honestly (returns MODEL_NOT_READY if no real model exists).
        - Gates personal baseline honestly (returns BASELINE_NOT_READY if < 5 sessions).
        - Includes non-diagnostic disclaimers.

        Returns:
            Dict[str, Any]: Full live assessment payload.
        """
        is_valid, quality_verdict, quality_details = validate_session_quality(self.session)
        telemetry = self.feature_buffer.extract_telemetry()
        model_gate = self.get_model_readiness()
        baseline_gate = self.get_baseline_readiness()

        # If technical data quality is insufficient, cannot compute further
        if not is_valid:
            return {
                "session_id": self.session.session_id,
                "user_id": self.user_id,
                "status": "INSUFFICIENT_DATA",
                "quality_details": quality_details,
                "telemetry": telemetry,
                "model_status": model_gate["status"],
                "baseline_status": baseline_gate["status"],
                "reason": "Session does not satisfy technical data quality criteria for analysis.",
                "disclaimer": EVALUATION_DISCLAIMER,
            }

        # Check sequence windows
        X_seq, meta_seq = self.feature_buffer.generate_sequence_windows(
            sequence_length=self.sequence_length,
            sequence_stride=self.sequence_stride,
        )

        # Baseline deviation if baseline is ready
        baseline_results: Optional[Dict[str, Any]] = None
        if baseline_gate["ready"] and self.user_session_history:
            current_features = self.user_session_history[-1]
            profile = baseline_gate["baseline_profile"]
            baseline_results = calculate_baseline_deviation(current_features, profile)

        return {
            "session_id": self.session.session_id,
            "user_id": self.user_id,
            "status": "COMPLETED",
            "quality_verdict": quality_verdict,
            "telemetry": telemetry,
            "sequence_windows_count": len(X_seq),
            "model_assessment": {
                "status": model_gate["status"],
                "message": model_gate["reason"],
                "details": model_gate.get("details", ""),
                "prediction": None,  # No fabricated predictions allowed
                "probabilities": None,
            },
            "baseline_assessment": {
                "status": baseline_gate["status"],
                "message": baseline_gate.get("message", "Personal baseline established."),
                "results": baseline_results,
            },
            "disclaimer": EVALUATION_DISCLAIMER,
        }

    def get_live_engine_status(self) -> Dict[str, Any]:
        """Return high-level operational status of the live typing engine."""
        return {
            "session_id": self.session.session_id,
            "user_id": self.user_id,
            "session_state": self.session.status.value,
            "event_count": self.feature_buffer.event_count,
            "active_duration_seconds": round(self.session.duration_seconds, 2),
            "privacy_guard": "READY and ENFORCED",
            "model_status": self.get_model_readiness()["status"],
            "baseline_status": self.get_baseline_readiness()["status"],
            "disclaimer": EVALUATION_DISCLAIMER,
        }
