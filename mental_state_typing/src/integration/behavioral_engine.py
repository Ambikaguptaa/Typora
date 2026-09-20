"""Central Orchestration Engine for Live Behavioral Intelligence Integration.

Orchestrates the complete flow:
Live Capture -> Privacy Filter -> Event Normalizer -> Data Quality Gate
-> Feature Extraction & 30-Event Temporal Windows
-> Personal Baseline Branch (TDI: 0–100)
-> Sequential Model Branch (Gated: MODEL_NOT_READY until real dataset)
-> Behavioral Assessment Layer (Decoupled, Non-Diagnostic)
-> Structured JSON & Text Reports

CRITICAL DESIGN INVARIANTS:
1. Zero Raw Text: No typed characters, message text, or raw input strings are admitted.
2. Conceptual Decoupling: Model classification and personal baseline deviations remain separate.
3. Honest Gating: If real model or sufficient baseline is unavailable, returns NOT_READY without fabrication.
4. Non-Diagnostic Language: Purely behavioral motor variation, never psychiatric or medical diagnoses.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.assessment.behavioral_assessment import AssessmentInput, generate_behavioral_assessment
from src.assessment.interpretation import (
    ASSESSMENT_DISCLAIMER,
    interpret_baseline_deviation,
    interpret_model_prediction,
)
from src.config.settings import settings
from src.data_engineering.baseline import (
    DEFAULT_BASELINE_FEATURES,
    build_user_baseline,
    calculate_baseline_deviation,
    load_user_baselines,
    save_user_baselines,
)
from src.data_engineering.feature_engineering import build_feature_table
from src.deep_learning.data_split import load_scaler, transform_sequence
from src.deep_learning.evaluate import EVALUATION_DISCLAIMER
from src.deep_learning.label_encoder import load_label_mapping
from src.integration.pipeline_state import PipelineState, SessionType
from src.integration.result_schema import (
    BaselineResult,
    BehavioralAssessmentResult,
    DataQualityResult,
    FeatureSummary,
    ModelResult,
)
from src.live_typing.event_types import RawBrowserEvent, TypingEvent
from src.live_typing.feature_buffer import CANONICAL_SEQUENCE_FEATURES, LiveFeatureBuffer
from src.live_typing.privacy_filter import (
    PrivacyViolationError,
    audit_payload_for_sensitive_keys,
    sanitize_event_batch,
)
from src.live_typing.session import LiveTypingSession, SessionStatus
from src.live_typing.validation import validate_session_quality

logger = logging.getLogger(__name__)


class BehavioralEngine:
    """Central orchestrator for the live typing behavioral intelligence pipeline."""

    def __init__(
        self,
        user_id: str = "participant_local",
        session_type: SessionType = SessionType.ANALYSIS,
        min_calibration_sessions: int = 5,
        sequence_length: int = 30,
        sequence_stride: int = 10,
        pause_threshold_ms: float = 500.0,
        models_dir: Optional[Union[str, Path]] = None,
        baselines_dir: Optional[Union[str, Path]] = None,
        assessments_dir: Optional[Union[str, Path]] = None,
        mock_model_engine: Optional[Any] = None,
    ):
        self.user_id = user_id
        self.session_type = session_type
        self.min_calibration_sessions = min_calibration_sessions
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.pause_threshold_ms = pause_threshold_ms

        self.models_dir = Path(models_dir) if models_dir else settings.models_path
        self.baselines_dir = Path(baselines_dir) if baselines_dir else settings.baselines_path
        self.assessments_dir = Path(assessments_dir) if assessments_dir else settings.processed_data_path / "assessments"

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.baselines_dir.mkdir(parents=True, exist_ok=True)
        self.assessments_dir.mkdir(parents=True, exist_ok=True)

        # Session & buffers
        self.session = LiveTypingSession()
        self.feature_buffer = LiveFeatureBuffer(
            pause_threshold_ms=pause_threshold_ms,
            sequence_length=sequence_length,
            sequence_stride=sequence_stride,
        )

        # Pipeline state
        self.state: PipelineState = PipelineState.IDLE
        self.last_result: Optional[BehavioralAssessmentResult] = None

        # Test-only mock model engine (strictly isolated, None in production)
        self.mock_model_engine = mock_model_engine

        # User historical calibration session features in memory / disk
        self._user_history_file = self.baselines_dir / f"history_{self.user_id}.json"
        self.user_calibration_history: List[Dict[str, Any]] = self._load_user_history()

    def _load_user_history(self) -> List[Dict[str, Any]]:
        """Load user's historical calibration session records."""
        if self._user_history_file.exists():
            try:
                with open(self._user_history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed loading calibration history: {e}")
                return []
        return []

    def _save_user_history(self) -> None:
        """Persist user's historical calibration session records."""
        try:
            with open(self._user_history_file, "w", encoding="utf-8") as f:
                json.dump(self.user_calibration_history, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed saving calibration history: {e}")

    def set_session_type(self, session_type: SessionType) -> None:
        """Switch session mode (CALIBRATION or ANALYSIS)."""
        self.session_type = session_type

    def start_session(self, session_type: Optional[SessionType] = None) -> str:
        """Start a new typing session."""
        if session_type:
            self.session_type = session_type
        self.session.start()
        self.feature_buffer.clear()
        self.state = PipelineState.CAPTURING
        self.last_result = None
        return self.session.session_id

    def pause_session(self) -> None:
        """Pause the current session."""
        self.session.pause()

    def resume_session(self) -> None:
        """Resume a paused session."""
        self.session.resume()
        self.state = PipelineState.CAPTURING

    def reset_session(self) -> str:
        """Completely reset session state, buffers, and indicators."""
        self.session.reset()
        self.feature_buffer.clear()
        self.state = PipelineState.IDLE
        self.last_result = None
        return self.session.session_id

    def ingest_raw_events(
        self,
        raw_events: List[Dict[str, Any]],
        strict_privacy: bool = True,
    ) -> int:
        """Ingest, audit, sanitize, normalize, and buffer raw browser keystroke events.

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

        all_paired = self.session.get_paired_events()
        new_events = all_paired[initial_count:]
        self.feature_buffer.add_events(new_events)

        if added > 0 and self.state in (PipelineState.CAPTURING, PipelineState.COLLECTING_DATA):
            self.state = PipelineState.COLLECTING_DATA

        return added

    def check_model_readiness(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate all 9 production model readiness criteria.

        Checks:
        1. model file exists ('lstm_model.keras')
        2. model metadata exists ('training_metadata.json')
        3. is_production_model == true
        4. trained_on_real_dataset == true
        5. feature manifest matches ('feature_manifest.json')
        6. sequence length matches (30)
        7. feature count matches (6)
        8. scaler exists ('feature_scaler.pkl')
        9. label mapping exists ('label_mapping.json')

        Returns:
            Tuple[bool, str, Dict[str, Any]]: (is_ready, reason, details)
        """
        # Test-only mock engine bypass (strictly marked is_mock=True)
        if self.mock_model_engine is not None and getattr(self.mock_model_engine, "is_mock", False):
            return True, "Mock test model engine active.", {"is_mock": True}

        model_path = self.models_dir / "lstm_model.keras"
        meta_path = self.models_dir / "training_metadata.json"
        manifest_path = self.models_dir / "feature_manifest.json"
        scaler_path = self.models_dir / "feature_scaler.pkl"
        label_path = self.models_dir / "label_mapping.json"

        # 1. Model file
        if not model_path.exists():
            return False, "No verified production LSTM model file exists.", {"missing": "lstm_model.keras"}

        # 2. Metadata file
        if not meta_path.exists():
            return False, "Model training metadata file is missing.", {"missing": "training_metadata.json"}

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            return False, f"Failed loading model metadata: {e}", {}

        # 3. is_production_model
        if not metadata.get("is_production_model", False):
            return False, "Available model is a development/smoke artifact, not an approved production model.", metadata

        # 4. trained_on_real_dataset
        if not metadata.get("trained_on_real_dataset", False):
            return False, "Model was not trained on an approved real research dataset.", metadata

        # 5. Manifest
        if not manifest_path.exists():
            return False, "Feature manifest is missing from model directory.", {"missing": "feature_manifest.json"}

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            return False, f"Failed loading feature manifest: {e}", {}

        # 6. Sequence length
        if manifest.get("sequence_length") != self.sequence_length:
            return (
                False,
                f"Sequence length mismatch: model expects {manifest.get('sequence_length')}, engine uses {self.sequence_length}.",
                manifest,
            )

        # 7. Feature count
        if manifest.get("num_features") != len(CANONICAL_SEQUENCE_FEATURES):
            return (
                False,
                f"Feature count mismatch: model expects {manifest.get('num_features')}, engine uses {len(CANONICAL_SEQUENCE_FEATURES)}.",
                manifest,
            )

        # 8. Scaler
        if not scaler_path.exists():
            return False, "Fitted feature scaler is missing.", {"missing": "feature_scaler.pkl"}

        # 9. Label mapping
        if not label_path.exists():
            return False, "Target label mapping file is missing.", {"missing": "label_mapping.json"}

        return True, "Verified production model trained on real research data is available.", metadata

    def check_baseline_readiness(self) -> Tuple[bool, str, int, int, Optional[Dict[str, Any]]]:
        """Verify personal baseline calibration progress and profile availability.

        Returns:
            Tuple[bool, str, int, int, Optional[Dict[str, Any]]]:
                (is_ready, message, completed_sessions, min_required, profile)
        """
        completed = len(self.user_calibration_history)
        min_required = self.min_calibration_sessions

        if completed < min_required:
            msg = (
                f"Participant has completed {completed}/{min_required} calibration sessions. "
                "Minimum 5 calibration sessions required to establish an individual baseline."
            )
            return False, msg, completed, min_required, None

        # Build baseline profile using canonical baseline analysis
        history_df = pd.DataFrame(self.user_calibration_history)
        candidate_cols = [
            c for c in history_df.columns
            if c not in ("user_id", "session_id", "session_type", "timestamp")
        ]
        profile = build_user_baseline(
            history_df,
            user_id=self.user_id,
            min_sessions=min_required,
            baseline_features=candidate_cols if candidate_cols else None,
        )
        if profile and "features" in profile:
            profile["means"] = {
                k: v.get("mean", 0.0) for k, v in profile["features"].items() if isinstance(v, dict)
            }
        msg = f"Personal baseline established from {completed} calibration sessions."
        return True, msg, completed, min_required, profile

    def stop_session(self) -> BehavioralAssessmentResult:
        """Stop typing session, execute full intelligence pipeline, and produce assessment result.

        Returns:
            BehavioralAssessmentResult: Complete structured assessment report.
        """
        self.session.stop()
        ts = datetime.now(timezone.utc).isoformat()

        # 1. Technical Data-Quality Verification
        is_quality_valid, quality_verdict, quality_details = validate_session_quality(self.session)
        quality_res = DataQualityResult(
            is_valid=is_quality_valid,
            verdict="PASS" if is_quality_valid else "FAIL",
            reasons=quality_details.get("reasons", []),
            metrics=quality_details.get("metrics", {}),
        )

        telemetry_dict = self.feature_buffer.extract_telemetry()
        feature_summary = FeatureSummary(**telemetry_dict)

        # If data quality gate fails: stop inference, report QUALITY_CHECK_FAILED
        if not is_quality_valid:
            self.state = PipelineState.QUALITY_CHECK_FAILED
            b_ready, b_msg, b_count, b_req, _ = self.check_baseline_readiness()
            m_ready, m_reason, _ = self.check_model_readiness()

            res = BehavioralAssessmentResult(
                session_id=self.session.session_id,
                session_type=self.session_type.value,
                timestamp=ts,
                pipeline_status=self.state.value,
                data_quality=quality_res,
                feature_summary=feature_summary,
                baseline_result=BaselineResult(
                    status="READY" if b_ready else "NOT_READY",
                    session_count=b_count,
                    min_required_sessions=b_req,
                    message=b_msg,
                ),
                model_result=ModelResult(
                    status="MODEL_READY" if m_ready else "MODEL_NOT_READY",
                    reason=m_reason,
                ),
                assessment_result=None,
                warnings=["Session contains insufficient micro-timing data for behavioral assessment."],
            )
            self.last_result = res
            return res

        self.state = PipelineState.FEATURES_READY

        # 2. Canonical Sequence Windows Generation (N, 30, 6)
        X_seq, meta_seq = self.feature_buffer.generate_sequence_windows(
            sequence_length=self.sequence_length,
            sequence_stride=self.sequence_stride,
        )
        quality_res.metrics["sequence_windows_count"] = len(X_seq)

        # 3. Session Feature Table Construction
        df_events = self.feature_buffer.to_dataframe(
            user_id=self.user_id,
            session_id=self.session.session_id,
        )
        session_features_df, _ = build_feature_table(
            df_events,
            user_col="user_id",
            session_col="session_id",
            timestamp_col="press_time",
            pause_threshold_ms=self.pause_threshold_ms,
        )
        current_session_features = (
            session_features_df.iloc[0].to_dict() if not session_features_df.empty else {}
        )
        if current_session_features:
            current_session_features["session_id"] = self.session.session_id
            current_session_features["user_id"] = self.user_id

        # 4. Personal Baseline Branch
        # If CALIBRATION session, store in history
        if self.session_type == SessionType.CALIBRATION and current_session_features:
            self.user_calibration_history.append(current_session_features)
            self._save_user_history()

        b_ready, b_msg, b_count, b_req, b_profile = self.check_baseline_readiness()
        baseline_res = BaselineResult(
            status="READY" if b_ready else "NOT_READY",
            session_count=b_count,
            min_required_sessions=b_req,
            message=b_msg,
        )

        baseline_deviation_dict: Optional[Dict[str, Any]] = None
        if b_ready and b_profile and current_session_features:
            baseline_deviation_dict = calculate_baseline_deviation(
                current_session=current_session_features,
                user_baseline=b_profile,
            )
            baseline_res.typing_deviation_index = baseline_deviation_dict.get("typing_deviation_index")
            baseline_res.deviations = baseline_deviation_dict.get("features")
            baseline_res.message = "Personal baseline comparison complete."

        # 5. Model Inference Branch (Honest Gating)
        m_ready, m_reason, m_details = self.check_model_readiness()
        model_res: ModelResult

        if not m_ready:
            model_res = ModelResult(
                status="MODEL_NOT_READY",
                reason=m_reason,
            )
        else:
            # Model is ready (or test mock engine is active)
            try:
                if self.mock_model_engine is not None and getattr(self.mock_model_engine, "is_mock", False):
                    # Test mock prediction
                    pred_res = self.mock_model_engine.predict_sequences(X_seq)
                    model_res = ModelResult(
                        status="MODEL_READY",
                        reason="Test mock prediction successful.",
                        predicted_class=pred_res["predicted_class"],
                        class_probabilities=pred_res["class_probabilities"],
                        top_probability=pred_res.get("top_probability"),
                        second_probability=pred_res.get("second_probability"),
                        probability_margin=pred_res.get("probability_margin"),
                        normalized_entropy=pred_res.get("normalized_entropy"),
                        reliability=pred_res.get("reliability"),
                        is_mock=True,
                    )
                else:
                    # Verified production model inference
                    import tensorflow as tf
                    scaler, feature_cols = load_scaler(self.models_dir / "feature_scaler.pkl")
                    label_map = load_label_mapping(self.models_dir / "label_mapping.json")
                    id_to_label = {v: k for k, v in label_map.items()}

                    # Transform sequence with fitted scaler without leakage
                    X_scaled = transform_sequence(scaler, X_seq)
                    model = tf.keras.models.load_model(str(self.models_dir / "lstm_model.keras"))
                    batch_preds = model.predict(X_scaled, verbose=0)
                    mean_probs = np.mean(batch_preds, axis=0)

                    class_probs = {id_to_label[i]: float(p) for i, p in enumerate(mean_probs)}
                    interp = interpret_model_prediction(class_probs)

                    model_res = ModelResult(
                        status="MODEL_READY",
                        reason="Production sequence inference complete.",
                        predicted_class=interp["predicted_class"],
                        class_probabilities=class_probs,
                        top_probability=interp.get("top_probability"),
                        second_probability=interp.get("second_probability"),
                        probability_margin=interp.get("probability_margin"),
                        normalized_entropy=interp.get("normalized_entropy"),
                        reliability=interp.get("reliability"),
                        is_mock=False,
                    )
            except Exception as e:
                logger.error(f"Inference failure: {e}")
                model_res = ModelResult(
                    status="MODEL_NOT_READY",
                    reason=f"Model inference failed: {e}",
                )

        # 6. Behavioral Assessment Layer Synthesis
        asmt_input = AssessmentInput(
            user_id=self.user_id,
            session_id=self.session.session_id,
            keystroke_count=self.feature_buffer.event_count,
            predicted_class=model_res.predicted_class,
            class_probabilities=model_res.class_probabilities,
            baseline_deviation_result=baseline_deviation_dict,
            sequence=X_seq if len(X_seq) > 0 else None,
            feature_names=CANONICAL_SEQUENCE_FEATURES,
            model_available=(model_res.status == "MODEL_READY"),
            is_mock=model_res.is_mock,
            timestamp=ts,
        )
        asmt_doc = generate_behavioral_assessment(
            input_data=asmt_input,
            expected_features=CANONICAL_SEQUENCE_FEATURES,
            expected_sequence_length=self.sequence_length,
        )

        # 7. Final Assessment Result Assembly
        self.state = PipelineState.ASSESSMENT_COMPLETE

        final_res = BehavioralAssessmentResult(
            session_id=self.session.session_id,
            session_type=self.session_type.value,
            timestamp=ts,
            pipeline_status=self.state.value,
            data_quality=quality_res,
            feature_summary=feature_summary,
            baseline_result=baseline_res,
            model_result=model_res,
            assessment_result=asmt_doc,
            privacy_status="ENFORCED",
            warnings=asmt_doc.get("warnings", []),
            disclaimer=EVALUATION_DISCLAIMER,
        )

        # Persist report JSON to local filesystem
        try:
            report_file = self.assessments_dir / f"assessment_{self.session.session_id}.json"
            final_res.save_json(report_file)
        except Exception as e:
            logger.warning(f"Failed to persist assessment JSON: {e}")

        # Persist report to database for cloud deployment durability
        try:
            from database.database import save_assessment_record
            save_assessment_record(final_res.to_dict())
        except Exception as e:
            logger.warning(f"Failed to persist assessment to database: {e}")

        self.last_result = final_res
        return final_res
