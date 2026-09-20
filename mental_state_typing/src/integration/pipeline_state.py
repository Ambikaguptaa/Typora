"""Pipeline state and session type enumerations for behavioral intelligence orchestration.

Guarantees deterministic state transitions across the end-to-end typing pipeline:
Live Capture -> Quality Gate -> Feature Extraction -> Baseline Analysis -> Model Gating -> Assessment.
"""

from enum import Enum


class SessionType(str, Enum):
    """Explicit mode selected for a typing session."""

    CALIBRATION = "CALIBRATION"  # Dedicated to establishing/updating individual baseline
    ANALYSIS = "ANALYSIS"        # Standard evaluation against established baseline & model


class PipelineState(str, Enum):
    """Deterministic lifecycle and processing states for the behavioral engine."""

    READY = "READY"
    CAPTURING = "CAPTURING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    VALIDATING = "VALIDATING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"

    # Compatibility aliases for pipeline inspection
    IDLE = "READY"
    ACTIVE = "CAPTURING"
    COLLECTING_DATA = "CAPTURING"
    FEATURES_READY = "COMPLETED"
    QUALITY_CHECK_FAILED = "COMPLETED"
    BASELINE_NOT_READY = "COMPLETED"
    MODEL_NOT_READY = "COMPLETED"
    READY_FOR_INFERENCE = "VALIDATING"
    INFERENCE_COMPLETE = "COMPLETED"
    ASSESSMENT_COMPLETE = "COMPLETED"
