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

    IDLE = "IDLE"
    CAPTURING = "CAPTURING"
    COLLECTING_DATA = "COLLECTING_DATA"
    FEATURES_READY = "FEATURES_READY"
    QUALITY_CHECK_FAILED = "QUALITY_CHECK_FAILED"
    BASELINE_NOT_READY = "BASELINE_NOT_READY"
    MODEL_NOT_READY = "MODEL_NOT_READY"
    READY_FOR_INFERENCE = "READY_FOR_INFERENCE"
    INFERENCE_COMPLETE = "INFERENCE_COMPLETE"
    ASSESSMENT_COMPLETE = "ASSESSMENT_COMPLETE"
    ERROR = "ERROR"
