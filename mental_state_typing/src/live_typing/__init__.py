"""Live typing behavior capture engine package.

Provides privacy-first, zero-text live keystroke dynamic capture, normalization,
feature extraction, quality validation, and readiness gating.
"""

from src.live_typing.collector import KeystrokeTimingEvent, TimingCollector
from src.live_typing.event_capture import render_live_typing_box
from src.live_typing.event_normalizer import LiveEventNormalizer
from src.live_typing.event_types import KeyEventType, RawBrowserEvent, TypingEvent
from src.live_typing.feature_buffer import (
    CANONICAL_SEQUENCE_FEATURES,
    LiveFeatureBuffer,
)
from src.live_typing.live_pipeline import LiveTypingPipeline
from src.live_typing.privacy_filter import (
    FORBIDDEN_PAYLOAD_FIELDS,
    PrivacyViolationError,
    audit_payload_for_sensitive_keys,
    sanitize_event_batch,
    sanitize_raw_event,
)
from src.live_typing.session import LiveTypingSession, SessionStatus
from src.live_typing.validation import validate_session_quality

__all__ = [
    "CANONICAL_SEQUENCE_FEATURES",
    "FORBIDDEN_PAYLOAD_FIELDS",
    "KeyEventType",
    "KeystrokeTimingEvent",
    "LiveEventNormalizer",
    "LiveFeatureBuffer",
    "LiveTypingPipeline",
    "LiveTypingSession",
    "PrivacyViolationError",
    "RawBrowserEvent",
    "SessionStatus",
    "TimingCollector",
    "TypingEvent",
    "audit_payload_for_sensitive_keys",
    "render_live_typing_box",
    "sanitize_event_batch",
    "sanitize_raw_event",
    "validate_session_quality",
]
