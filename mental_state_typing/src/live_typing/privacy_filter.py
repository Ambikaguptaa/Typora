"""Privacy filter module for live typing browser event payloads.

Strictly intercepts and discards any payload containing character identities,
words, typed text, passwords, or input values.
Logs only generic security tokens (PRIVACY_EVENT_REJECTED) without echoing sensitive content.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.live_typing.event_types import KeyEventType, RawBrowserEvent

logger = logging.getLogger(__name__)

# Master set of disallowed payload keys that might hold textual or character content
FORBIDDEN_EVENT_KEYS: Set[str] = {
    "key",
    "char",
    "character",
    "text",
    "word",
    "sentence",
    "message",
    "typed_text",
    "password",
    "user_input",
    "content",
    "value",
    "input",
    "keystring",
    "raw_input",
    "keystrokes_text",
    "raw_text",
    "input_text",
    "val",
    "str",
}
FORBIDDEN_PAYLOAD_FIELDS: Set[str] = FORBIDDEN_EVENT_KEYS


def audit_payload_for_sensitive_keys(payload: Dict[str, Any]) -> List[str]:
    """Audit a dictionary payload for any forbidden textual keys."""
    if not isinstance(payload, dict):
        return []
    lowered_keys = {str(k).strip().lower(): k for k in payload.keys()}
    return [k for k in lowered_keys if k in FORBIDDEN_PAYLOAD_FIELDS]


# Permitted abstract key token patterns (e.g. k_alpha, k_backspace, k_enter, k_space, k_other, k_8)
ABSTRACT_TOKEN_PATTERN = re.compile(r"^k_[a-zA-Z0-9_]+$")


class PrivacyViolationError(ValueError):
    """Raised when an incoming browser payload violates the Zero-Raw-Text policy."""

    pass


def validate_browser_event(
    event_dict: Dict[str, Any],
    strict_raise: bool = False,
) -> Optional[RawBrowserEvent]:
    """Inspect and sanitize an individual raw browser event payload."""
    if not isinstance(event_dict, dict):
        logger.warning("PRIVACY_EVENT_REJECTED: Malformed event payload (non-dict).")
        return None

    lowered_keys = {str(k).strip().lower(): k for k in event_dict.keys()}

    # 1. Audit for forbidden textual keys
    forbidden_found = [k for k in lowered_keys if k in FORBIDDEN_EVENT_KEYS]
    if forbidden_found:
        logger.warning(
            f"PRIVACY_EVENT_REJECTED: Payload contains forbidden keys: {sorted(forbidden_found)}. "
            "Raw text is strictly prohibited."
        )
        if strict_raise:
            raise PrivacyViolationError(
                f"Privacy filter violation: Forbidden keys detected in event: {sorted(forbidden_found)}"
            )
        return None

    # 2. Validate event_type (support both 'event_type' and 'type')
    raw_type = str(event_dict.get("event_type", event_dict.get("type", ""))).strip().lower()
    if raw_type in ("down", "keydown"):
        event_type = KeyEventType.KEY_DOWN.value
    elif raw_type in ("up", "keyup"):
        event_type = KeyEventType.KEY_UP.value
    else:
        logger.warning("PRIVACY_EVENT_REJECTED: Unrecognized event_type.")
        return None

    # 3. Validate timestamp
    try:
        timestamp_ms = float(event_dict.get("timestamp_ms", event_dict.get("timestamp", 0.0)))
        if timestamp_ms < 0 or not (timestamp_ms == timestamp_ms):  # NaN check
            return None
    except (ValueError, TypeError):
        return None

    # 4. Validate and sanitize key token
    token = str(event_dict.get("key_token", "k_other")).strip()

    # Reject raw characters that are not abstract tokens
    if len(token) == 1:
        if strict_raise:
            raise PrivacyViolationError(f"Privacy filter violation: Raw character detected: '{token}'")
        return None

    if not ABSTRACT_TOKEN_PATTERN.match(token):
        if token.lower() in ("backspace", "enter", "space", "tab"):
            token = f"k_{token.lower()}"
        else:
            token = "k_other"

    # 5. Extract boolean modifier flags
    is_backspace = bool(event_dict.get("is_backspace", token == "k_backspace"))
    is_enter = bool(event_dict.get("is_enter", token == "k_enter"))
    is_space = bool(event_dict.get("is_space", token == "k_space"))

    return RawBrowserEvent(
        event_type=event_type,
        timestamp_ms=timestamp_ms,
        key_token=token,
        is_backspace=is_backspace,
        is_enter=is_enter,
        is_space=is_space,
    )


# Alias for backwards compatibility
sanitize_raw_event = validate_browser_event


def sanitize_event_batch(
    raw_events: List[Dict[str, Any]],
    strict_raise: bool = False,
) -> Tuple[List[RawBrowserEvent], int]:
    """Sanitize a batch of raw browser events, filtering out invalid or forbidden events."""
    if not isinstance(raw_events, list):
        return [], 0

    clean_batch: List[RawBrowserEvent] = []
    rejected_count = 0
    for ev in raw_events:
        valid_ev = validate_browser_event(ev, strict_raise=strict_raise)
        if valid_ev is not None:
            clean_batch.append(valid_ev)
        else:
            rejected_count += 1

    return clean_batch, rejected_count

