"""Streamlit custom component wrapper for the live typing capture engine.

Bridges the browser-side vanilla HTML/JS timing event listener with the Python
runtime via Streamlit's custom components API.

Guarantees:
- Zero global hooks (no pynput, no keyboard, no system-level listeners).
- Focus is strictly constrained to the component's internal textarea.
- Received data contains exclusively relative timestamps and non-text tokens.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import streamlit as st
import streamlit.components.v1 as components

_FRONTEND_DIR = Path(__file__).parent / "frontend"

# Declare the Streamlit custom component pointing to the frontend directory
_live_typing_component = components.declare_component(
    "live_typing_box",
    path=str(_FRONTEND_DIR),
)


def render_live_typing_box(
    session_active: bool = False,
    session_paused: bool = False,
    prompt_text: Optional[str] = None,
    reset_signal: bool = False,
    telemetry: Optional[Dict[str, Any]] = None,
    key: str = "live_typing_box",
) -> Optional[List[Dict[str, Any]]]:
    """Render the privacy-enforced live typing input box and capture timing event batches.

    Args:
        session_active: Whether keyboard event capture is currently active (START pressed, not COMPLETED).
        session_paused: Whether session is in PAUSED state (active but not accumulating).
        prompt_text: Standardized text prompt presented to participant.
        reset_signal: Trigger signal to clear the client-side text area and counters.
        telemetry: Dict with keys: events_captured, feature_rows, windows_ready.
        key: Unique Streamlit component key.

    Returns:
        Optional[List[Dict[str, Any]]]: Batch of sanitized timing event dictionaries, or None.
    """
    default_prompt = (
        "Type naturally as if composing a short paragraph. Your timing cadence is what is being analyzed, "
        "not the content of your words."
    )

    return _live_typing_component(
        session_active=session_active,
        session_paused=session_paused,
        prompt_text=prompt_text or default_prompt,
        reset_signal=reset_signal,
        telemetry=telemetry or {"events_captured": 0, "feature_rows": 0, "windows_ready": 0},
        key=key,
        default=None,
    )
