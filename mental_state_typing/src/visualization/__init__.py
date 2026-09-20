"""Visualization package for typing metrics and behavioral trends."""

from src.visualization.charts import (
    create_flight_time_timeline,
    create_hold_time_histogram,
    create_strain_gauge,
)
from src.visualization.dashboard import (
    render_console_header,
    render_detected_signals_card,
    render_instrument_readouts,
    render_privacy_guard_banner,
    render_quality_console_card,
)
from src.visualization.theme import apply_workstation_theme

__all__ = [
    "create_flight_time_timeline",
    "create_hold_time_histogram",
    "create_strain_gauge",
    "apply_workstation_theme",
    "render_console_header",
    "render_instrument_readouts",
    "render_quality_console_card",
    "render_detected_signals_card",
    "render_privacy_guard_banner",
]
