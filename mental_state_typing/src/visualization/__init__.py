"""Visualization package for typing metrics and behavioral trends."""

from src.visualization.charts import (
    create_baseline_comparison_chart,
    create_correction_activity_chart,
    create_flight_time_timeline,
    create_hold_time_histogram,
    create_pause_timeline_chart,
    create_strain_gauge,
    create_tdi_breakdown_chart,
    create_timing_timeline_chart,
    create_typing_rate_chart,
    validate_chart_privacy,
)
from src.visualization.dashboard import (
    render_console_header,
    render_detected_signals_card,
    render_instrument_readouts,
    render_metric_readout,
    render_privacy_guard_banner,
    render_quality_console_card,
    render_section_divider,
    render_status_badge,
    render_system_masthead,
)
from src.visualization.theme import apply_workstation_theme

__all__ = [
    "apply_workstation_theme",
    "create_baseline_comparison_chart",
    "create_correction_activity_chart",
    "create_flight_time_timeline",
    "create_hold_time_histogram",
    "create_pause_timeline_chart",
    "create_strain_gauge",
    "create_tdi_breakdown_chart",
    "create_timing_timeline_chart",
    "create_typing_rate_chart",
    "render_console_header",
    "render_detected_signals_card",
    "render_instrument_readouts",
    "render_metric_readout",
    "render_privacy_guard_banner",
    "render_quality_console_card",
    "render_section_divider",
    "render_status_badge",
    "render_system_masthead",
    "validate_chart_privacy",
]
