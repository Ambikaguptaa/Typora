"""Tests for Plotly Visualization Chart generators and privacy data validation."""

import plotly.graph_objects as go
import pytest

from src.live_typing.privacy_filter import PrivacyViolationError
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
)


def test_timing_timeline_chart_generation():
    """Verify that create_timing_timeline_chart returns a valid Plotly Figure."""
    dwells = [85.0, 92.4, 78.1, 105.0]
    flights = [110.0, 125.2, 95.8, 140.1]
    fig = create_timing_timeline_chart(dwells, flights)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # Dwell and Flight traces
    assert fig.data[0].name == "Dwell Time (Hold)"
    assert fig.data[1].name == "Flight Time (Latency)"


def test_pause_timeline_chart_generation():
    """Verify that create_pause_timeline_chart returns a valid Plotly Figure."""
    pauses = [120.0, 650.0, 410.0, 920.0]
    fig = create_pause_timeline_chart(pauses, threshold_ms=500.0)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].name == "Pause Duration"


def test_typing_rate_chart_generation():
    """Verify that create_typing_rate_chart returns a valid Plotly Figure."""
    rates = [42.0, 48.5, 51.0, 47.8]
    fig = create_typing_rate_chart(rates, unit="WPM")

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert "Cadence (WPM)" in fig.data[0].name


def test_correction_activity_chart_generation():
    """Verify that create_correction_activity_chart returns instantaneous and cumulative traces."""
    flags = [0, 1, 0, 0, 1, 0]
    fig = create_correction_activity_chart(flags)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # Bar flag and cumulative line


def test_baseline_comparison_chart_generation():
    """Verify that create_baseline_comparison_chart generates comparative bars."""
    current = {
        "mean_dwell_ms": 95.0,
        "mean_flight_ms": 118.0,
        "pause_rate": 0.05,
        "estimated_wpm": 49.0,
        "backspace_count": 2,
    }
    profile = {
        "means": {
            "mean_dwell_ms": 88.0,
            "mean_flight_ms": 112.0,
            "pause_rate": 0.04,
            "estimated_wpm": 52.0,
            "backspace_count": 1,
        }
    }
    fig = create_baseline_comparison_chart(current, profile)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # Baseline bar and Current bar
    assert fig.data[0].name == "Personal Baseline"
    assert fig.data[1].name == "Current Session"


def test_tdi_breakdown_chart_generation():
    """Verify that create_tdi_breakdown_chart plots component z-scores."""
    z_scores = {
        "mean_dwell_ms": 1.25,
        "mean_flight_ms": -0.85,
        "pause_rate": 0.42,
        "estimated_wpm": 1.60,
    }
    fig = create_tdi_breakdown_chart(z_scores)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert len(fig.data[0].x) == 4


def test_charts_reject_sensitive_raw_text_payloads():
    """Verify that passing raw text fields to charts raises PrivacyViolationError."""
    tainted_dict = {"mean_dwell_ms": 90.0, "typed_text": "Sensitive message"}
    with pytest.raises(PrivacyViolationError):
        create_baseline_comparison_chart(tainted_dict, {"means": {}})
