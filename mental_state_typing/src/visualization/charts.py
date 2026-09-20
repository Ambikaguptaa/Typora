"""Plotly chart generation module for typing dynamics metadata.

Visualizes hold times, flight times, and estimated strain gauges.
"""

from typing import List
import plotly.graph_objects as go


def create_hold_time_histogram(hold_times_ms: List[float]) -> go.Figure:
    """Generate a histogram figure of key hold times."""
    fig = go.Figure(
        data=[
            go.Histogram(
                x=hold_times_ms,
                nbinsx=20,
                marker_color="#3B82F6",
                opacity=0.8,
            )
        ]
    )
    fig.update_layout(
        title="Key Hold Duration Distribution",
        xaxis_title="Hold Time (ms)",
        yaxis_title="Count",
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def create_flight_time_timeline(flight_times_ms: List[float]) -> go.Figure:
    """Generate a sequence scatter line of inter-key flight times."""
    fig = go.Figure(
        data=[
            go.Scatter(
                y=flight_times_ms,
                mode="lines+markers",
                marker=dict(size=6, color="#10B981"),
                line=dict(color="#10B981", width=1.5),
            )
        ]
    )
    fig.update_layout(
        title="Inter-Key Flight Times Across Keystroke Sequence",
        xaxis_title="Keystroke Transition Index",
        yaxis_title="Flight Time (ms)",
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def create_strain_gauge(strain_score: float) -> go.Figure:
    """Generate an indicator gauge for behavioral mental strain estimation."""
    clamped_score = max(0.0, min(100.0, strain_score))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=clamped_score,
            title={"text": "Estimated Behavioral Strain Score (0-100)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#6366F1"},
                "steps": [
                    {"range": [0, 35], "color": "#DCFCE7"},
                    {"range": [35, 70], "color": "#FEF3C7"},
                    {"range": [70, 100], "color": "#FEE2E2"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 70,
                },
            },
        )
    )
    fig.update_layout(
        margin=dict(l=30, r=30, t=50, b=20),
        height=280,
    )
    return fig
