"""Plotly chart generation module for typing dynamics metadata.

Visualizes hold times, flight times, pauses, typing cadence, corrections,
and personal baseline divergence comparisons.
All charts operate strictly on privacy-safe engineered numerical features.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.live_typing.privacy_filter import PrivacyViolationError, audit_payload_for_sensitive_keys


def validate_chart_privacy(data: Any) -> None:
    """Validate that input data for charts contains no raw text or sensitive keys.

    Args:
        data: Dict, DataFrame, Series, or list to inspect.

    Raises:
        PrivacyViolationError: If sensitive keys or text strings are detected.
    """
    if isinstance(data, dict):
        violations = audit_payload_for_sensitive_keys(data)
        if violations:
            raise PrivacyViolationError(f"Sensitive raw text keys detected in chart data: {violations}")
        for v in data.values():
            if isinstance(v, (dict, list)):
                validate_chart_privacy(v)
    elif isinstance(data, pd.DataFrame):
        for col in data.columns:
            if isinstance(col, str):
                violations = audit_payload_for_sensitive_keys({col: 0})
                if violations:
                    raise PrivacyViolationError(f"Sensitive raw text column in chart DataFrame: {col}")
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                validate_chart_privacy(item)


def create_hold_time_histogram(hold_times_ms: List[float]) -> go.Figure:
    """Generate a histogram figure of key hold times."""
    validate_chart_privacy({"hold_times_ms": hold_times_ms})
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
        template="plotly_dark",
        paper_bgcolor="#15181E",
        plot_bgcolor="#15181E",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def create_flight_time_timeline(flight_times_ms: List[float]) -> go.Figure:
    """Generate a sequence scatter line of inter-key flight times."""
    validate_chart_privacy({"flight_times_ms": flight_times_ms})
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
        template="plotly_dark",
        paper_bgcolor="#15181E",
        plot_bgcolor="#15181E",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def create_strain_gauge(strain_score: float) -> go.Figure:
    """Generate an indicator gauge for behavioral mental strain estimation."""
    clamped_score = max(0.0, min(100.0, float(strain_score)))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=clamped_score,
            title={"text": "Estimated Behavioral Strain Score (0-100)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#6366F1"},
                "steps": [
                    {"range": [0, 35], "color": "#1C3829"},
                    {"range": [35, 70], "color": "#3B3214"},
                    {"range": [70, 100], "color": "#3D1A24"},
                ],
                "threshold": {
                    "line": {"color": "#FF667A", "width": 4},
                    "thickness": 0.75,
                    "value": 70,
                },
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#15181E",
        margin=dict(l=30, r=30, t=50, b=20),
        height=260,
    )
    return fig


def create_timing_timeline_chart(
    dwell_times: List[float],
    flight_times: List[float],
) -> go.Figure:
    """Generate a dual-trace timeline showing key hold (dwell) and flight durations.

    Shows the temporal typing rhythm across sequential keystrokes.
    """
    validate_chart_privacy({"dwell": dwell_times, "flight": flight_times})

    fig = go.Figure()
    indices = list(range(len(dwell_times)))

    fig.add_trace(
        go.Scatter(
            x=indices,
            y=dwell_times,
            mode="lines+markers",
            name="Dwell Time (Hold)",
            line=dict(color="#35D6FF", width=1.5),
            marker=dict(size=4, color="#35D6FF"),
        )
    )

    flight_indices = list(range(len(flight_times)))
    fig.add_trace(
        go.Scatter(
            x=flight_indices,
            y=flight_times,
            mode="lines+markers",
            name="Flight Time (Latency)",
            line=dict(color="#45E0A8", width=1.5),
            marker=dict(size=4, color="#45E0A8"),
        )
    )

    fig.update_layout(
        title="Keystroke Timing Rhythm (Dwell & Flight Latencies)",
        xaxis_title="Keystroke Event Index",
        yaxis_title="Duration (ms)",
        template="plotly_dark",
        paper_bgcolor="#15181E",
        plot_bgcolor="#15181E",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        height=300,
    )
    return fig


def create_pause_timeline_chart(
    pause_durations: List[float],
    event_indices: Optional[List[int]] = None,
    threshold_ms: float = 500.0,
) -> go.Figure:
    """Generate a timeline chart of hesitations / pauses during typing.

    Visualizes motor cognitive pause distributions throughout the session.
    """
    validate_chart_privacy({"pauses": pause_durations})

    if event_indices is None or len(event_indices) != len(pause_durations):
        event_indices = list(range(len(pause_durations)))

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=event_indices,
            y=pause_durations,
            name="Pause Duration",
            marker_color="#FFB84D",
            opacity=0.85,
        )
    )

    # Reference line for pause threshold
    fig.add_hline(
        y=threshold_ms,
        line_dash="dot",
        line_color="#8993A4",
        annotation_text=f"Threshold ({threshold_ms:.0f} ms)",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Motor Hesitations & Pauses (≥ Threshold)",
        xaxis_title="Event Sequence Index",
        yaxis_title="Pause Duration (ms)",
        template="plotly_dark",
        paper_bgcolor="#15181E",
        plot_bgcolor="#15181E",
        margin=dict(l=20, r=20, t=50, b=20),
        height=280,
    )
    return fig


def create_typing_rate_chart(
    rate_values: List[float],
    indices: Optional[List[int]] = None,
    unit: str = "WPM",
) -> go.Figure:
    """Generate a timeline showing how typing speed / cadence evolves.

    Shows motor velocity stability or slowing during typing.
    """
    validate_chart_privacy({"rates": rate_values})

    if indices is None or len(indices) != len(rate_values):
        indices = list(range(len(rate_values)))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=indices,
            y=rate_values,
            mode="lines+markers",
            name=f"Cadence ({unit})",
            line=dict(color="#8B7CFF", width=2),
            marker=dict(size=4, color="#8B7CFF"),
            fill="tozeroy",
            fillcolor="rgba(139, 124, 255, 0.12)",
        )
    )

    fig.update_layout(
        title=f"Typing Cadence Over Session ({unit})",
        xaxis_title="Event Window Index",
        yaxis_title=f"Typing Cadence ({unit})",
        template="plotly_dark",
        paper_bgcolor="#15181E",
        plot_bgcolor="#15181E",
        margin=dict(l=20, r=20, t=50, b=20),
        height=280,
    )
    return fig


def create_correction_activity_chart(
    backspace_flags: List[int],
    indices: Optional[List[int]] = None,
) -> go.Figure:
    """Generate a chart showing error correction and backspace activity.

    Shows editing bursts and cumulative correction frequency.
    """
    validate_chart_privacy({"backspaces": backspace_flags})

    if indices is None or len(indices) != len(backspace_flags):
        indices = list(range(len(backspace_flags)))

    cumulative = np.cumsum(backspace_flags).tolist()

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=indices,
            y=backspace_flags,
            name="Correction Event",
            marker_color="#FF667A",
            opacity=0.7,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=indices,
            y=cumulative,
            mode="lines",
            name="Cumulative Corrections",
            yaxis="y2",
            line=dict(color="#FFB84D", width=2),
        )
    )

    fig.update_layout(
        title="Editing & Error Correction Activity",
        xaxis_title="Event Index",
        yaxis=dict(title="Instantaneous Flag (0 or 1)", tickmode="linear", tick0=0, dtick=1),
        yaxis2=dict(
            title="Cumulative Corrections",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        template="plotly_dark",
        paper_bgcolor="#15181E",
        plot_bgcolor="#15181E",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        height=280,
    )
    return fig


def create_baseline_comparison_chart(
    current_features: Dict[str, float],
    baseline_profile: Dict[str, Any],
) -> go.Figure:
    """Generate a comparative radar or bar chart contrasting current session with personal baseline.

    Uses neutral, non-diagnostic terminology: 'Difference from personal baseline'.
    """
    validate_chart_privacy(current_features)
    validate_chart_privacy(baseline_profile)

    metrics = [
        ("mean_dwell_ms", "Dwell Time (ms)"),
        ("mean_flight_ms", "Flight Time (ms)"),
        ("pause_rate", "Pause Rate (%)"),
        ("estimated_wpm", "Cadence (WPM)"),
        ("backspace_count", "Corrections"),
    ]

    categories: List[str] = []
    current_vals: List[float] = []
    baseline_vals: List[float] = []

    # Support both direct 'means' dictionary and baseline profile 'features' dictionary
    means = baseline_profile.get("means", {})
    if not means and "features" in baseline_profile:
        feat_dict = baseline_profile.get("features", {})
        means = {
            k: (v.get("mean", 0.0) if isinstance(v, dict) else float(v))
            for k, v in feat_dict.items()
        }

    for key, label in metrics:
        if key in current_features and key in means:
            categories.append(label)
            c_val = current_features[key]
            b_val = means[key]
            if key == "pause_rate":
                c_val = c_val * 100.0 if c_val <= 1.0 else c_val
                b_val = b_val * 100.0 if b_val <= 1.0 else b_val
            current_vals.append(round(float(c_val), 2))
            baseline_vals.append(round(float(b_val), 2))

    if not categories:
        # Fallback comparison if specific keys differ
        for k in list(current_features.keys())[:5]:
            if k in means:
                categories.append(k.replace("_", " ").title())
                current_vals.append(round(float(current_features[k]), 2))
                baseline_vals.append(round(float(means[k]), 2))

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Personal Baseline",
            x=categories,
            y=baseline_vals,
            marker_color="#8993A4",
            opacity=0.7,
        )
    )

    fig.add_trace(
        go.Bar(
            name="Current Session",
            x=categories,
            y=current_vals,
            marker_color="#35D6FF",
            opacity=0.9,
        )
    )

    fig.update_layout(
        title="Difference from Personal Baseline (Direct Parameter Comparison)",
        barmode="group",
        xaxis_title="Typing Dynamics Metric",
        yaxis_title="Observed Value",
        template="plotly_dark",
        paper_bgcolor="#15181E",
        plot_bgcolor="#15181E",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        height=320,
    )
    return fig


def create_tdi_breakdown_chart(component_deviations: Dict[str, float]) -> go.Figure:
    """Generate a horizontal divergence bar chart showing feature z-score contributions.

    Labels deviations as +/- standard deviations (sigma) from historical baseline.
    """
    validate_chart_privacy(component_deviations)

    features = list(component_deviations.keys())
    z_scores = [round(float(v), 2) for v in component_deviations.values()]
    cleaned_names = [f.replace("_", " ").title() for f in features]

    colors = [
        "#FF667A" if abs(z) >= 2.0 else "#FFB84D" if abs(z) >= 1.0 else "#45E0A8"
        for z in z_scores
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=cleaned_names,
            x=z_scores,
            orientation="h",
            marker_color=colors,
            text=[f"{z:+.2f}σ" for z in z_scores],
            textposition="outside",
        )
    )

    # Reference lines for +- 1 sigma and +- 2 sigma
    fig.add_vline(x=0, line_width=1.5, line_color="#8993A4")
    fig.add_vline(x=1.0, line_dash="dot", line_color="#FFB84D", line_width=1)
    fig.add_vline(x=-1.0, line_dash="dot", line_color="#FFB84D", line_width=1)

    fig.update_layout(
        title="Typing Deviation Index (TDI) Feature Divergence Breakdown",
        xaxis_title="Deviation from Individual Baseline (Z-Score σ)",
        yaxis_title="Motor Dynamic Feature",
        template="plotly_dark",
        paper_bgcolor="#15181E",
        plot_bgcolor="#15181E",
        margin=dict(l=20, r=40, t=50, b=20),
        height=300,
    )
    return fig
