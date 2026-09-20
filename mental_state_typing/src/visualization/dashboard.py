"""Dashboard layout and Streamlit UI component helper.

Provides reusable skeuomorphic presentation components for the Streamlit web interface.
Includes console headers, instrument readout grids, quality panels, signal indicators,
privacy banners, system masthead, metric readouts, and status badges.
"""

from typing import Any, Dict, List, Optional
import streamlit as st


def render_console_header(
    title: str,
    subtitle: str = "",
    status_text: str = "READY",
    status_type: str = "ready",
) -> None:
    """Render a physical console header card with an illuminated status LED."""
    html = f"""
    <div class="console-header-card">
        <div class="console-title-group">
            <h2 class="console-title">{title}</h2>
            {f'<div class="console-subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
        <div class="led-indicator">
            <span class="led-dot {status_type}"></span>
            <span>{status_text}</span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_system_masthead(
    version: str = "v0.3.0",
    build_label: str = "BEHAVIORAL WORKSTATION",
) -> None:
    """Render a prominent sidebar masthead with project branding and version badge.

    This replaces the inline sidebar header HTML with a reusable component
    using the system-masthead CSS classes.
    """
    html = f"""
    <div style="padding: 12px 0 16px 0; border-bottom: 1px solid #1E232B; margin-bottom: 14px;">
        <div class="system-masthead-title">
            RESEARCH CONSOLE {version}
        </div>
        <div class="system-masthead-subtitle">
            {build_label}
        </div>
        <div style="margin-top: 6px;">
            <span class="system-masthead-badge">BUILD {version}</span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_metric_readout(
    label: str,
    value: str,
    unit: str = "",
    accent: str = "#E9EEF5",
) -> None:
    """Render a single instrument-style metric readout cell.

    Args:
        label: Uppercase metric label text.
        value: Primary display value (rendered in monospace).
        unit: Optional unit suffix (e.g. 'ms', 'WPM', '%').
        accent: CSS color for the value text.
    """
    unit_html = f'<div class="metric-readout-unit">{unit}</div>' if unit else ""
    html = f"""
    <div class="metric-readout">
        <div class="metric-readout-label">{label}</div>
        <div class="metric-readout-value" style="color: {accent};">{value}</div>
        {unit_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_status_badge(
    text: str,
    status_type: str = "ready",
) -> str:
    """Return an inline HTML status badge with LED-style coloring.

    Args:
        text: Badge text (rendered uppercase).
        status_type: One of 'ready', 'warning', 'critical', 'cyan'.

    Returns:
        HTML string for the badge (for embedding in other markup).
    """
    return f'<span class="status-badge {status_type}">{text}</span>'


def render_section_divider(label: str = "") -> None:
    """Render a horizontal section divider with an optional center label.

    Args:
        label: Optional label text centered on the divider line.
    """
    if label:
        html = f"""
        <div class="section-divider">
            <div class="section-divider-line"></div>
            <div class="section-divider-label">{label}</div>
            <div class="section-divider-line"></div>
        </div>
        """
    else:
        html = '<div style="border-top: 1px solid #1E232B; margin: 14px 0;"></div>'
    st.markdown(html, unsafe_allow_html=True)


def render_instrument_readouts(
    records: int,
    features: int,
    users: int,
    sessions: int,
) -> None:
    """Render a 4-cell instrument readout grid."""
    html = f"""
    <div class="readout-grid">
        <div class="readout-cell active-cyan">
            <div class="readout-label">RECORDS</div>
            <div class="readout-value">{records:,}</div>
        </div>
        <div class="readout-cell active-violet">
            <div class="readout-label">FEATURES</div>
            <div class="readout-value">{features}</div>
        </div>
        <div class="readout-cell active-green">
            <div class="readout-label">USERS</div>
            <div class="readout-value">{users if users > 0 else 'N/A'}</div>
        </div>
        <div class="readout-cell">
            <div class="readout-label">SESSIONS</div>
            <div class="readout-value">{sessions if sessions > 0 else 'N/A'}</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_quality_console_card(quality_summary: Dict[str, Any]) -> None:
    """Render the physical data quality panel."""
    score = quality_summary.get("quality_score", 100.0)
    badge = quality_summary.get("status_badge", "positive")
    missing_pct = quality_summary.get("missing_percentage", 0.0)
    dup_rows = quality_summary.get("duplicate_rows", 0)
    neg_durations = quality_summary.get("negative_durations", 0)

    html = f"""
    <div class="console-card">
        <div class="console-card-header">
            <span>DATA QUALITY AUDIT</span>
            <div class="led-indicator">
                <span class="led-dot {badge}"></span>
                <span>INDEX: {score}/100</span>
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
            <div class="recessed-panel" style="margin: 0; text-align: center;">
                <div class="readout-label">Missing Values</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #E9EEF5;">{missing_pct}%</div>
            </div>
            <div class="recessed-panel" style="margin: 0; text-align: center;">
                <div class="readout-label">Duplicate Rows</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #E9EEF5;">{dup_rows}</div>
            </div>
            <div class="recessed-panel" style="margin: 0; text-align: center;">
                <div class="readout-label">Invalid Durations</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: {'#45E0A8' if neg_durations == 0 else '#FF667A'};">
                    {neg_durations}
                </div>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_detected_signals_card(signals: Dict[str, bool]) -> None:
    """Render a card displaying detected behavioral signals and variables."""
    pills_html = []
    for signal_name, is_detected in signals.items():
        css_class = "detected" if is_detected else "missing"
        icon = "✓" if is_detected else "✕"
        pills_html.append(
            f'<div class="signal-pill {css_class}">{icon} {signal_name}</div>'
        )

    joined_pills = "".join(pills_html)
    html = f"""
    <div class="console-card">
        <div class="console-card-header">
            <span>DETECTED SIGNALS & VARIABLES</span>
        </div>
        <div class="signals-container">
            {joined_pills}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_privacy_guard_banner(sensitive_columns: List[str]) -> None:
    """Render a privacy alert banner if sensitive textual content is present."""
    if not sensitive_columns:
        return

    col_list_str = ", ".join([f"'{c}'" for c in sensitive_columns])
    html = f"""
    <div class="privacy-guard-banner">
        <div style="font-size: 1.4rem;">🔒</div>
        <div class="privacy-guard-text">
            <span class="privacy-guard-highlight">ZERO-TEXT PRIVACY GUARD ACTIVE:</span>
            Detected sensitive text columns ({col_list_str}).
            Content has been quarantined and masked from inspection. 
            This research strictly models timing metadata, not typed words.
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
