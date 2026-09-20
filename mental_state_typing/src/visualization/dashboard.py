"""Dashboard layout and Streamlit UI component helper.

Provides reusable skeuomorphic presentation components for the Streamlit web interface.
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
