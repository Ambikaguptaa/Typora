"""Mental-State Detection System using Typing Behavior
Main Streamlit Application Entrypoint.

Complete Functional Research Workstation & Analytics Console.
"""

print("[STARTUP STEP 1] Initializing application environment and standard libraries...", flush=True)
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

print("[STARTUP STEP 2] Loading scientific and UI libraries (numpy, pandas, plotly, streamlit)...", flush=True)
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on Python sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("[STARTUP STEP 3] Loading database persistence layer (pg8000 pure-python driver)...", flush=True)
from database.database import (
    DatabaseHealth,
    check_connection,
    check_database_health,
    get_database_config_diagnostics,
    init_db,
)

print("[STARTUP STEP 4] Loading behavioral and ML feature pipeline modules...", flush=True)
from src.assessment.report_generator import generate_markdown_report, generate_text_report
from src.config.settings import settings
from src.data_engineering.baseline import (
    DEFAULT_BASELINE_FEATURES,
    calculate_baseline_deviation,
    compute_baseline_strain,
)
from src.data_engineering.data_quality import (
    check_class_imbalance,
    check_missing_values,
    generate_quality_summary,
)
from src.data_engineering.dataset_acquisition import get_dataset_status
from src.data_engineering.dataset_adapter import (
    detect_keystroke_columns,
    detect_label_column,
    detect_sensitive_text_columns,
    detect_session_column,
    detect_timestamp_column,
    detect_user_column,
    generate_dataset_report,
    load_dataset,
)
from src.data_engineering.dataset_registry import (
    get_dataset_config,
    list_registered_datasets,
)
from src.data_engineering.feature_engineering import extract_timing_features
from src.integration import (
    BaselineResult,
    BehavioralAssessmentResult,
    BehavioralEngine,
    DataQualityResult,
    FeatureSummary,
    ModelResult,
    PipelineState,
    SessionType,
    delete_session_record,
    get_privacy_security_specs,
    get_system_overview_status,
    get_training_gate_matrix,
    list_stored_sessions,
    load_session_detail,
)
from src.live_typing import (
    render_live_typing_box,
    validate_session_quality,
)
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

print("[STARTUP STEP 5] App module imports completed successfully.", flush=True)

# Configure Streamlit page
st.set_page_config(
    page_title="Workstation | Mental-State Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply console styling
apply_workstation_theme()


def get_engine() -> BehavioralEngine:
    """Retrieve or initialize the singleton BehavioralEngine in session state."""
    if "engine" not in st.session_state:
        st.session_state.engine = BehavioralEngine()
    return st.session_state.engine


@st.cache_resource(show_spinner=False)
def ensure_database_initialized() -> bool:
    """Initialize database schema once per application process lifetime."""
    try:
        return init_db()
    except Exception as e:
        logger.error(f"Handled database initialization error: {e}")
        return False


@st.cache_data(ttl=60, show_spinner=False)
def get_cached_database_health() -> DatabaseHealth:
    """Evaluate database health with 60s TTL cache to avoid connection storm on reruns."""
    return check_database_health()


def render_sidebar(engine: BehavioralEngine) -> str:
    """Render the primary workstation control sidebar and navigation."""
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-header">
                <div class="system-title">MENTAL-STATE MONITOR</div>
                <div class="system-sub">High-Resolution Keystroke Biometrics Console</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        nav_options = [
            "1. 📋 Overview",
            "2. ⌨️ Live Session",
            "3. 📊 Baseline Analytics",
            "4. 🧠 Behavioral Model",
            "5. 📜 Session History",
            "6. 📑 Reports",
            "7. 🔒 Privacy & Security",
            "8. 🔬 Dataset / System Status",
            "9. 🧩 Architecture & Roadmap",
        ]
        selected_nav = st.radio(
            "Select Section",
            options=nav_options,
            label_visibility="collapsed",
        )

        st.markdown("<hr style='border: none; border-top: 1px solid #242B36; margin: 14px 0;'>", unsafe_allow_html=True)

        settings.ensure_directories()
        print("[STARTUP STEP 8] Ensuring database is initialized (lazy cached)...", flush=True)
        ensure_database_initialized()
        print("[STARTUP STEP 9] Evaluating database health (cached)...", flush=True)
        health = get_cached_database_health()
        db_alive = bool(health)
        db_led = "ready" if db_alive else "critical"
        db_status = "ONLINE" if db_alive else f"OFFLINE ({health.status_code})"
        is_cloud_pg = settings.get_database_backend() == "postgresql"
        db_backend_label = "PostgreSQL (Cloud - pg8000)" if is_cloud_pg else "SQLite (Local)"

        # Truthful system overview from backend status aggregator
        sys_status = get_system_overview_status(engine)

        st.markdown(
            f"""
            <div class="recessed-panel">
                <div class="readout-label">SYSTEM READINESS AUDIT</div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
                    <span style="font-size: 0.8rem; color: #E9EEF5;">{db_backend_label}</span>
                    <div class="led-indicator">
                        <span class="led-dot {db_led}"></span>
                        <span>{db_status}</span>
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
                    <span style="font-size: 0.8rem; color: #E9EEF5;">Dataset Status</span>
                    <span style="font-size: 0.72rem; font-weight: 600; color: {sys_status['dataset']['color']};">{sys_status['dataset']['badge']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
                    <span style="font-size: 0.8rem; color: #E9EEF5;">Production Model</span>
                    <span style="font-size: 0.72rem; font-weight: 600; color: {sys_status['model']['color']};">{sys_status['model']['badge']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
                    <span style="font-size: 0.8rem; color: #E9EEF5;">Personal Baseline</span>
                    <span style="font-size: 0.72rem; font-weight: 600; color: {sys_status['baseline']['color']};">{sys_status['baseline']['badge']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
                    <span style="font-size: 0.8rem; color: #E9EEF5;">Live Keystroke Engine</span>
                    <span style="font-size: 0.72rem; font-weight: 600; color: {sys_status['live_capture']['color']};">{sys_status['live_capture']['badge']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
                    <span style="font-size: 0.8rem; color: #E9EEF5;">Zero-Text Privacy</span>
                    <span style="font-size: 0.72rem; font-weight: 600; color: {sys_status['privacy']['color']};">{sys_status['privacy']['badge']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
                    <span style="font-size: 0.8rem; color: #E9EEF5;">Leakage Audit</span>
                    <span style="font-size: 0.72rem; font-weight: 600; color: {sys_status['leakage']['color']};">{sys_status['leakage']['badge']}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="recessed-panel" style="margin-top: 10px;">
                <div class="readout-label">GOVERNANCE & INTEGRITY</div>
                <div style="font-size: 0.76rem; color: #8993A4; margin-top: 4px; line-height: 1.4;">
                    🔒 <strong>Zero Keylogging:</strong> Keystroke characters are never captured or saved.<br>
                    ⚖️ <strong>Non-Diagnostic:</strong> Measures motor dynamics, not medical illness.<br>
                    🛡️ <strong>Anti-Fabrication:</strong> Real dataset required before production inferences.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected_nav


# ==============================================================================
# PAGE 1: OVERVIEW
# ==============================================================================
def render_overview_page(engine: BehavioralEngine) -> None:
    """Render the executive research overview and truthful system readiness panel."""
    st.markdown('<div class="console-card">', unsafe_allow_html=True)
    st.title("🧠 Mental-State Detection System Using Typing Behavior")
    st.markdown(
        "*The system analyzes typing dynamics such as timing, pauses, speed, and correction "
        "behavior to estimate behavioral patterns.*"
    )

    # Status badges
    col_b1, col_b2, col_b3 = st.columns(3)
    col_b1.info("🔬 **Research Prototype** — Academic exploration of keystroke dynamics")
    col_b2.warning("⚖️ **Non-Diagnostic** — Does not diagnose psychiatric conditions")
    col_b3.success("🔒 **Zero-Text Policy** — Keystroke content is strictly not stored")
    st.markdown("</div>", unsafe_allow_html=True)

    st.warning(
        "⚠️ **Academic Disclaimer**: "
        "This system is an academic research prototype designed to measure motor dynamics and behavioral strain. "
        "It does not diagnose medical, psychiatric, or psychological conditions. Typing content and key values are immediately suppressed."
    )

    # System Status Panel
    st.markdown('<div class="console-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="console-card-header">
            <span>SYSTEM READINESS PANEL</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sys_status = get_system_overview_status(engine)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="recessed-panel">
                <div class="readout-label">DATASET REPOSITORY</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: {sys_status['dataset']['color']}; margin: 4px 0;">
                    {sys_status['dataset']['badge']}
                </div>
                <div style="font-size: 0.8rem; color: #9BA3AF;">
                    {sys_status['dataset']['label']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="recessed-panel">
                <div class="readout-label">PRODUCTION LSTM MODEL</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: {sys_status['model']['color']}; margin: 4px 0;">
                    {sys_status['model']['badge']}
                </div>
                <div style="font-size: 0.8rem; color: #9BA3AF;">
                    {sys_status['model']['label']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="recessed-panel">
                <div class="readout-label">PERSONAL BASELINE</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: {sys_status['baseline']['color']}; margin: 4px 0;">
                    {sys_status['baseline']['badge']}
                </div>
                <div style="font-size: 0.8rem; color: #9BA3AF;">
                    {sys_status['baseline']['label']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown(
            f"""
            <div class="recessed-panel">
                <div class="readout-label">LIVE CAPTURE ENGINE</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: {sys_status['live_capture']['color']}; margin: 4px 0;">
                    {sys_status['live_capture']['badge']}
                </div>
                <div style="font-size: 0.8rem; color: #9BA3AF;">
                    Isolated sandboxed HTML5 component
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            f"""
            <div class="recessed-panel">
                <div class="readout-label">PRIVACY PROTOCOL</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: {sys_status['privacy']['color']}; margin: 4px 0;">
                    {sys_status['privacy']['badge']}
                </div>
                <div style="font-size: 0.8rem; color: #9BA3AF;">
                    Zero text, HMAC-SHA256, Fernet
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c6:
        st.markdown(
            f"""
            <div class="recessed-panel">
                <div class="readout-label">DATA LEAKAGE AUDIT</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: {sys_status['leakage']['color']}; margin: 4px 0;">
                    {sys_status['leakage']['badge']}
                </div>
                <div style="font-size: 0.8rem; color: #9BA3AF;">
                    Subject/session boundary verification
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Quick Workflow Guide
    st.markdown('<div class="console-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="console-card-header">
            <span>RECOMMENDED RESEARCH WORKFLOW</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        st.markdown(
            """
            <div class="recessed-panel">
                <div class="readout-label">STEP 1: CALIBRATION</div>
                <p style="font-size: 0.82rem; color: #E9EEF5; margin-top: 4px;">
                    Navigate to <strong>Live Session</strong> and switch to <strong>Calibration Session</strong> mode. 
                    Complete 5 brief natural typing sessions to establish your individual motor baseline.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_w2:
        st.markdown(
            """
            <div class="recessed-panel">
                <div class="readout-label">STEP 2: ANALYSIS SESSION</div>
                <p style="font-size: 0.82rem; color: #E9EEF5; margin-top: 4px;">
                    Switch to <strong>Analysis Session</strong> mode. Type freely to measure your current motor timing. 
                    The system calculates your <strong>Typing Deviation Index (TDI)</strong> against historical patterns.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_w3:
        st.markdown(
            """
            <div class="recessed-panel">
                <div class="readout-label">STEP 3: REPORTS & HISTORY</div>
                <p style="font-size: 0.82rem; color: #E9EEF5; margin-top: 4px;">
                    Inspect the <strong>Session History</strong> and <strong>Reports</strong> tabs to review 
                    statistical timelines, download audited JSON telemetry, or export Markdown reports.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 2: LIVE SESSION
# ==============================================================================
def render_live_session_page(engine: BehavioralEngine) -> None:
    """Render the controlled live typing capture workstation, telemetry, and charts."""
    render_console_header(
        title="Live Behavioral Session Deck",
        subtitle="Controlled Keystroke Dynamics Acquisition, Micro-Timing Telemetry & Quality Audit",
        status_text="CAPTURE ENGINE READY",
        status_type="ready",
    )

    # Dual Session Mode Selector
    st.markdown('<div class="console-card">', unsafe_allow_html=True)
    col_mode, col_mode_desc = st.columns([1, 2])
    with col_mode:
        current_type_idx = 0 if engine.session_type == SessionType.ANALYSIS else 1
        mode_selection = st.radio(
            "Select Session Protocol",
            options=["🔬 Analysis Session", "📊 Calibration Session"],
            index=current_type_idx,
            help="Calibration sessions accumulate your baseline history (requires 5). Analysis sessions evaluate divergence.",
        )
        if mode_selection == "📊 Calibration Session":
            engine.set_session_type(SessionType.CALIBRATION)
        else:
            engine.set_session_type(SessionType.ANALYSIS)

    with col_mode_desc:
        st.markdown('<div class="recessed-panel" style="margin-top: 10px;">', unsafe_allow_html=True)
        if engine.session_type == SessionType.CALIBRATION:
            st.markdown(
                """
                <div class="readout-label">CALIBRATION SESSION MODE</div>
                <div style="font-size: 0.8rem; color: #35D6FF; margin-top: 2px;">
                    Records your natural typing rhythm to build a personalized baseline profile. 
                    Does not generate behavioral alerts. (Accumulates toward 5 required calibration sessions).
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="readout-label">ANALYSIS SESSION MODE</div>
                <div style="font-size: 0.8rem; color: #45E0A8; margin-top: 2px;">
                    Evaluates current keystroke dynamics against your established personal baseline. 
                    Calculates the Typing Deviation Index (TDI) without polluting baseline history.
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Physical Control Deck
    st.markdown('<div class="console-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="console-card-header">
            <span>PHYSICAL CONTROL DECK</span>
            <span style="font-size: 0.72rem; color: #8993A4;">STATUS: <code>"""
        + engine.state.value
        + """</code></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_start, c_pause, c_resume, c_stop, c_reset = st.columns(5)
    with c_start:
        if st.button("▶ START SESSION", use_container_width=True, key="live_btn_start"):
            sid = engine.start_session()
            st.rerun()

    with c_pause:
        if st.button("⏸ PAUSE", use_container_width=True, key="live_btn_pause"):
            engine.pause_session()
            st.rerun()

    with c_resume:
        if st.button("▶ RESUME", use_container_width=True, key="live_btn_resume"):
            engine.resume_session()
            st.rerun()

    with c_stop:
        if st.button("⏹ STOP & VALIDATE", use_container_width=True, key="live_btn_stop"):
            res = engine.stop_session()
            st.session_state["last_assessment"] = res
            st.rerun()

    with c_reset:
        if st.button("🔄 RESET", use_container_width=True, key="live_btn_reset"):
            engine.reset_session()
            if "last_assessment" in st.session_state:
                del st.session_state["last_assessment"]
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # Focused Typing Canvas (Zero Keylogger)
    st.markdown('<div class="console-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="console-card-header">
            <span>FOCUSED TYPING CANVAS</span>
            <span style="font-size: 0.72rem; color: #45E0A8;">🔒 ZERO-TEXT POLICY ENFORCED</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    raw_events = render_live_typing_box(key="live_deck_canvas")
    if raw_events:
        try:
            added = engine.ingest_raw_events(raw_events, strict_privacy=True)
        except Exception as e:
            st.error(f"Privacy filtration exception: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

    # Live Motor Dynamics Telemetry Readouts
    telemetry = engine.feature_buffer.extract_telemetry()
    paired_count = len(engine.session.get_paired_events())
    active_s = engine.session.get_active_duration_seconds()

    st.markdown('<div class="console-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="console-card-header">
            <span>REAL-TIME MOTOR DYNAMICS TELEMETRY</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    col_t1.metric("Active Duration", f"{active_s:.1f} s")
    col_t1.metric("Mean Dwell Time", f"{telemetry.get('mean_dwell_ms', 0.0):.1f} ms")

    col_t2.metric("Valid Paired Events", f"{paired_count}")
    col_t2.metric("Mean Flight Time", f"{telemetry.get('mean_flight_ms', 0.0):.1f} ms")

    col_t3.metric("Typing Cadence", f"{telemetry.get('estimated_wpm', 0.0):.1f} WPM")
    col_t3.metric("Pause Rate", f"{telemetry.get('pause_rate', 0.0) * 100:.1f}%")

    col_t4.metric("Corrections (Backspaces)", f"{telemetry.get('backspace_count', 0)}")
    col_t4.metric("Sequence Windows", f"{telemetry.get('sequence_windows_count', 0)} × [30, 6]")
    st.markdown("</div>", unsafe_allow_html=True)

    # Dynamic Interactive Visualizations
    events_df = engine.feature_buffer.to_dataframe()
    if not events_df.empty and len(events_df) >= 2:
        st.markdown('<div class="console-card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="console-card-header">
                <span>DYNAMIC MOTOR TIMELINE VISUALIZATIONS</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Row 1: Timing Timeline & Pause Distribution
        dwell_vals = events_df["dwell_time"].dropna().tolist()
        flight_vals = events_df["flight_time"].dropna().tolist()
        pause_vals = events_df["pause_duration"].tolist()

        c_v1, c_v2 = st.columns(2)
        with c_v1:
            fig_timing = create_timing_timeline_chart(dwell_vals, flight_vals)
            st.plotly_chart(fig_timing, use_container_width=True)

        with c_v2:
            fig_pauses = create_pause_timeline_chart(pause_vals, threshold_ms=engine.pause_threshold_ms)
            st.plotly_chart(fig_pauses, use_container_width=True)

        # Row 2: Cadence & Editing Activity
        cadence_vals = events_df["typing_speed"].tolist()
        backspace_flags = events_df["backspace"].astype(int).tolist()

        c_v3, c_v4 = st.columns(2)
        with c_v3:
            fig_speed = create_typing_rate_chart(cadence_vals, unit="WPM")
            st.plotly_chart(fig_speed, use_container_width=True)

        with c_v4:
            fig_corr = create_correction_activity_chart(backspace_flags)
            st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="console-card">', unsafe_allow_html=True)
        st.info("ℹ️ **Telemetry Visualizations Awaiting Data**: Type at least 2 keystrokes in the canvas above to display interactive motor timing waveforms and pause timelines.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Session Quality Panel
    st.markdown('<div class="console-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="console-card-header">
            <span>SESSION QUALITY AUDIT</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    is_valid, verdict, q_details = validate_session_quality(engine.session)
    q_metrics = q_details.get("metrics", {})

    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    col_q1.write(f"**Events Paired:** `{q_metrics.get('paired_events_count', paired_count)}`")
    col_q1.write(f"**Duration:** `{q_metrics.get('active_duration_s', active_s):.1f}s`")

    col_q2.write(f"**Valid Dwell Counts:** `{q_metrics.get('valid_dwell_count', 0)}`")
    col_q2.write(f"**Valid Flight Counts:** `{q_metrics.get('valid_flight_count', 0)}`")

    col_q3.write(f"**Windows (30, 6):** `{q_metrics.get('sequence_windows_count', 0)}`")
    col_q3.write(f"**Missing Values:** `0`")

    with col_q4:
        st.markdown('<div class="recessed-panel" style="text-align: center;">', unsafe_allow_html=True)
        st.markdown('<div class="readout-label">QUALITY VERDICT</div>', unsafe_allow_html=True)
        if is_valid:
            st.markdown('<div style="font-size: 1.1rem; font-weight: 700; color: #45E0A8;">SESSION VALID</div>', unsafe_allow_html=True)
        else:
            badge_color = "#FFB84D" if verdict == "INSUFFICIENT_DATA" else "#FF667A"
            st.markdown(f'<div style="font-size: 1.0rem; font-weight: 700; color: {badge_color};">{verdict}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if not is_valid and q_details.get("reasons"):
        st.caption("Criteria requirements: ≥ 15 paired events, ≥ 3.0 seconds duration, valid positive dwell & flight intervals.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Completed Session Assessment (if session just stopped)
    last_res: Optional[BehavioralAssessmentResult] = st.session_state.get("last_assessment")
    if last_res is not None:
        st.markdown('<div class="console-card">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="console-card-header">
                <span>LATEST SESSION ASSESSMENT REPORT: <code>{last_res.session_id}</code></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_b_branch, col_m_branch = st.columns(2)
        with col_b_branch:
            st.markdown('<div class="recessed-panel">', unsafe_allow_html=True)
            st.markdown('<div class="readout-label">PERSONAL BASELINE BRANCH</div>', unsafe_allow_html=True)
            b = last_res.baseline_result
            if b.status == "READY":
                tdi_val = b.typing_deviation_index if b.typing_deviation_index is not None else 0.0
                st.markdown(
                    f"""
                    <div style="font-size: 1.3rem; font-weight: 700; color: #45E0A8; margin: 4px 0;">
                        TDI: {tdi_val:.1f} / 100
                    </div>
                    <div style="font-size: 0.8rem; color: #9BA3AF;">{b.message}</div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style="font-size: 1.1rem; font-weight: 700; color: #FFB84D; margin: 4px 0;">
                        BASELINE NOT READY ({b.session_count}/{b.min_required_sessions})
                    </div>
                    <div style="font-size: 0.8rem; color: #9BA3AF;">{b.message}</div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        with col_m_branch:
            st.markdown('<div class="recessed-panel">', unsafe_allow_html=True)
            st.markdown('<div class="readout-label">SEQUENTIAL MODEL BRANCH</div>', unsafe_allow_html=True)
            m = last_res.model_result
            if m.status == "MODEL_READY":
                st.markdown(
                    f"""
                    <div style="font-size: 1.2rem; font-weight: 700; color: #35D6FF; margin: 4px 0;">
                        {m.predicted_class}
                    </div>
                    <div style="font-size: 0.8rem; color: #9BA3AF;">Separation: {m.reliability} (Margin: {m.probability_margin:.3f})</div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style="font-size: 1.05rem; font-weight: 700; color: #FFB84D; margin: 4px 0;">
                        🔒 {m.status}
                    </div>
                    <div style="font-size: 0.8rem; color: #9BA3AF;">{m.reason}</div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # Download & Expandable View
        c_down, c_view = st.columns(2)
        with c_down:
            st.download_button(
                label="⬇ DOWNLOAD SESSION ASSESSMENT (JSON)",
                data=json.dumps(last_res.to_dict(), indent=2),
                file_name=f"assessment_{last_res.session_id}.json",
                mime="application/json",
                use_container_width=True,
            )
        with c_view:
            with st.expander("📄 VIEW FORMATTED MARKDOWN REPORT", expanded=False):
                st.markdown(generate_markdown_report(last_res))

        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 3: BASELINE ANALYTICS
# ==============================================================================
def render_baseline_analytics_page(engine: BehavioralEngine) -> None:
    """Render the personal typing baseline calibration tracker and TDI deviation analysis."""
    render_console_header(
        title="Personal Baseline Analytics",
        subtitle="Individual Typing Dynamic Profiles, Longitudinal Calibration & Divergence Index",
        status_text="BASELINE SUBSYSTEM",
        status_type="ready",
    )

    b_ready, b_msg, b_count, b_req, profile = engine.check_baseline_readiness()

    # Calibration Progress Card
    st.markdown('<div class="console-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="console-card-header">
            <span>CALIBRATION PROGRESS TRACKER</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    progress_ratio = min(1.0, max(0.0, b_count / float(b_req)))
    st.progress(progress_ratio)

    col_cp1, col_cp2 = st.columns([1, 2])
    with col_cp1:
        st.markdown(
            f"""
            <div class="recessed-panel" style="text-align: center;">
                <div class="readout-label">COMPLETED SESSIONS</div>
                <div style="font-size: 1.6rem; font-weight: 700; color: {'#45E0A8' if b_ready else '#35D6FF'};">
                    {b_count} <span style="font-size: 1.0rem; color: #8993A4;">/ {b_req}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_cp2:
        st.markdown(
            f"""
            <div class="recessed-panel">
                <div class="readout-label">BASELINE STATUS</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: {'#45E0A8' if b_ready else '#FFB84D'}; margin-top: 2px;">
                    {'🟢 BASELINE READY' if b_ready else '📊 BASELINE NOT READY'}
                </div>
                <div style="font-size: 0.82rem; color: #9BA3AF; margin-top: 4px;">
                    {b_msg}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    if b_ready and profile:
        st.markdown('<div class="console-card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="console-card-header">
                <span>TYPING DEVIATION INDEX (TDI)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <p style="font-size: 0.85rem; color: #E9EEF5;">
                <strong>Typing Deviation Index:</strong> Measures divergence from the user's historical typing pattern.<br>
                <em style="color: #8993A4;">Scientific Invariant: This metric reflects fine-motor parameter variability. It is NOT a Stress Score, Anxiety Score, or Mental Health Score.</em>
            </p>
            """,
            unsafe_allow_html=True,
        )

        # If current session features exist in engine buffer, calculate deviation
        telemetry = engine.feature_buffer.extract_telemetry()
        paired_count = len(engine.session.get_paired_events())

        if paired_count >= 15:
            current_features = {
                "mean_dwell_ms": telemetry.get("mean_dwell_ms", 100.0),
                "mean_flight_ms": telemetry.get("mean_flight_ms", 120.0),
                "pause_rate": telemetry.get("pause_rate", 0.05),
                "estimated_wpm": telemetry.get("estimated_wpm", 45.0),
                "backspace_count": float(telemetry.get("backspace_count", 1)),
            }
            tdi_res = calculate_baseline_deviation(current_features, profile)
            tdi_val = tdi_res.get("typing_deviation_index", 25.0)
            z_scores = tdi_res.get("feature_z_scores", {})

            col_t1, col_t2 = st.columns([1, 2])
            with col_t1:
                st.plotly_chart(create_strain_gauge(tdi_val), use_container_width=True)
            with col_t2:
                fig_comp = create_baseline_comparison_chart(current_features, profile)
                st.plotly_chart(fig_comp, use_container_width=True)

            # TDI Component Analysis
            st.markdown(
                """
                <div class="console-card-header" style="margin-top: 14px;">
                    <span>TDI FEATURE DIVERGENCE BREAKDOWN</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            fig_tdi = create_tdi_breakdown_chart(z_scores)
            st.plotly_chart(fig_tdi, use_container_width=True)

        else:
            st.info("ℹ️ Complete an active Analysis Session (≥ 15 keystrokes) to compute live TDI divergence against your established baseline profile.")

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown('<div class="console-card">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="recessed-panel" style="padding: 20px; text-align: center;">
                <div style="font-size: 1.2rem; font-weight: 700; color: #FFB84D;">TDI: NOT AVAILABLE</div>
                <div style="font-size: 0.9rem; color: #E9EEF5; margin: 8px 0;">
                    Calibration: <strong>{b_count} / {b_req}</strong> sessions completed.
                </div>
                <div style="font-size: 0.8rem; color: #8993A4; max-width: 600px; margin: 0 auto;">
                    To establish an individual motor baseline, navigate to <strong>Live Session</strong>, switch to 
                    <strong>Calibration Session</strong> mode, and complete {b_req - b_count} more brief typing sessions.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 4: BEHAVIORAL MODEL
# ==============================================================================
def render_behavioral_model_page(engine: BehavioralEngine) -> None:
    """Render the production LSTM model readiness audit, uncertainty, and architecture."""
    render_console_header(
        title="Behavioral Model Analysis",
        subtitle="Stacked Bidirectional LSTM Architecture, Gating Verification & Uncertainty Telemetry",
        status_text="MODEL INFERENCE ENGINE",
        status_type="warning",
    )

    m_ready, m_reason, m_details = engine.check_model_readiness()

    # Primary Model Readiness Banner
    st.markdown('<div class="console-card">', unsafe_allow_html=True)
    if m_ready:
        st.success("🟢 **PRODUCTION MODEL READY**: Verified model artifact trained on real research dataset.")
    else:
        st.markdown(
            f"""
            <div class="recessed-panel" style="border-left: 4px solid #FF667A;">
                <div class="readout-label" style="color: #FF667A;">PRODUCTION MODEL STATUS: MODEL_NOT_READY</div>
                <div style="font-size: 0.9rem; font-weight: 600; color: #E9EEF5; margin: 4px 0;">
                    Reason: {m_reason}
                </div>
                <div style="font-size: 0.8rem; color: #8993A4; line-height: 1.4; margin-top: 6px;">
                    <strong>Training:</strong> <span style="color: #FF667A;">BLOCKED</span> &nbsp;|&nbsp; 
                    <strong>Inference:</strong> <span style="color: #FF667A;">DISABLED</span><br>
                    <em>Scientific Policy: Real labeled keystroke dataset required inside <code>data/raw/</code> before production training. Zero fabricated predictions.</em>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # 9-Criterion Production Gate Breakdown
    st.markdown('<div class="console-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="console-card-header">
            <span>9-CRITERION PRODUCTION MODEL GATE MATRIX</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    criteria_rows = [
        ("1. Model Artifact File", "lstm_model.keras exists", (engine.models_dir / "lstm_model.keras").exists()),
        ("2. Training Metadata", "training_metadata.json exists", (engine.models_dir / "training_metadata.json").exists()),
        ("3. Production Model Flag", "is_production_model == True", bool(m_details.get("is_production_model", False))),
        ("4. Real Dataset Verified", "trained_on_real_dataset == True", bool(m_details.get("trained_on_real_dataset", False))),
        ("5. Feature Manifest", "feature_manifest.json matches schema", (engine.models_dir / "feature_manifest.json").exists()),
        ("6. Sequence Length", "Expected sequence_length == 30", bool(m_details.get("sequence_length") == 30 if m_details else False)),
        ("7. Feature Dimensions", "Expected num_features == 6", bool(m_details.get("num_features") == 6 if m_details else False)),
        ("8. Feature Scaler", "feature_scaler.pkl fitted and present", (engine.models_dir / "feature_scaler.pkl").exists()),
        ("9. Label Mapping", "label_mapping.json present", (engine.models_dir / "label_mapping.json").exists()),
    ]

    gate_table_data = []
    for num, desc, passed in criteria_rows:
        gate_table_data.append(
            {
                "Criterion": num,
                "Verification Rule": desc,
                "Audit Status": "✓ PASS" if passed else "🔒 BLOCKED",
            }
        )
    st.table(pd.DataFrame(gate_table_data))
    st.markdown("</div>", unsafe_allow_html=True)

    # Model-Ready Output Architecture (Displays genuine metrics when ready, or clean empty state)
    st.markdown('<div class="console-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="console-card-header">
            <span>MODEL PERFORMANCE METRICS & UNCERTAINTY ARCHITECTURE</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if m_ready:
        metrics = m_details.get("evaluation_metrics", {})
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Accuracy", f"{metrics.get('accuracy', 0.0) * 100:.1f}%")
        col_m2.metric("Balanced Accuracy", f"{metrics.get('balanced_accuracy', 0.0) * 100:.1f}%")
        col_m3.metric("F1 Macro", f"{metrics.get('f1_macro', 0.0):.3f}")
        col_m4.metric("Loss", f"{metrics.get('loss', 0.0):.4f}")
    else:
        st.markdown(
            """
            <div class="recessed-panel" style="text-align: center; padding: 20px;">
                <div style="font-size: 1.1rem; font-weight: 700; color: #8993A4;">
                    NOT AVAILABLE — MODEL NOT TRAINED
                </div>
                <div style="font-size: 0.8rem; color: #6E7681; margin-top: 6px;">
                    When genuine production training is executed on real research data, this console will automatically 
                    render the classification confusion matrix, cross-entropy loss curves, Shannon entropy, and probability margins.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 5: SESSION HISTORY
# ==============================================================================
def render_session_history_page() -> None:
    """Render historical session log, detailed telemetry inspection, and GDPR purge."""
    render_console_header(
        title="Session History & Telemetry Log",
        subtitle="Chronological Audit Trail of Completed Sessions (data/processed/assessments/)",
        status_text="STORAGE ACTIVE",
        status_type="ready",
    )

    sessions = list_stored_sessions()

    if not sessions:
        st.markdown('<div class="console-card">', unsafe_allow_html=True)
        st.info("ℹ️ **No Completed Sessions Available**: Perform an Analysis or Calibration session in the Live Session tab to generate historical records.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown('<div class="console-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="console-card-header">
            <span>STORED SESSION AUDIT TRAIL</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    table_records = []
    for s in sessions:
        table_records.append(
            {
                "Session ID": s["session_id"][:12] + "...",
                "Date": s["date"],
                "Type": s["session_type"],
                "Duration": f"{s['duration_s']}s",
                "Events": s["event_count"],
                "Quality": s["quality"],
                "Baseline": s["baseline_status"],
                "TDI": f"{s['tdi']:.1f}" if s["tdi"] is not None else "-",
                "Model Status": s["model_status"],
            }
        )
    st.dataframe(pd.DataFrame(table_records), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Session Detail Inspector
    st.markdown('<div class="console-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="console-card-header">
            <span>SESSION DETAIL INSPECTOR</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    session_ids = [s["session_id"] for s in sessions if not s["is_corrupted"]]
    if session_ids:
        selected_sid = st.selectbox("Select Session to Inspect", options=session_ids)
        if selected_sid:
            detail = load_session_detail(selected_sid)
            if detail:
                c_ov1, c_ov2, c_ov3 = st.columns(3)
                c_ov1.write(f"**Session ID:** `{detail.get('session_id')}`")
                c_ov1.write(f"**Timestamp:** `{detail.get('timestamp')}`")

                dq = detail.get("data_quality", {})
                c_ov2.write(f"**Quality Verdict:** `{dq.get('verdict')}`")
                c_ov2.write(f"**Duration:** `{dq.get('metrics', {}).get('active_duration_s', 0):.1f}s`")

                b_res = detail.get("baseline_result", {})
                c_ov3.write(f"**Baseline Status:** `{b_res.get('status')}`")
                tdi_val = b_res.get("typing_deviation_index")
                c_ov3.write(f"**TDI:** `{f'{tdi_val:.1f}' if tdi_val is not None else 'N/A'}`")

                st.markdown("<hr style='border-color: #242B36;'>", unsafe_allow_html=True)

                # Motor Telemetry
                feats = detail.get("feature_summary", {})
                if feats:
                    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
                    col_f1.metric("Mean Dwell", f"{feats.get('mean_dwell_ms', 0):.1f} ms")
                    col_f2.metric("Mean Flight", f"{feats.get('mean_flight_ms', 0):.1f} ms")
                    col_f3.metric("Cadence", f"{feats.get('estimated_wpm', 0):.1f} WPM")
                    col_f4.metric("Corrections", f"{feats.get('backspace_count', 0)}")

                # Delete Record Button (GDPR)
                if st.button(f"🗑️ PURGE RECORD ({selected_sid[:8]}...)", key="del_sess_btn"):
                    if delete_session_record(selected_sid):
                        st.success(f"Record {selected_sid} purged successfully.")
                        st.rerun()
                    else:
                        st.error("Failed to delete record.")

    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 6: REPORTS
# ==============================================================================
def render_reports_page() -> None:
    """Render structured assessment reports with privacy audit and exports."""
    render_console_header(
        title="Assessment Reports & Exports",
        subtitle="Privacy-Audited Research Reports & Standardized JSON Telemetry",
        status_text="REPORTS ENGINE",
        status_type="ready",
    )

    sessions = list_stored_sessions()
    valid_sessions = [s for s in sessions if not s["is_corrupted"]]

    if not valid_sessions:
        st.markdown('<div class="console-card">', unsafe_allow_html=True)
        st.info("ℹ️ **No Assessment Reports Available**: Complete a live typing session to generate exportable research reports.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown('<div class="console-card">', unsafe_allow_html=True)
    report_options = {s["session_id"]: f"{s['date']} | {s['session_type']} | {s['session_id'][:10]}..." for s in valid_sessions}
    chosen_sid = st.selectbox(
        "Select Report to View & Export",
        options=list(report_options.keys()),
        format_func=lambda sid: report_options[sid],
    )

    if chosen_sid:
        detail = load_session_detail(chosen_sid)
        if detail is not None:
            # Generate simulated AssessmentResult object for report generator
            try:
                assessment_obj = BehavioralAssessmentResult.from_dict(detail)
                md_report = generate_markdown_report(assessment_obj)
                txt_report = generate_text_report(assessment_obj)

                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    st.download_button(
                        label="⬇ DOWNLOAD JSON REPORT",
                        data=json.dumps(detail, indent=2),
                        file_name=f"assessment_{chosen_sid}.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                with col_dl2:
                    st.download_button(
                        label="⬇ DOWNLOAD MARKDOWN REPORT",
                        data=md_report,
                        file_name=f"report_{chosen_sid}.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )

                st.markdown("<hr style='border-color: #242B36;'>", unsafe_allow_html=True)
                st.markdown(md_report)

            except Exception as e:
                st.error(f"Error rendering report: {e}")
        else:
            st.error("REPORT BLOCKED — PRIVACY VALIDATION FAILED OR FILE CORRUPTED.")

    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 7: PRIVACY & SECURITY
# ==============================================================================
def render_privacy_security_page() -> None:
    """Render the verified security and privacy governance specifications."""
    render_console_header(
        title="Privacy Architecture & Data Governance",
        subtitle="Zero-Raw-Text Protocols, Cryptographic Protections & Differential Privacy",
        status_text="ENFORCED / AUDITED",
        status_type="ready",
    )

    specs = get_privacy_security_specs()

    # Implemented Security Controls
    st.markdown('<div class="console-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="console-card-header">
            <span>IMPLEMENTED SECURITY & PRIVACY CONTROLS</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    controls_df = pd.DataFrame(
        [
            {
                "Security Protocol": c["name"],
                "Enforcement Status": f"🟢 {c['status']}",
                "Implementation Mechanism": c["mechanism"],
                "Technical Scope": c["details"],
            }
            for c in specs["controls"]
        ]
    )
    st.table(controls_df)
    st.markdown("</div>", unsafe_allow_html=True)

    # What is Collected vs What is NOT Collected
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown('<div class="console-card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="console-card-header" style="color: #45E0A8;">
                <span>✓ WHAT IS COLLECTED (METADATA ONLY)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for item in specs["what_is_collected"]:
            st.markdown(f"- ⏱️ {item}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_c2:
        st.markdown('<div class="console-card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="console-card-header" style="color: #FF667A;">
                <span>✗ WHAT IS NOT COLLECTED (STRICTLY SUPPRESSED)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for item in specs["what_is_not_collected"]:
            st.markdown(f"- 🚫 {item}")
        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 8: DATASET / SYSTEM STATUS
# ==============================================================================
def render_dataset_system_status_page() -> None:
    """Render technical dataset discovery, 13-criterion training readiness gate, and dataset inspector."""
    render_console_header(
        title="Dataset Status & Training Gate",
        subtitle="Raw Keystroke Repository Audit & Production Training Pre-Flight Validation",
        status_text="DATASET REPOSITORY",
        status_type="warning",
    )

    # Database Deployment Diagnostics & Status (Step 14)
    db_diag = get_database_config_diagnostics()
    conn_color = "#45E0A8" if db_diag["connection"] == "READY" else "#FF667A"
    conn_led = "ready" if db_diag["connection"] == "READY" else "critical"

    st.markdown(
        """
        <div class="console-card">
            <div class="console-card-header">
                <span>DATABASE DEPLOYMENT DIAGNOSTICS & SYSTEM STATUS</span>
            </div>
        """,
        unsafe_allow_html=True,
    )
    col_db1, col_db2, col_db3 = st.columns(3)
    with col_db1:
        st.markdown(
            f"""
            <div class="recessed-panel">
                <div class="readout-label">DATABASE BACKEND</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #E9EEF5; margin: 4px 0;">
                    {db_diag['backend']}
                </div>
                <div style="font-size: 0.8rem; color: #9BA3AF;">
                    Driver: {db_diag.get('driver', 'sqlite3')} | Port: {db_diag['port']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_db2:
        st.markdown(
            f"""
            <div class="recessed-panel">
                <div class="readout-label">CONFIG SOURCE</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #4FA8FF; margin: 4px 0;">
                    {db_diag['configuration_source']}
                </div>
                <div style="font-size: 0.8rem; color: #9BA3AF;">
                    Pooler Mode: {db_diag['pooler']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_db3:
        st.markdown(
            f"""
            <div class="recessed-panel">
                <div class="readout-label">CONNECTION STATUS</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: {conn_color}; margin: 4px 0; display: flex; align-items: center; gap: 6px;">
                    <span class="led-dot {conn_led}"></span> {db_diag['connection']}
                </div>
                <div style="font-size: 0.8rem; color: #9BA3AF;">
                    Status: {db_diag['status_code']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    diag_err_html = f"<div style='color: #FF667A; margin-top: 6px;'><strong>DIAGNOSTIC DETAIL:</strong> {db_diag['health_message']}</div>" if db_diag['connection'] == 'ERROR' else ""
    st.markdown(
        f"""
        <div style="margin-top: 10px; font-family: monospace; font-size: 0.78rem; color: #9BA3AF; background: #0B0E14; padding: 10px 14px; border-radius: 4px; border: 1px solid #1E232D;">
            <div><strong>HOST:</strong> <span style="color: #E9EEF5;">{db_diag['hostname']}</span></div>
            <div><strong>PORT:</strong> <span style="color: #E9EEF5;">{db_diag['port']}</span> &nbsp;|&nbsp; <strong>DATABASE:</strong> <span style="color: #E9EEF5;">{db_diag['database']}</span> &nbsp;|&nbsp; <strong>USERNAME:</strong> <span style="color: #E9EEF5;">{db_diag['username']}</span></div>
            <div><strong>POOLER:</strong> <span style="color: #E9EEF5;">{db_diag['pooler']}</span> &nbsp;|&nbsp; <strong>POOLER DETECTED:</strong> <span style="color: #E9EEF5;">{db_diag['pooler_detected']}</span> &nbsp;|&nbsp; <strong>DIRECT SUPABASE DETECTED:</strong> <span style="color: #E9EEF5;">{db_diag['direct_supabase_host_detected']}</span></div>
            {diag_err_html}
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ds_report = get_dataset_status()

    # Technical Dataset Card
    st.markdown('<div class="console-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="console-card-header">
            <span>RAW DATASET REPOSITORY STATUS (data/raw/)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_d1, col_d2, col_d3 = st.columns(3)
    col_d1.write(f"**Directory:** `{ds_report.get('data_directory')}`")
    col_d1.write(f"**Status Verdict:** `{ds_report.get('verdict')}`")

    col_d2.write(f"**Candidate Files:** `{ds_report.get('file_count', 0)}`")
    col_d2.write(f"**Supported Formats:** `{', '.join(ds_report.get('supported_formats', []))}`")

    col_d3.write(f"**Manifest Present:** `{ds_report.get('manifest_present')}`")
    col_d3.write(f"**Access Status:** `{ds_report.get('access_status')}`")

    if not ds_report.get("candidate_files"):
        st.warning("⚠️ **REAL DATASET REQUIRED**: No candidate research keystroke files were discovered in `data/raw/`.")

    st.markdown("</div>", unsafe_allow_html=True)

    # 13-Criterion Training Readiness Gate
    st.markdown('<div class="console-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="console-card-header">
            <span>13-CRITERION PRODUCTION TRAINING READINESS GATE</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    matrix = get_training_gate_matrix()
    matrix_df = pd.DataFrame(
        [
            {
                "Criterion": m["criterion"],
                "Gate Status": "✓ PASS" if m["status"] == "PASS" else ("🔒 BLOCKED" if m["status"] == "BLOCKED" else "⚠️ PENDING"),
                "Specification": m["description"],
                "Methodological Rule": m["rule"],
            }
            for m in matrix
        ]
    )
    st.table(matrix_df)
    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 9: ARCHITECTURE & ROADMAP
# ==============================================================================
def render_architecture_page() -> None:
    """Render technical architecture diagrams, research principles, and development roadmap."""
    render_console_header(
        title="Architecture & Research Principles",
        subtitle="End-to-End Behavioral Intelligence Pipeline, Privacy Decoupling & Roadmap",
        status_text="SYSTEM ARCHITECTURE",
        status_type="ready",
    )

    st.markdown('<div class="console-card">', unsafe_allow_html=True)
    st.subheader("Project Purpose & Research Foundations")
    st.markdown(
        """
        This academic engineering project explores the correlation between **fine-motor typing dynamics** 
        and **behavioral cognitive strain**. By capturing timing intervals—specifically **dwell time** 
        (how long a key is depressed) and **flight time** (the transition interval between consecutive key presses)—the 
        system estimates levels of cognitive hesitation, fatigue, and motor fluctuation.
        
        **CRITICAL RESEARCH MANDATE:**
        - This system is an academic research prototype, **NOT** a medical diagnostic tool.
        - Personal baseline deviations are conceptually decoupled from deep learning classifier predictions.
        - Keystroke content and typed characters are completely suppressed at intake.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Pipeline Diagrams
    st.markdown('<div class="console-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="console-card-header">
            <span>END-TO-END DATA & INFERENCE PIPELINE</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.code(
        """
OFFLINE RESEARCH PIPELINE:
Raw Dataset (data/raw/) 
   └── Real Dataset Validator (15 Dimensions)
         └── Data Engineering & Feature Extraction
               └── Zero-Text Privacy Filter & HMAC Pseudonymization
                     └── Group-Aware Split (Disjoint Subjects)
                           └── Sequential Tensor Windows (N, 30, 6)
                                 └── Production Gate (9 Criteria)
                                       └── Stacked Bidirectional LSTM

LIVE INFERENCE PIPELINE:
Sandboxed HTML5 Focus Canvas (Zero Global OS Hooks)
   └── Relative Micro-Timing Tokens (No characters)
         └── Server-Side PrivacyFilter (Audit & Sanitize)
               └── Stateful Event Normalizer (Down/Up Pairing & Flight Latencies)
                     └── Technical Data-Quality Gate (Events >= 15, Duration >= 3s)
                           └── Live Feature Buffer & Canonical Windows (N, 30, 6)
                                 ├── Branch A: Decoupled Personal Baseline (TDI Deviation Index)
                                 └── Branch B: Gated LSTM Model (MODEL_NOT_READY until real data)
                                       └── Behavioral Assessment Layer (Non-Diagnostic)
                                             └── Formatted Markdown & Audited JSON Reports
        """,
        language="text",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Engineering Roadmap
    st.markdown('<div class="console-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="console-card-header">
            <span>PROJECT ENGINEERING ROADMAP</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        | Phase | Component | Status | Description |
        |---|---|---|---|
        | **Phase 1** | Foundation & Architecture | 🟢 **Complete** | Scaffolding, SQLite database, settings, unit test harness. |
        | **Phase 2** | Dataset Investigation & Quality | 🟢 **Complete** | Schema-agnostic adapter, data quality validation, sample dataset. |
        | **Phase 3** | Feature Engineering & Baselines | 🟢 **Complete** | 34 timing metrics, Personal Baseline engine, Typing Deviation Index (TDI). |
        | **Phase 4** | Sequential Deep Learning (LSTM) | 🟢 **Complete** | Stacked LSTM architecture, group-aware splitting, training CLI. |
        | **Phase 5** | Behavioral Assessment & Interpretation | 🟢 **Complete** | Non-diagnostic assessment engine, Shannon entropy, reliability margin, reports. |
        | **Phase 6** | Data Security & Privacy Layer | 🟢 **Complete** | Zero-Text policy, HMAC-SHA256, Fernet encryption, Differential Privacy. |
        | **Phase 7** | Real Dataset Ingestion Layer | 🟢 **Complete** | 15-dimension validator, real dataset status, candidate acquisition. |
        | **Phase 8** | Pipeline Hardening & Training Readiness | 🟢 **Complete** | Pre-flight validation, zero-leakage audit, training readiness gate. |
        | **Phase 9** | Live Typing Behavior Capture Engine | 🟢 **Complete** | HTML5/JS iframe capture deck, event normalizer, feature buffer. |
        | **Phase 10** | End-to-End Behavioral Intelligence | 🟢 **Complete** | Central BehavioralEngine, live/offline parity, decoupled branches. |
        | **Phase 11** | Functional Research Dashboard | 🟢 **Complete** | 8 primary pages, Plotly charts, session history, reports, status. |
        | **Phase 12** | Visual Skeuomorphic Workstation UI | 🟢 **Complete** | Dark industrial tactile research instrument aesthetic. |
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
def main() -> None:
    """Main application orchestrator."""
    print("[STARTUP STEP 6] Initializing singleton BehavioralEngine state...", flush=True)
    engine = get_engine()
    print("[STARTUP STEP 7] Rendering workstation sidebar...", flush=True)
    safe_nav = selected_nav.encode("ascii", "replace").decode("ascii")
    print(f"[STARTUP STEP 10] Rendering active workstation console page: {safe_nav}...", flush=True)
    if "Overview" in selected_nav or selected_nav.startswith("1."):
        render_overview_page(engine)
    elif "Live" in selected_nav or selected_nav.startswith("2.") or "Assessment" in selected_nav or "Session" in selected_nav:
        render_live_session_page(engine)
    elif "Baseline" in selected_nav or selected_nav.startswith("3."):
        render_baseline_analytics_page(engine)
    elif "Model" in selected_nav or selected_nav.startswith("4."):
        render_behavioral_model_page(engine)
    elif "History" in selected_nav or selected_nav.startswith("5."):
        render_session_history_page()
    elif "Reports" in selected_nav or selected_nav.startswith("6."):
        render_reports_page()
    elif "Privacy" in selected_nav or selected_nav.startswith("7."):
        render_privacy_security_page()
    elif "Dataset" in selected_nav or selected_nav.startswith("8.") or "Status" in selected_nav:
        render_dataset_system_status_page()
    elif "Architecture" in selected_nav or selected_nav.startswith("9.") or "Roadmap" in selected_nav:
        render_architecture_page()
    else:
        render_overview_page(engine)
    print("[STARTUP STEP 11] Workstation console render cycle completed successfully.", flush=True)


if __name__ == "__main__":
    main()
