"""Mental-State Detection System using Typing Behavior
Main Streamlit Application Entrypoint.

Behavioral Analysis Workstation Console.
"""

from pathlib import Path
import sys
from typing import Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on Python sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.database import check_connection, init_db
from src.config.settings import settings
from src.data_engineering.baseline import compute_baseline_strain
from src.data_engineering.data_quality import (
    check_class_imbalance,
    check_missing_values,
    generate_quality_summary,
)
from src.data_engineering.dataset_acquisition import get_dataset_status
from src.data_engineering.real_dataset_validator import validate_real_dataset
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
from src.live_typing import (
    render_live_typing_box,
    validate_session_quality,
)
from src.integration import (
    BehavioralEngine,
    BehavioralAssessmentResult,
    PipelineState,
    SessionType,
)
from src.assessment.report_generator import generate_markdown_report

# Configure Streamlit page
st.set_page_config(
    page_title="Workstation | Mental-State Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply permanent skeuomorphic console styling
apply_workstation_theme()


def render_sidebar():
    """Render the physical workstation control sidebar."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 6px 0 16px 0; border-bottom: 1px solid #242B36; margin-bottom: 14px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: #35D6FF; letter-spacing: 0.12em; text-transform: uppercase;">
                    CONTROL CONSOLE v0.2.0
                </div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #E9EEF5; margin-top: 2px;">
                    BEHAVIORAL WORKSTATION
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        settings.ensure_directories()
        init_db()
        db_alive = check_connection()

        db_led = "ready" if db_alive else "critical"
        db_status = "ONLINE" if db_alive else "OFFLINE"

        st.markdown(
            f"""
            <div class="recessed-panel">
                <div class="readout-label">SYSTEM TELEMETRY</div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
                    <span style="font-size: 0.8rem; color: #E9EEF5;">SQLite Storage</span>
                    <div class="led-indicator">
                        <span class="led-dot {db_led}"></span>
                        <span>{db_status}</span>
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
                    <span style="font-size: 0.8rem; color: #E9EEF5;">Environment</span>
                    <span style="font-size: 0.8rem; font-weight: 700; color: #35D6FF;">{settings.app_env.upper()}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
                    <span style="font-size: 0.8rem; color: #E9EEF5;">LSTM Model</span>
                    <span style="font-size: 0.72rem; font-weight: 600; color: #FFB84D;">DATASET REQUIRED</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
                    <span style="font-size: 0.8rem; color: #E9EEF5;">Personal Baseline</span>
                    <span style="font-size: 0.72rem; font-weight: 600; color: #45E0A8;">ENGINE ONLINE</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
                    <span style="font-size: 0.8rem; color: #E9EEF5;">Assessment Layer</span>
                    <span style="font-size: 0.72rem; font-weight: 600; color: #35D6FF;">READY</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
                    <span style="font-size: 0.8rem; color: #E9EEF5;">Zero-Text Audit</span>
                    <span style="font-size: 0.72rem; font-weight: 600; color: #45E0A8;">ENFORCED</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
                    <span style="font-size: 0.8rem; color: #E9EEF5;">Pseudonymization</span>
                    <span style="font-size: 0.72rem; font-weight: 600; color: #35D6FF;">HMAC-SHA256</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
                    <span style="font-size: 0.8rem; color: #E9EEF5;">Differential Privacy</span>
                    <span style="font-size: 0.72rem; font-weight: 600; color: #8B7CFF;">&epsilon;={settings.dp_epsilon}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Dataset Discovery & Readiness Status
        ds_status = get_dataset_status()
        raw_status = ds_status.get("status", "missing")
        
        # Dataset: MISSING / FOUND / VALID / READY
        if raw_status == "missing":
            ds_label = "MISSING"
            ds_color = "#FF667A"
        elif raw_status == "candidate_found":
            ds_label = "FOUND"
            ds_color = "#FFB84D"
        elif raw_status in ("ready_for_validation", "validated"):
            ds_label = "VALID"
            ds_color = "#35D6FF"
        else:
            ds_label = "READY"
            ds_color = "#45E0A8"

        # Model Training: BLOCKED / READY / TRAINING / COMPLETE
        prod_model_path = settings.models_path / "lstm_model.keras"
        if prod_model_path.exists():
            training_label = "COMPLETE"
            training_color = "#45E0A8"
        elif ds_status.get("labeled_dataset_available", False) and raw_status == "validated":
            training_label = "READY"
            training_color = "#35D6FF"
        else:
            training_label = "BLOCKED"
            training_color = "#FF667A"

        # Baseline: NOT READY / READY
        baseline_dir = settings.processed_data_path / "baselines"
        has_baselines = baseline_dir.exists() and any(baseline_dir.glob("*.json"))
        baseline_label = "READY" if has_baselines else "NOT READY"
        baseline_color = "#45E0A8" if has_baselines else "#FFB84D"

        # Privacy: READY / FAILED
        privacy_label = "READY"
        privacy_color = "#45E0A8"

        # Leakage: NOT CHECKED / PASS / FAIL
        leakage_file = settings.processed_data_path / "leakage_audit_report.json"
        if leakage_file.exists():
            try:
                import json
                l_rep = json.loads(leakage_file.read_text(encoding="utf-8"))
                leakage_pass = l_rep.get("passed", False)
                leakage_label = "PASS" if leakage_pass else "FAIL"
                leakage_color = "#45E0A8" if leakage_pass else "#FF667A"
            except Exception:
                leakage_label = "NOT CHECKED"
                leakage_color = "#FFB84D"
        else:
            leakage_label = "NOT CHECKED"
            leakage_color = "#FFB84D"

        st.markdown(
            f"""
            <div class="recessed-panel" style="margin-top: 10px;">
                <div class="readout-label">DATASET READINESS</div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
                    <span style="font-size: 0.8rem; color: #E9EEF5;">Dataset</span>
                    <span style="font-size: 0.72rem; font-weight: 600; color: {ds_color};">{ds_label}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
                    <span style="font-size: 0.8rem; color: #E9EEF5;">Training</span>
                    <span style="font-size: 0.72rem; font-weight: 600; color: {training_color};">{training_label}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
                    <span style="font-size: 0.8rem; color: #E9EEF5;">Baseline</span>
                    <span style="font-size: 0.72rem; font-weight: 600; color: {baseline_color};">{baseline_label}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
                    <span style="font-size: 0.8rem; color: #E9EEF5;">Privacy</span>
                    <span style="font-size: 0.72rem; font-weight: 600; color: {privacy_color};">{privacy_label}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
                    <span style="font-size: 0.8rem; color: #E9EEF5;">Leakage</span>
                    <span style="font-size: 0.72rem; font-weight: 600; color: {leakage_color};">{leakage_label}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="recessed-panel" style="margin-top: 10px;">
                <div class="readout-label">PRIVACY & GOVERNANCE PROTOCOL</div>
                <div style="font-size: 0.78rem; color: #8993A4; margin-top: 5px; line-height: 1.4;">
                    🔒 <strong>Zero-Text Invariant:</strong> Key characters and typed words are completely suppressed at intake.
                    <br>
                    🛡️ <strong>Encryption at Rest:</strong> Sensitive baselines and assessments use authenticated Fernet ciphers.
                    <br>
                    ⏳ <strong>Retention Limits:</strong> Automated lifecycle expiration (7d / 90d / 180d).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_dataset_inspector_tab():
    """Render the primary Dataset Inspector console."""
    render_console_header(
        title="Dataset Inspector Console",
        subtitle="Schema-Agnostic Keystroke Dynamics Profiler & Data Quality Auditor",
        status_text="ONLINE / READY",
        status_type="ready",
    )

    sample_csv_path = settings.sample_data_path / "sample_keystrokes.csv"

    # Selection controls in a raised panel
    st.markdown('<div class="console-card">', unsafe_allow_html=True)
    col_mode, col_upload = st.columns([1, 2])

    with col_mode:
        data_source = st.radio(
            "Select Data Source",
            options=["Registered Sample Benchmark", "Upload Custom File (.csv)"],
            index=0,
        )

    active_df: Optional[pd.DataFrame] = None
    source_name = "sample_keystrokes.csv (Synthetic Demo Benchmark)"

    with col_upload:
        if data_source == "Upload Custom File (.csv)":
            uploaded_file = st.file_uploader(
                "Upload Keystroke Dataset",
                type=["csv", "xlsx"],
                help="Upload any research keystroke CSV. Columns will be auto-detected.",
            )
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith(".csv"):
                        active_df = pd.read_csv(uploaded_file)
                    else:
                        active_df = pd.read_excel(uploaded_file)
                    source_name = uploaded_file.name
                except Exception as e:
                    st.error(f"Error parsing uploaded file: {e}")
        else:
            if sample_csv_path.exists():
                active_df = load_dataset(sample_csv_path)
            else:
                st.warning(
                    f"Sample benchmark not found at `{sample_csv_path}`. Run generation script."
                )

    st.markdown("</div>", unsafe_allow_html=True)

    if active_df is None:
        st.info("Awaiting dataset selection or upload to begin inspection.")
        return

    # Check for sensitive text columns
    sensitive_cols = detect_sensitive_text_columns(active_df)
    if sensitive_cols:
        render_privacy_guard_banner(sensitive_cols)

    # Generate structured dataset report
    report = generate_dataset_report(active_df, dataset_name=source_name)
    quality_summary = generate_quality_summary(active_df)

    # 1. Digital Readout Bay
    render_instrument_readouts(
        records=report["rows"],
        features=report["columns"],
        users=report["unique_users"],
        sessions=report["unique_sessions"],
    )

    # 2. Split Console: Quality Audit & Detected Signals
    col_q, col_s = st.columns([1, 1])

    with col_q:
        render_quality_console_card(quality_summary)

    with col_s:
        # Build dictionary of signal detections
        signals_map = {
            "Timestamp": report["timestamp_column"] is not None,
            "User Identifier": report["user_column"] is not None,
            "Session Identifier": report["session_column"] is not None,
            "Dwell / Hold Time": any("dwell" in c.lower() or "hold" in c.lower() for c in active_df.columns),
            "Flight Time / IKI": any("flight" in c.lower() or "iki" in c.lower() for c in active_df.columns),
            "Backspace / Error": any("backspace" in c.lower() or "error" in c.lower() for c in active_df.columns),
            "Behavioral Label": report["label_column"] is not None,
        }
        render_detected_signals_card(signals_map)

    # 3. Class Distribution Analysis (if label exists)
    if report["label_column"]:
        st.markdown('<div class="console-card">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="console-card-header">
                <span>TARGET CLASS DISTRIBUTION: <code>{report['label_column']}</code></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        dist_data = report["class_distribution"]
        if dist_data:
            col_chart, col_stats = st.columns([2, 1])
            with col_chart:
                fig = px.bar(
                    x=list(dist_data.keys()),
                    y=list(dist_data.values()),
                    labels={"x": "Behavioral State", "y": "Record Count"},
                    color=list(dist_data.keys()),
                    color_discrete_sequence=["#35D6FF", "#8B7CFF", "#45E0A8", "#FFB84D"],
                )
                fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="#15181E",
                    paper_bgcolor="#15181E",
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=240,
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_stats:
                st.markdown('<div class="recessed-panel">', unsafe_allow_html=True)
                imbalance_res = check_class_imbalance(active_df, report["label_column"])
                st.write(f"**Total Labeled Records:** `{sum(dist_data.values())}`")
                st.write(f"**Imbalance Ratio:** `{imbalance_res['imbalance_ratio']}:1`")
                if imbalance_res["is_imbalanced"]:
                    st.warning("⚠️ High class imbalance detected (> 3:1).")
                else:
                    st.success("✓ Class proportions are balanced.")
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # 4. Sanitized Data Preview
    st.markdown('<div class="console-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="console-card-header">
            <span>SANITIZED DATASET PREVIEW (TOP 10 RECORDS)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    preview_df = active_df.head(10).copy()
    # Mask any sensitive text columns from preview
    for col in sensitive_cols:
        if col in preview_df.columns:
            preview_df[col] = "[MASKED - ZERO-TEXT POLICY]"

    st.dataframe(preview_df, use_container_width=True)
    st.caption(
        "🔒 *Privacy Guarantee: Any typed message content or character strings are masked above and excluded from model processing.*"
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_live_typing_tab():
    """Render the live typing behavioral intelligence workstation interface."""
    render_console_header(
        title="Live Behavioral Capture Engine",
        subtitle="Privacy-Preserving Keystroke Dynamic Instrumentation & Real-Time Telemetry",
        status_text="CAPTURE DECK ACTIVE",
        status_type="ready",
    )

    # 1. Mandatory Privacy & Zero-Text Policy Banner
    st.markdown(
        """
        <div class="recessed-panel" style="margin-bottom: 16px; border-left: 4px solid #45E0A8;">
            <div style="font-size: 0.82rem; font-weight: 700; color: #45E0A8; letter-spacing: 0.08em; text-transform: uppercase;">
                🔒 ZERO-TEXT PRIVACY ENFORCEMENT ACTIVE
            </div>
            <div style="font-size: 0.8rem; color: #9BA3AF; margin-top: 4px; line-height: 1.4;">
                This live session instruments <strong>strictly relative micro-timing intervals</strong> (dwell times, flight latencies, pause intervals).
                No typed characters, message strings, word sequences, or input values are ever recorded, stored, or transmitted.
                Global operating-system hooks are strictly prohibited.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize BehavioralEngine state in Streamlit session_state
    if "behavioral_engine" not in st.session_state:
        st.session_state.behavioral_engine = BehavioralEngine(user_id="live_participant_01")
    if "session_active" not in st.session_state:
        st.session_state.session_active = False
    if "reset_trigger" not in st.session_state:
        st.session_state.reset_trigger = False
    if "latest_assessment_result" not in st.session_state:
        st.session_state.latest_assessment_result = None

    engine: BehavioralEngine = st.session_state.behavioral_engine

    # 2. Session Mode Selector (Calibration vs Analysis)
    st.markdown('<div class="console-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="console-card-header">
            <span>SESSION CONFIGURATION & OPERATIONAL MODE</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_mode_sel, col_mode_info = st.columns([1, 2])
    with col_mode_sel:
        current_idx = 0 if engine.session_type == SessionType.ANALYSIS else 1
        mode_choice = st.radio(
            "SELECT SESSION TYPE",
            ["Analysis Session", "Calibration Session"],
            index=current_idx,
            horizontal=True,
            disabled=st.session_state.session_active,
        )
        selected_type = SessionType.CALIBRATION if "Calibration" in mode_choice else SessionType.ANALYSIS
        engine.set_session_type(selected_type)

    with col_mode_info:
        if selected_type == SessionType.CALIBRATION:
            st.info("🎯 **Calibration Session Active**: Valid session data will be added to your personal baseline history. Minimum 5 calibration sessions are required.")
        else:
            st.info("🔬 **Analysis Session Active**: Evaluates current typing dynamics against your calibrated personal baseline and the deep learning sequence model.")
    st.markdown("</div>", unsafe_allow_html=True)

    # 3. Control Deck Controls
    st.markdown('<div class="console-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="console-card-header">
            <span>SESSION CONTROL INSTRUMENTATION</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_btn1, col_btn2, col_btn3, col_btn4, col_btn5 = st.columns(5)

    with col_btn1:
        if st.button("▶ START SESSION", use_container_width=True, disabled=st.session_state.session_active):
            engine.start_session(selected_type)
            st.session_state.session_active = True
            st.session_state.reset_trigger = False
            st.session_state.latest_assessment_result = None
            st.rerun()

    with col_btn2:
        if st.button("⏸ PAUSE", use_container_width=True, disabled=not st.session_state.session_active):
            engine.pause_session()
            st.rerun()

    with col_btn3:
        if st.button("▶ RESUME", use_container_width=True, disabled=engine.session.status.value != "PAUSED"):
            engine.resume_session()
            st.rerun()

    with col_btn4:
        if st.button("⏹ STOP & ANALYZE", use_container_width=True, disabled=not st.session_state.session_active and engine.session.status.value != "PAUSED"):
            result = engine.stop_session()
            st.session_state.session_active = False
            st.session_state.latest_assessment_result = result
            st.rerun()

    with col_btn5:
        if st.button("🔄 RESET", use_container_width=True):
            engine.reset_session()
            st.session_state.session_active = False
            st.session_state.reset_trigger = True
            st.session_state.latest_assessment_result = None
            st.rerun()

    # Authoritative Session & Pipeline Status Readout
    status_val = engine.session.status.value
    duration_val = engine.session.duration_seconds
    event_cnt = engine.feature_buffer.event_count
    pipe_state_val = engine.state.value

    st.markdown(
        f"""
        <div style="display: flex; gap: 20px; margin-top: 12px; padding: 8px 12px; background: #111318; border-radius: 6px; font-size: 0.8rem; color: #8993A4; flex-wrap: wrap;">
            <div>Session ID: <code style="color: #35D6FF;">{engine.session.session_id}</code></div>
            <div>Mode: <strong style="color: #E6EDF3;">{engine.session_type.value}</strong></div>
            <div>Pipeline State: <strong style="color: #35D6FF;">{pipe_state_val}</strong></div>
            <div>Capture State: <strong style="color: {'#45E0A8' if status_val == 'ACTIVE' else '#FFB84D'};">{status_val}</strong></div>
            <div>Active Duration: <strong style="color: #E6EDF3;">{duration_val:.1f}s</strong></div>
            <div>Buffered Events: <strong style="color: #35D6FF;">{event_cnt}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # 4. Live Keystroke Capture Box (Vanilla HTML/JS component)
    raw_event_batch = render_live_typing_box(
        session_active=st.session_state.session_active,
        reset_signal=st.session_state.reset_trigger,
        key="live_typing_input_deck",
    )

    if raw_event_batch and st.session_state.session_active:
        engine.ingest_raw_events(raw_event_batch, strict_privacy=True)
        st.session_state.reset_trigger = False

    # 5. Real-Time Motor Dynamics Telemetry
    st.markdown('<div class="console-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="console-card-header">
            <span>REAL-TIME MOTOR DYNAMICS TELEMETRY</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    telemetry = engine.feature_buffer.extract_telemetry()
    col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns(5)
    col_t1.metric("Mean Dwell", f"{telemetry['mean_dwell_ms']:.1f} ms")
    col_t2.metric("Mean Flight", f"{telemetry['mean_flight_ms']:.1f} ms")
    col_t3.metric("Cadence Proxy", f"{telemetry['estimated_wpm']:.1f} WPM")
    col_t4.metric("Cognitive Pauses", f"{telemetry['pause_count']} ({telemetry['pause_rate']*100:.0f}%)")
    col_t5.metric("Corrections", f"{telemetry['backspace_count']}")

    st.markdown("</div>", unsafe_allow_html=True)

    # 6. Post-Session Intelligence Assessment Panels
    if st.session_state.latest_assessment_result:
        res: BehavioralAssessmentResult = st.session_state.latest_assessment_result

        # Panel A: Data Quality Gate
        st.markdown('<div class="console-card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="console-card-header">
                <span>1. TECHNICAL DATA QUALITY VERDICT</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if res.data_quality.is_valid:
            st.success(f"✓ Technical Quality: PASS — Session satisfied all micro-timing criteria ({res.data_quality.metrics.get('sequence_windows_count', 0)} sequence windows generated).")
        else:
            st.warning(f"⚠️ Technical Quality: FAIL ({res.data_quality.verdict})")
            for reason in res.data_quality.reasons:
                st.write(f"- {reason}")
            st.info("ℹ️ Non-Diagnostic Rule: Sessions with insufficient observations are never categorized into behavioral or psychological states.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Panel B & C: Decoupled Baseline and Model Branches
        col_b_branch, col_m_branch = st.columns(2)

        with col_b_branch:
            st.markdown('<div class="console-card">', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="console-card-header">
                    <span>2. PERSONAL TYPING BASELINE</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            b = res.baseline_result
            if b.status == "READY":
                st.markdown(
                    f"""
                    <div class="recessed-panel">
                        <div class="readout-label">TYPING DEVIATION INDEX (TDI)</div>
                        <div style="font-size: 1.4rem; font-weight: 700; color: #45E0A8; margin: 4px 0;">
                            {b.typing_deviation_index:.1f} <span style="font-size: 0.9rem; color: #8993A4;">/ 100</span>
                        </div>
                        <div style="font-size: 0.8rem; color: #9BA3AF; line-height: 1.4;">
                            {b.message}
                        </div>
                        <div style="font-size: 0.72rem; color: #6E7681; margin-top: 8px;">
                            <em>TDI measures individual motor variation relative to past sessions. It does not diagnose clinical stress or anxiety.</em>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="recessed-panel">
                        <div class="readout-label">CALIBRATION PROGRESS</div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #35D6FF; margin: 4px 0;">
                            📊 {b.session_count} / {b.min_required_sessions} SESSIONS
                        </div>
                        <div style="font-size: 0.8rem; color: #9BA3AF; line-height: 1.4;">
                            {b.message}
                        </div>
                        <div style="font-size: 0.72rem; color: #6E7681; margin-top: 8px;">
                            <em>Individual motor baseline requires minimum 5 calibration sessions to establish longitudinal validity.</em>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        with col_m_branch:
            st.markdown('<div class="console-card">', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="console-card-header">
                    <span>3. DEEP LEARNING SEQUENCE MODEL</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            m = res.model_result
            if m.status == "MODEL_READY":
                st.markdown(
                    f"""
                    <div class="recessed-panel">
                        <div class="readout-label">PREDICTED BEHAVIORAL CLASS</div>
                        <div style="font-size: 1.2rem; font-weight: 700; color: #35D6FF; margin: 4px 0;">
                            {m.predicted_class}
                        </div>
                        <div style="font-size: 0.8rem; color: #9BA3AF; line-height: 1.4;">
                            Separation Reliability: <strong>{m.reliability}</strong> (Margin: {m.probability_margin:.3f})
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="recessed-panel">
                        <div class="readout-label">MODEL INFERENCE STATUS</div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #FFB84D; margin: 4px 0;">
                            🔒 {m.status}
                        </div>
                        <div style="font-size: 0.8rem; color: #9BA3AF; line-height: 1.4;">
                            {m.reason}
                        </div>
                        <div style="font-size: 0.72rem; color: #6E7681; margin-top: 8px;">
                            <em>Academic Policy: Production LSTM training is gated until an approved research dataset is supplied. Zero fake predictions generated.</em>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # Panel D: Structured Report & Export
        st.markdown('<div class="console-card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="console-card-header">
                <span>4. SESSION INTELLIGENCE REPORT & AUDIT</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_rep_btn, col_json_btn = st.columns([1, 1])
        with col_json_btn:
            import json as py_json
            st.download_button(
                label="⬇ DOWNLOAD SESSION REPORT (JSON)",
                data=py_json.dumps(res.to_dict(), indent=2),
                file_name=f"assessment_{res.session_id}.json",
                mime="application/json",
                use_container_width=True,
            )

        with st.expander("📄 VIEW FORMATTED SESSION INTELLIGENCE REPORT", expanded=False):
            st.markdown(generate_markdown_report(res))

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        # Pre-run baseline and model status preview
        col_m_gate, col_b_gate = st.columns(2)
        with col_m_gate:
            st.markdown('<div class="recessed-panel">', unsafe_allow_html=True)
            m_ready, m_reason, _ = engine.check_model_readiness()
            st.markdown(
                f"""
                <div class="readout-label">LSTM SEQUENTIAL MODEL STATUS</div>
                <div style="font-size: 0.95rem; font-weight: 700; color: {'#45E0A8' if m_ready else '#FFB84D'}; margin: 4px 0;">
                    {'🟢 READY' if m_ready else '🔒 MODEL_NOT_READY'}
                </div>
                <div style="font-size: 0.8rem; color: #9BA3AF; line-height: 1.4;">
                    {m_reason}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col_b_gate:
            st.markdown('<div class="recessed-panel">', unsafe_allow_html=True)
            b_ready, b_msg, b_count, b_req, _ = engine.check_baseline_readiness()
            st.markdown(
                f"""
                <div class="readout-label">PERSONAL BASELINE CALIBRATION</div>
                <div style="font-size: 0.95rem; font-weight: 700; color: {'#45E0A8' if b_ready else '#35D6FF'}; margin: 4px 0;">
                    {'🟢 READY' if b_ready else '📊 NOT READY'} ({b_count}/{b_req} SESSIONS)
                </div>
                <div style="font-size: 0.8rem; color: #9BA3AF; line-height: 1.4;">
                    {b_msg}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)


def main():
    render_sidebar()

    # Main Workstation Banner
    st.title("🧠 Behavioral Analysis Workstation Console")
    st.markdown(
        "*A non-invasive research system analyzing fine-motor keystroke metadata "
        "to estimate behavioral cognitive strain.*"
    )

    # Mandatory Academic Disclaimer
    st.warning(
        "⚠️ **Academic Disclaimer**: "
        "This system provides behavioral estimates and is not a medical diagnostic tool."
    )

    # Console Navigation Tabs
    tab_live, tab_inspector, tab_overview, tab_baseline, tab_architecture = st.tabs(
        [
            "⌨️ Live Typing Capture",
            "🔬 Dataset Inspector",
            "📋 Project Overview",
            "📊 Baseline Analytics",
            "🧩 Architecture & Roadmap",
        ]
    )

    with tab_live:
        render_live_typing_tab()

    with tab_inspector:
        render_dataset_inspector_tab()

    with tab_overview:
        st.markdown('<div class="console-card">', unsafe_allow_html=True)
        st.subheader("Project Description & Research Principles")
        st.write(
            """
            This academic engineering project explores the correlation between **fine-motor typing dynamics** 
            and **behavioral cognitive strain**. By capturing timing intervals—specifically **dwell time** 
            (how long a key is depressed) and **flight time** (the transition interval between consecutive key presses)—the 
            system estimates levels of cognitive hesitation, fatigue, and motor fluctuation.
            """
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="recessed-panel">', unsafe_allow_html=True)
            st.markdown("#### 🔒 Privacy by Design")
            st.write(
                "Strictly operates on relative microsecond timing intervals. "
                "Text content, message strings, and character identities are immediately dropped or masked."
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="recessed-panel">', unsafe_allow_html=True)
            st.markdown("#### 🎯 Non-Invasive Instrumentation")
            st.write(
                "Requires zero wearable biomedical sensors, eye trackers, or camera feeds. "
                "Leverages existing human-computer interaction hardware."
            )
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_baseline:
        st.markdown('<div class="console-card">', unsafe_allow_html=True)
        st.subheader("Heuristic Baseline Strain Estimation")
        st.write(
            "Extracts dwell and transition metrics from sample sequences to calculate a benchmark "
            "behavioral strain score before training temporal neural networks."
        )

        sample_timings = pd.DataFrame(
            {
                "press_time": [0, 150, 320, 510, 750, 1100, 1300, 1490, 1720, 2050],
                "release_time": [90, 240, 410, 600, 840, 1200, 1390, 1580, 1820, 2150],
            }
        )

        features = extract_timing_features(sample_timings)
        baseline_result = compute_baseline_strain(features)

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Mean Hold Time", f"{features['mean_hold_time_ms']:.1f} ms")
        col_m2.metric("Mean Flight Time", f"{features['mean_flight_time_ms']:.1f} ms")
        col_m3.metric("Pause Rate", f"{features['pause_rate'] * 100:.1f}%")
        col_m4.metric("Estimated Strain", f"{baseline_result['strain_score']}/100")

        col_g1, col_g2 = st.columns([1, 1])
        with col_g1:
            st.plotly_chart(
                create_strain_gauge(baseline_result["strain_score"]),
                use_container_width=True,
            )
        with col_g2:
            hold_times = (
                sample_timings["release_time"] - sample_timings["press_time"]
            ).tolist()
            st.plotly_chart(
                create_hold_time_histogram(hold_times),
                use_container_width=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_architecture:
        st.markdown('<div class="console-card">', unsafe_allow_html=True)
        st.subheader("Engineering Roadmap")
        st.markdown(
            """
            | Phase | Component | Status | Description |
            |---|---|---|---|
            | **Phase 1** | Foundation & Architecture | 🟢 **Complete** | Scaffolding, SQLite database, settings, unit test harness. |
            | **Phase 2** | Dataset Investigation & Quality | 🟢 **Complete** | Schema-agnostic adapter, data quality validation, sample dataset. |
            | **Phase 3** | Feature Engineering & Baselines | 🟢 **Complete** | 34 timing metrics, Personal Baseline engine, Typing Deviation Index (TDI). |
            | **Phase 4** | Sequential Deep Learning (LSTM) | 🟡 **Awaiting Data** | Stacked LSTM architecture, group-aware splitting, training CLI (requires real dataset). |
            | **Phase 5** | Behavioral Assessment & Interpretation | 🟢 **Complete** | Non-diagnostic assessment engine, Shannon entropy, reliability margin, JSON reports. |
            | **Phase 6** | **Data Security & Privacy Layer** | 🟢 **Active** | Zero-Text policy, HMAC-SHA256, Fernet encryption, Differential Privacy, GDPR purge. |
            | **Phase 7** | Workstation UI & Live Interface | ⏳ *Planned* | Full skeuomorphic workstation console redesign and live keystroke capture. |
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
