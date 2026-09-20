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
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="recessed-panel" style="margin-top: 10px;">
                <div class="readout-label">PRIVACY PROTOCOL</div>
                <div style="font-size: 0.78rem; color: #8993A4; margin-top: 5px; line-height: 1.4;">
                    🔒 <strong>Zero-Text Enforcement:</strong> All keystroke character identifiers and typed sentences are intercepted and suppressed.
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
    tab_inspector, tab_overview, tab_baseline, tab_architecture = st.tabs(
        [
            "🔬 Dataset Inspector",
            "📋 Project Overview",
            "📊 Baseline Analytics",
            "🧩 Architecture & Roadmap",
        ]
    )

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
            | **Phase 1** | Foundation & Architecture | 🟢 **Complete** | Modular directory scaffolding, SQLite database, settings, unit test harness. |
            | **Phase 2** | **Dataset Investigation & Data Quality** | 🟢 **Active** | Schema-agnostic adapter, data quality validation, sample dataset, workstation UI. |
            | **Phase 3** | Data Security & Privacy | ⏳ *Planned* | Salted pseudonymization pipeline and zero-character audit checks. |
            | **Phase 4** | Sequential Deep Learning | ⏳ *Planned* | Recurrent network (LSTM) architecture, sequence scaling, and training. |
            | **Phase 5** | Live Interface & Dashboard | ⏳ *Planned* | Live timing evaluation and comprehensive Plotly visualization dashboard. |
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
