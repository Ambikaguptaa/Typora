# Functional Research Dashboard Architecture

**Project**: Mental-State Detection System using Typing Behavior  
**Phase**: Prompt 12 — Complete Functional Research Dashboard & Analytics Layer  
**Document**: `docs/dashboard_architecture.md`  

---

## 1. Overview & Architectural Principles

The Functional Research Workstation Dashboard is an academic, privacy-preserving analytical console designed for behavioral research into fine-motor typing dynamics. It serves as the primary user interface for:
- Live keystroke data collection in a sandboxed, zero-keylogging environment.
- Micro-timing motor telemetry extraction and visualization.
- Longitudinal personal baseline calibration ($0/5 \to 5/5$).
- Divergence assessment via the Typing Deviation Index (TDI).
- Gated production LSTM model verification and uncertainty diagnostics.
- Session history audit trails and privacy-audited report downloads.

### Core Invariants Enforced Across the Entire Dashboard:
1. **Strict Zero-Raw-Text Policy**: Typed characters, letters, words, sentences, and passwords are immediately stripped in the browser client and checked by server-side `PrivacyFilter`. No raw text ever enters Streamlit session state, database records, charts, or JSON reports.
2. **Conceptual Decoupling**: Model classification and personal baseline deviations remain strictly isolated pathways and are never combined into a composite "stress" or "mental health" score.
3. **Truthful Readiness & Anti-Fabrication**: Production LSTM training and inference remain gated (`MODEL_NOT_READY`) until an approved real research dataset is placed into `data/raw/`. Missing models never display fabricated accuracy or mock probabilities.
4. **Non-Diagnostic Research Disclaimer**: All pages prominently feature academic disclaimers stating that the system measures motor dynamic variation, not clinical or psychiatric disorders.

---

## 2. Navigation Structure & Page Responsibilities

The primary navigation is housed in the workstation sidebar, providing seamless routing across 9 dedicated analytical sections:

```
┌───────────────────────────────────────────────────────────┐
│                      PRIMARY NAVIGATION                   │
├───────────────────────────────────────────────────────────┤
│  1. 📋 Overview                                           │
│  2. ⌨️ Live Session                                       │
│  3. 📊 Baseline Analytics                                 │
│  4. 🧠 Behavioral Model                                   │
│  5. 📜 Session History                                    │
│  6. 📑 Reports                                            │
│  7. 🔒 Privacy & Security                                 │
│  8. 🔬 Dataset / System Status                            │
│  9. 🧩 Architecture & Roadmap                             │
└───────────────────────────────────────────────────────────┘
```

### Page Responsibilities

| Page | Primary Purpose | Key Components |
|---|---|---|
| **1. Overview** | Executive system summary, project principles, and readiness audit | Project description, research badges, non-diagnostic disclaimer, 6-category system status grid (Dataset, Model, Baseline, Live Capture, Privacy, Leakage), research workflow guide. |
| **2. Live Session** | Real-time keystroke acquisition, live telemetry, and interactive waveforms | Dual-mode protocol switch (Analysis vs Calibration), physical control deck (Start, Pause, Resume, Stop, Reset), sandboxed HTML5 typing canvas, 8 live motor metrics, 4 Plotly dynamic charts (Timing rhythm, Pause distribution, Cadence, Corrections), session quality audit card. |
| **3. Baseline Analytics** | Individual motor baseline profiles, longitudinal calibration, and TDI | Calibration progress tracker ($0/5 \to 5/5$), TDI strain gauge, comparative bar chart (Personal Baseline vs Current Session), feature divergence breakdown chart ($\pm\sigma$). |
| **4. Behavioral Model** | Deep learning model verification, 9-criterion production gate, and uncertainty | Model readiness status banner, 9-criterion gate audit table, model-ready performance architecture (accuracy, confusion matrix, Shannon entropy, probability margin). |
| **5. Session History** | Chronological audit trail and historical session inspector | Table of stored sessions from `data/processed/assessments/`, detail inspector, motor telemetry summary, baseline & model status, GDPR-compliant record purge button. |
| **6. Reports** | Structured academic assessment reports and data exports | Report selector, Markdown formatted view, audited JSON download, text export, privacy violation blocking guard. |
| **7. Privacy & Security** | Data governance specifications, cryptographic controls, and disclosures | Implemented controls table (7 verified protocols), "What is Collected" vs "What is NOT Collected", differential privacy ($\epsilon=0.5$), encryption specs. |
| **8. Dataset / System Status** | Technical raw dataset repository audit and training readiness | Dataset discovery report (`data/raw/`), 13-criterion training readiness pre-flight matrix, integrated Dataset Inspector profiler tool. |
| **9. Architecture & Roadmap** | Technical system design, inference pipelines, and engineering roadmap | End-to-end pipeline diagrams (offline vs live), research principles, 12-phase engineering milestone roadmap. |

---

## 3. Backend Dependencies & Data Flow

Business logic is strictly decoupled from the UI presentation layer (`app.py`), residing in modular backend services under `src/`:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        BACKEND-FIRST ARCHITECTURE                      │
└────────────────────────────────────────────────────────────────────────┘

[ User Keystrokes ]
       │
       ▼
[ src.live_typing.event_capture ] ─── Focused HTML5/JS iframe canvas
       │
       ▼
[ src.live_typing.privacy_filter ] ── Enforces Zero-Text policy; rejects raw characters
       │
       ▼
[ src.live_typing.event_normalizer ] ─ Pairs down/up events; computes flight times
       │
       ▼
[ src.live_typing.validation ] ────── Enforces duration (≥3s) & events (≥15)
       │
       ▼
[ src.live_typing.feature_buffer ] ── Computes telemetry & generates (N, 30, 6) windows
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
[ src.data_engineering.baseline ]     [ src.integration.behavioral_engine ]
  - Calculates TDI                      - 9-criterion production model gate
  - Manages calibration history         - Evaluates model uncertainty
       │                                         │
       └────────────────────┬────────────────────┘
                            ▼
           [ src.integration.session_history ]
             - data/processed/assessments/
             - JSON serialization & privacy audits
                            │
                            ▼
           [ src.assessment.report_generator ]
             - Generates Markdown & plain-text
                            │
                            ▼
           [ src.visualization.charts ]
             - Generates Plotly timing charts
                            │
                            ▼
           [ app.py (Streamlit Workstation) ]
```

---

## 4. Visualization Sources & Privacy Safeguards

All charts are generated via Plotly using pure numerical vectors. No raw event dictionary or character token is permitted to enter any visualization pipeline.

### Implemented Visualizations (`src/visualization/charts.py`)

1. **Typing Timing Timeline (`create_timing_timeline_chart`)**:
   - **X-Axis**: Keystroke transition index ($0, 1, 2, \dots$).
   - **Y-Axis**: Duration in milliseconds.
   - **Traces**: Dwell time (key depression duration) and Flight time (inter-key latency).
   - **Analytical Purpose**: Identifies motor hesitation patterns, rhythm regularity, and fatigue slowing.
2. **Pause Timeline (`create_pause_timeline_chart`)**:
   - **X-Axis**: Keystroke event sequence index.
   - **Y-Axis**: Pause duration in milliseconds.
   - **Threshold Reference**: Horizontal reference line at pause threshold (default: $500\text{ ms}$).
   - **Analytical Purpose**: Visualizes temporal distribution of cognitive pauses and hesitations.
3. **Typing Cadence Chart (`create_typing_rate_chart`)**:
   - **X-Axis**: Event window index.
   - **Y-Axis**: Typing speed in Words Per Minute (WPM).
   - **Analytical Purpose**: Tracks velocity stability and deceleration trends across the session.
4. **Correction Activity Chart (`create_correction_activity_chart`)**:
   - **X-Axis**: Event index.
   - **Y-Axis (Left)**: Instantaneous backspace flag ($0$ or $1$).
   - **Y-Axis (Right)**: Cumulative correction count.
   - **Analytical Purpose**: Tracks editing frequency and rapid burst corrections.
5. **Baseline Comparison Chart (`create_baseline_comparison_chart`)**:
   - **X-Axis**: Motor dynamics metrics (Dwell, Flight, Pause Rate, Cadence, Corrections).
   - **Y-Axis**: Observed metric value.
   - **Grouping**: Personal Baseline vs Current Session.
   - **Terminology**: Labeled neutrally as *"Difference from personal baseline"*.
6. **TDI Feature Divergence Chart (`create_tdi_breakdown_chart`)**:
   - **X-Axis**: Deviation from individual baseline in standard deviations ($Z\text{-score } \sigma$).
   - **Y-Axis**: Feature name.
   - **Threshold Lines**: Reference markers at $\pm 1.0\sigma$ and $\pm 2.0\sigma$.

---

## 5. Truthful Empty States & Error Handling

Every dashboard page includes explicit, truthful empty states:

| Condition | UI Behavior | Truthful Message |
|---|---|---|
| **No Dataset in `data/raw/`** | Warning panel on Overview and Dataset pages | *"REAL DATASET REQUIRED: No candidate research keystroke files were discovered in data/raw/."* |
| **Model Untrained** | Red status badge and blocked state panel | *"MODEL NOT READY: Approved real research dataset required before production training. Zero fake predictions generated."* |
| **Baseline $< 5$ Sessions** | Amber status and calibration tracker | *"TDI: NOT AVAILABLE. Calibration: X / 5 sessions completed. Complete 5 calibration sessions to establish baseline."* |
| **No Sessions in History** | Informational alert in Session History | *"No completed sessions available. Conduct an Analysis or Calibration session in the Live Session tab."* |
| **No Reports Available** | Informational alert in Reports page | *"No assessment reports available. Complete a live typing session to generate exportable research reports."* |
| **Corrupted Assessment File** | Highlighted as CORRUPTED in table | *"Malformed JSON: skips parsing gracefully without crashing the UI."* |
| **Privacy Violation in File** | Highlighted as BLOCKED in table | *"REPORT BLOCKED — PRIVACY VALIDATION FAILED: File contains forbidden raw text fields."* |

---

## 6. Model-Ready Extensibility

Even though the production model currently reports `MODEL_NOT_READY`, the Behavioral Model page and report generator are architected with full compatibility for real training outputs:
- When a genuine model is trained using `python -m src.deep_learning.train` on a verified dataset in `data/raw/`, `training_metadata.json` will be populated.
- `engine.check_model_readiness()` will automatically evaluate all 9 criteria and transition to `MODEL_READY`.
- The dashboard will immediately display:
  - Accuracy, Balanced Accuracy, Macro F1, and Loss.
  - Multi-class probability distribution bar chart.
  - Model separation margin and Shannon entropy.
  - Validation confusion matrix and training curves.
- **Zero code modifications** to the dashboard UI will be required when real training is completed.
