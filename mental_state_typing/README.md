# Mental-State Detection System using Typing Behavior

> **Academic Final-Year Project**  
> **Important Disclaimer:** *This system provides behavioral estimates and is not a medical diagnostic tool.*

---

## 📌 Project Objective

The **Mental-State Detection System using Typing Behavior** is an academic research project that explores the relationship between fine-motor keystroke dynamics and cognitive strain/mental fatigue. 

Traditional mental-state and stress detection systems often depend on intrusive hardware (e.g., heart rate monitors, EEG headsets, eye trackers) or visual surveillance (e.g., facial emotion recognition via webcams). This project investigates a non-invasive, privacy-preserving alternative: analyzing micro-timing patterns in keystroke rhythms—specifically **hold duration (dwell time)**, **flight time (inter-key latency)**, and **hesitation pauses**.

---

## 🏛️ Four Major Components

The system is organized into four decoupled, modular subsystems:

```
                  ┌─────────────────────────────────────┐
                  │ 1. Data Engineering Subsystem       │
                  │ - Timing ingestion & validation     │
                  │ - Physiological outlier filtering   │
                  │ - Feature extraction & baselines    │
                  └──────────────────┬──────────────────┘
                                     │
                  ┌──────────────────▼──────────────────┐
                  │ 2. Data Security & Privacy Subsystem│
                  │ - Salted SHA-256 pseudonymization   │
                  │ - Strict Zero-Text enforcement      │
                  │ - Keystroke character suppression   │
                  └──────────────────┬──────────────────┘
                                     │
                  ┌──────────────────▼──────────────────┐
                  │ 3. Deep Learning Subsystem (Phase 4)│
                  │ - Sliding-window sequence encoding  │
                  │ - Temporal pattern modeling (LSTM)  │
                  │ - Behavioral strain inference       │
                  └──────────────────┬──────────────────┘
                                     │
                  ┌──────────────────▼──────────────────┐
                  │ 4. Data Visualization Subsystem     │
                  │ - Interactive Plotly analytics      │
                  │ - Streamlit behavioral dashboard    │
                  │ - Real-time rhythm metrics & gauges │
                  └─────────────────────────────────────┘
```

### 1. Data Engineering (`src/data_engineering/`)
- Ingests raw keystroke timing logs (press timestamps and release timestamps).
- Filters noise and physiological anomalies (e.g., negative intervals or implausibly extended key holds).
- Extracts statistical behavioral features: mean dwell time, variance of flight time, typing cadence irregularity, and pause frequency.
- Generates an interpretable, rule-based baseline strain score prior to deep neural modeling.

### 2. Data Security & Privacy (`src/privacy/`)
- **Zero-Text Policy**: No actual character, word, or sentence content is ever logged, retained, or transmitted. The system only processes numerical microsecond timing deltas.
- **Pseudonymization**: Participant identifiers are irreversibly hashed using salted SHA-256 (`usr_<hash>`).
- **Audit Compliance**: Includes automated checks to reject datasets containing sensitive key labels or character columns.

### 3. Deep Learning (`src/deep_learning/`)
- Formulates keystroke time series into sliding-window sequences suitable for recurrent architectures.
- *(Scheduled for Phase 4)*: Long Short-Term Memory (LSTM) network architecture to model temporal cognitive fluctuations over time.
- Inference pipeline mapping sequence representations to behavioral strain indicators.

### 4. Data Visualization (`src/visualization/`)
- Interactive Plotly visualization suite for distribution histograms, flight time timelines, and strain gauges.
- Responsive Streamlit dashboard for monitoring system health, reviewing data engineering metrics, and tracking session histories.

---

## 📊 Dataset Strategy

### 1. Why a Labeled Keystroke Dataset is Required
Supervised training of temporal models (such as recurrent networks and LSTMs) requires empirical ground-truth pairing between continuous fine-motor micro-timings (dwell times, flight times, pause rates) and objective or validated psychological/behavioral states. Without labeled benchmarks, model representations cannot be evaluated for predictive fidelity.

### 2. Meaningful Behavioral & Psychological Labels
Useful target categories for cognitive typing dynamics include:
- **Cognitive Workload / Task Demand**: High vs. low mental effort or multi-tasking conditions.
- **Mental Fatigue / Prolonged Depletion**: Alert typing cadence vs. degraded post-fatigue motor rhythm.
- **Stress / Temporal Pressure**: Fluent execution vs. pressured motor hesitation.
- **Affective Valence / Frustration**: Fluent keystroke sequences vs. erratic burst/backspace patterns.

### 3. Schema-Agnostic Adapter Design
Academic keystroke dynamics corpora (e.g. from research publications and open science repositories) exhibit diverse schemas: some supply raw keydown/keyup timestamps, others record precomputed inter-key intervals (`IKI`), and identifiers vary (`subject`, `participant`, `user_id`). 

Rather than enforcing a brittle schema, our `DatasetAdapter` dynamically infers variable roles, detects behavioral targets, and isolates sensitive textual fields automatically.

### 4. Selection Criteria for the Final Real-World Dataset
The final dataset will be chosen based on:
- Millisecond-precision timing records for hold and transition intervals.
- Well-documented experimental protocols establishing behavioral/psychological state labels.
- Cohort diversity across sessions to allow participant-split cross-validation.
- Strict ethical compliance ensuring participants consented to keystroke metadata logging.
*Note: Datasets are only recognized as scientifically validated if accompanied by peer-reviewed experimental methodology.*

### 5. Role of Synthetic Sample Data
The dataset located at `data/sample/sample_keystrokes.csv` is generated purely for engineering validation, schema testing, and UI rendering. **Synthetic data does not represent genuine psychological behavior** and will never be used for formal model training or behavioral conclusions.

---

## 🔄 Planned Workflow

The system is developed through a five-phase incremental roadmap:

1. **Phase 1: Foundation & Architecture (Complete)**
   - Project directory scaffolding, environment management, SQLite connection layer, foundational module interfaces, and unit testing suite.
2. **Phase 2: Dataset Investigation & Data Engineering (Current Phase)**
   - Schema-agnostic dataset adapter, data quality validation framework, dataset registry, synthetic sample dataset, and skeuomorphic workstation inspector.
3. **Phase 3: Privacy & Security Enforcement**
   - Salted pseudonymization, character suppression audit tests, and secure local metadata handling.
4. **Phase 4: Sequential Deep Learning Pipeline**
   - Sequence windowing, model definition, training workflows, loss evaluation, and saved model checkpoints.
5. **Phase 5: Live Behavioral Estimation & Interactive UI**
   - Live timing event buffer, session management, and integrated end-user dashboard.

---

## 🚀 Current Development Phase

- **Current Status**: **Phase 2 (Dataset Investigation & Data Engineering Layer)**
- **Completed in this Phase**:
  - `DatasetAdapter` with heuristic auto-detection for users, sessions, timestamps, micro-timings, and labels.
  - `DatasetRegistry` providing decoupled dataset configurations.
  - `DataQuality` module evaluating missing values, duplicate rows, negative durations, timestamp monotonicity, and class balance.
  - 250-record realistic synthetic sample dataset (`data/sample/sample_keystrokes.csv`).
  - Skeuomorphic "behavioral workstation" UI console and interactive Dataset Inspector in Streamlit.
  - Zero-text privacy filter masking sensitive text fields during data ingestion and inspection.
  - Comprehensive unit tests covering adapters, registry, and quality metrics (32 total passing tests).

---

## 📂 Project Structure

```text
mental_state_typing/
│
├── app.py                      # Main Streamlit dashboard application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .env.example                # Example environment variables template
├── .gitignore                  # Git exclusions for secrets, caches, and DBs
│
├── data/
│   ├── raw/                    # Raw timing records (gitignored)
│   ├── processed/              # Extracted and sanitized feature sets
│   └── sample/                 # Mock sample timing data for unit tests
│
├── models/                     # Saved neural model checkpoints and scalers
│
├── database/
│   ├── __init__.py
│   └── database.py             # SQLite database connection & schema manager
│
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py         # Configuration loader from environment
│   ├── data_engineering/
│   │   ├── __init__.py
│   │   ├── loader.py           # Timing data ingestion
│   │   ├── cleaner.py          # Outlier filtering
│   │   ├── feature_engineering.py # Hold/flight time feature extraction
│   │   └── baseline.py         # Heuristic baseline strain calculation
│   ├── deep_learning/
│   │   ├── __init__.py
│   │   ├── preprocessing.py    # Sequence preparation
│   │   ├── model.py            # Deep learning model stub
│   │   ├── train.py            # Training pipeline stub
│   │   └── predict.py          # Prediction engine stub
│   ├── privacy/
│   │   ├── __init__.py
│   │   ├── pseudonymization.py # Salted participant ID hashing
│   │   ├── encryption.py       # Cryptographic helper stubs
│   │   └── privacy_utils.py    # Zero-text sanitization filters
│   ├── live_typing/
│   │   ├── __init__.py
│   │   └── collector.py        # Keystroke buffer interface
│   └── visualization/
│       ├── __init__.py
│       ├── charts.py           # Plotly charts (gauges, histograms, timelines)
│       └── dashboard.py        # Streamlit presentation helpers
│
└── tests/
    ├── __init__.py
    ├── test_data_engineering.py # Data engineering unit tests
    ├── test_baseline.py         # Baseline strain metric tests
    ├── test_privacy.py          # Privacy & pseudonymization tests
    └── test_config_and_db.py    # Settings and SQLite connectivity tests
```

---

## ⚙️ Installation & Quick Start

### 1. Prerequisites
- Python 3.11+
- Virtual environment (recommended: `venv` or `uv`)

### 2. Setup Virtual Environment & Install Dependencies
```bash
# Navigate to the project directory
cd mental_state_typing

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
```bash
# Copy the example environment template
copy .env.example .env
```

### 4. Run the Test Suite
```bash
pytest tests -v
```

### 5. Launch the Streamlit Application
```bash
streamlit run app.py
```

---

## 🛡️ Ethical & Privacy Statement

This project adheres strictly to privacy-by-design guidelines:
- **No Keylogging**: The system does not intercept or record key names or textual content.
- **No Medical Claim**: The system evaluates motor timing fluctuations as a proxy for cognitive strain and strictly does **not** provide clinical diagnosis or medical treatment advice.
