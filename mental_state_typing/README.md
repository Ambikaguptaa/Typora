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
                  │ 3. Deep Learning Subsystem (LSTM)   │
                  │ - Sliding-window sequence encoding  │
                  │ - Stacked recurrent LSTM classifier │
                  │ - Group-aware leakage prevention    │
                  └──────────────────┬──────────────────┘
                                     │
                  ┌──────────────────▼──────────────────┐
                  │ 4. Assessment & Interpretation      │
                  │ - Model uncertainty (margin, entropy│
                  │ - Personal baseline interpretation  │
                  │ - Data-quality gate & Zero-Text JSON│
                  └──────────────────┬──────────────────┘
                                     │
                  ┌──────────────────▼──────────────────┐
                  │ 5. Data Visualization Subsystem     │
                  │ - Interactive Plotly analytics      │
                  │ - Streamlit workstation dashboard   │
                  │ - Telemetry readouts & status LEDs  │
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

## 📈 Personal Typing Baseline

### 1. Why a Personal Baseline is Needed
Typing cadence is highly idiosyncratic. One user naturally types at 75 WPM with 85ms dwell times, while another fluently types at 40 WPM with 130ms dwell times. Absolute thresholds inevitably produce false positives. To evaluate behavioral fluctuations meaningfully, the system computes an individual historical baseline for each user and evaluates deviations relative to their own past habits.

### 2. Baseline Statistical Calculation
For each participant with sufficient historical observations, the baseline engine calculates both parametric and robust non-parametric metrics across numerical typing variables (WPM, dwell time, flight interval, pause rate, backspace rate, CV of flight):
- **Mean & Standard Deviation**: Captures central tendency and overall dispersion.
- **Median & Median Absolute Deviation (MAD)**: Provides outlier-resistant measures of central tendency:
  $$\text{MAD} = \text{median}(|x - \text{median}(x)|)$$
- **Interquartile Range (IQR)**: Evaluates spread across the middle 50% of sessions ($p_{75} - p_{25}$).
- **Historical Outlier Auditing**: Sessions exceeding $2.5 \times \text{MAD}$ are flagged for investigation without blind deletion.

### 3. Deviation Metrics
When evaluating a new typing session against an established personal baseline:
- **Absolute Deviation**: $\Delta = x_{\text{current}} - \mu_{\text{baseline}}$
- **Percentage Deviation**: $\%\Delta = \frac{x_{\text{current}} - \mu_{\text{baseline}}}{\mu_{\text{baseline}}} \times 100$ (with zero-division protection)
- **Standardized Deviation ($z$-score)**: $z = \frac{x_{\text{current}} - \mu_{\text{baseline}}}{\sigma_{\text{baseline}}}$ (with zero-variance fallback to MAD)
- **Direction of Change**: Categorized as `"higher"`, `"lower"`, or `"within_baseline"` based on a configurable tolerance band ($|z| \le 0.5$).

### 4. Typing Deviation Index (TDI)
The **Typing Deviation Index (TDI)** is a normalized composite score from 0.0 to 100.0 that aggregates standardized deviations across all evaluated behavioral features:
$$\text{TDI} = \min\left(100.0, \frac{1}{N} \sum_{i=1}^N \min(|z_i|, 4.0) \times 25.0\right)$$
- An average $|z| = 1.0$ yields a TDI of 25.0.
- An average $|z| = 2.0$ yields a TDI of 50.0.
- An average $|z| \ge 4.0$ reaches the ceiling of 100.0.

### 5. Strict Non-Medical Boundary
**The Typing Deviation Index is strictly a statistical indicator of motor rhythm divergence.** It does NOT measure or diagnose stress, anxiety, depression, burnout, or any clinical disorder. A high TDI simply indicates that the typing cadence differed from historical averages (e.g. faster execution, longer hesitations, or increased revisions).

### 6. Cold-Start Management & Controlled Updates
- **Cold Start**: If a user has fewer than `MIN_BASELINE_SESSIONS` (default: 5), their baseline status is marked `"insufficient_history"`. The system **never fabricates** a baseline.
- **Baseline Updates**: As new valid sessions accumulate, `update_user_baseline()` recalculates statistics across the verified session history, preventing any single anomalous session from arbitrarily corrupting baseline distributions.

---

## 🔄 Planned Workflow

The system is developed through a five-phase incremental roadmap:

1. **Phase 1: Foundation & Architecture (Complete)**
   - Project directory scaffolding, environment management, SQLite connection layer, foundational module interfaces, and unit testing suite.
2. **Phase 2: Dataset Investigation & Data Engineering (Complete)**
   - Schema-agnostic dataset adapter, data quality validation framework, dataset registry, synthetic sample dataset, and skeuomorphic workstation inspector.
3. **Phase 3: Personal Baseline & Temporal Sequence Preparation (Current Phase)**
   - Personal typing baseline engine, robust MAD/IQR statistics, Typing Deviation Index, leakage-safe grouped splitting, and 3D temporal sequence arrays.
4. **Phase 4: Sequential Deep Learning Pipeline**
   - Recurrent network (LSTM) architecture, sequence scaling, model training workflows, loss evaluation, and saved model checkpoints.
5. **Phase 5: Live Behavioral Estimation & Dedicated UI/UX Phase**
   - Live timing event buffer, end-to-end inference, and full workstation UI redesign.

---

## 🚀 Current Development Phase

- **Current Status**: **Phase 3 (Personal Baseline & Temporal Sequences Complete)**
- **Completed in this Phase**:
  - `src/data_engineering/baseline.py`: Personal baseline engine with empirical & robust stats (Mean, Median, Std, MAD, IQR).
  - Configurable `MIN_BASELINE_SESSIONS` (default: 5) and strict cold-start `"insufficient_history"` handling.
  - Standardized deviations ($z$-scores) with safe zero-variance fallback and tolerance bands.
  - Non-diagnostic **Typing Deviation Index (TDI, 0–100)**.
  - Baseline updating and persistence in `data/processed/baselines/`.
  - Feature Manifest cataloging 25+ behavioral features and documenting reasons for unavailable variables.
  - Leakage-safe grouped train/test splitting by user ID.
  - 3D temporal sequence slicing `(samples, timesteps, features)` with zero cross-user sequence bleed.
  - Comprehensive unit test suite with 65 passing tests (100% pass rate).

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
│   │   ├── baseline.py         # Personal baseline and TDI calculation
│   │   └── pipeline.py         # Feature engineering orchestrator
│   ├── deep_learning/
│   │   ├── __init__.py
│   │   ├── training_validation.py # Training readiness validation checks
│   │   ├── data_split.py       # Group-aware 3-way split & sequence scaling
│   │   ├── label_encoder.py    # Categorical label encoder & mapping
│   │   ├── model.py            # Stacked LSTM classifier & ModelConfig
│   │   ├── train.py            # Training pipeline & CLI entrypoint
│   │   ├── evaluate.py         # Metrics, confusion matrix, overfitting checks
│   │   ├── baseline_classifier.py # Non-deep-learning benchmark classifier
│   │   ├── predict.py          # Sequence inference engine
│   │   └── preprocessing.py    # Sliding-window sequence slicing
│   ├── assessment/
│   │   ├── __init__.py
│   │   ├── quality_checks.py   # Data-quality gate & explicit assessment states
│   │   ├── interpretation.py   # Prediction uncertainty, entropy & baseline ranking
│   │   └── behavioral_assessment.py # Assessment synthesis & Zero-Text JSON reports
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
│       ├── dashboard.py        # Skeuomorphic workstation console components
│       └── theme.py            # Console visual styling tokens
│
└── tests/
    ├── __init__.py
    ├── test_data_engineering.py # Data engineering unit tests
    ├── test_baseline.py         # Baseline strain metric tests
    ├── test_data_quality.py     # Data quality validation tests
    ├── test_dataset_adapter.py  # Schema-agnostic adapter tests
    ├── test_feature_engineering.py # Feature extraction tests
    ├── test_privacy.py          # Privacy & pseudonymization tests
    ├── test_config_and_db.py    # Settings and SQLite connectivity tests
    ├── test_sequence_preprocessing.py # Sliding window tests
    ├── test_training_validation.py # Training readiness validation tests
    ├── test_lstm_model.py       # LSTM architecture tests
    ├── test_data_split.py       # Grouped 3-way split & scaler tests
    ├── test_evaluation.py       # Evaluation metrics & confusion matrix tests
    ├── test_prediction.py       # Sequence inference tests
    ├── test_lstm_smoke_train.py # Technical pipeline smoke tests
    ├── test_interpretation.py   # Model uncertainty & baseline ranking tests
    ├── test_assessment_quality.py # Quality gate & state transition tests
    └── test_behavioral_assessment.py # Complete assessment pipeline integration tests
```

---

## 🧠 Deep Learning / LSTM Sequential Pipeline

The Deep Learning subsystem implements a temporal sequence classification pipeline designed to capture fine-grained behavioral changes across successive keystroke events without analyzing typed text.

### 1. Why LSTM is Used
Typing behavior is inherently temporal and non-Markovian: motor rhythms exhibit transient pauses, hesitation bursts, typing speed fluctuations, and backspace corrective clusters that span multiple consecutive keystrokes. Standard feedforward neural networks treat each event independently, losing the temporal context. Long Short-Term Memory (LSTM) recurrent networks maintain internal memory cell states ($c_t$) and hidden states ($h_t$) gated by input, forget, and output mechanisms, enabling the model to learn long-range temporal dependencies and rhythm transitions indicative of cognitive fatigue or workload.

### 2. Input Sequence Representation
Input tensors have shape:
$$\text{Input Tensor} \in \mathbb{R}^{\text{samples} \times \text{timesteps} \times \text{features}}$$
- **Samples ($N$)**: The number of extracted sliding-window sequence segments.
- **Timesteps ($T$)**: The sequence window size (default: 30 consecutive keystroke events).
- **Features ($D$)**: Numeric behavioral timing dimensions per keystroke:
  1. `dwell_time`: Key hold duration in milliseconds ($t_{\text{release}} - t_{\text{press}}$).
  2. `flight_time`: Inter-key transition interval in milliseconds ($t_{\text{press}, k} - t_{\text{release}, k-1}$).
  3. `pause_duration`: Cognitive hesitation gaps exceeding threshold (e.g. 2000ms).
  4. `typing_speed`: Keystroke rate or rolling WPM.
  5. `backspace`: Binary indicator or frequency of corrective keystrokes.
  6. `error_flag`: Flagged timing anomaly or editing burst.

### 3. What the Model Predicts
The model predicts the **behavioral state category** associated with the typing sequence, defined strictly by the labels present in the research dataset (e.g., `Calm`, `Fatigued`, `High_Workload`). The output probabilities represent model alignment with annotated experimental conditions, **not** clinical diagnoses.

### 4. Group-Aware Train / Validation / Test Splitting
To evaluate genuine generalization to unseen participants:
- Data is partitioned into three disjoint sets: **Train (70%)**, **Validation (15%)**, and **Test (15%)**.
- Partitioning uses **Group-Aware Splitting** (`GroupShuffleSplit`) grouped by `user_id`.
- If a dataset contains fewer than 3 users, grouped splitting gracefully falls back to `session_id`.

### 5. Prevention of Data Leakage
- **Zero Group Overlap**: All sequences from a given user (or session) exist exclusively in Train, Validation, or Test:
  $$\text{Train}_{\text{groups}} \cap \text{Val}_{\text{groups}} = \emptyset, \quad \text{Val}_{\text{groups}} \cap \text{Test}_{\text{groups}} = \emptyset, \quad \text{Train}_{\text{groups}} \cap \text{Test}_{\text{groups}} = \emptyset$$
- **Window Isolation**: Temporal sliding windows are constructed *within* each user session independently and never cross session or partition boundaries.
- **Untouched Test Set**: The test partition is held out completely and only evaluated once training is finalized.

### 6. Feature Scaling Protocol
- The feature scaler (`StandardScaler` or `RobustScaler`) is fitted **strictly on the training sequences**.
- Validation, test, and future live sequences are transformed using the fitted training statistics without re-estimating mean or variance:
  $$\mu_{\text{train}}, \sigma_{\text{train}} \leftarrow \text{Fit}(X_{\text{train}}), \quad X_{\text{val}}^{\text{scaled}} \leftarrow \frac{X_{\text{val}} - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
- The fitted scaler is serialized to `models/feature_scaler.pkl` along with `models/feature_manifest.json` preserving the exact feature ordering.

### 7. Class Imbalance Handling
- Class distributions are inspected dynamically from training labels.
- Class weights are calculated **exclusively on the training split** using balanced inverse frequency:
  $$w_c = \frac{N_{\text{train}}}{K \times N_c}$$
- These weights are applied during loss optimization to prevent majority-class bias.

### 8. Metrics Reported
Model evaluation avoids relying solely on raw accuracy (which is deceptive under class imbalance) and computes:
- **Balanced Accuracy**: Macro-averaged recall across all classes.
- **Macro & Weighted Precision, Recall, and F1-Score**.
- **Per-Class Breakdown**: Precision, recall, F1, and support for each specific class.
- **Confusion Matrix**: Saved in machine-readable JSON (`confusion_matrix.json`), tabular CSV (`confusion_matrix.csv`), and high-resolution styled heatmap (`confusion_matrix.png`).
- **Overfitting Diagnostics**: Objective comparison between training, validation, and test performance.

### 9. Non-Deep-Learning Baseline Comparison
To prove whether recurrent sequence modeling provides true empirical benefit over simpler approaches, a non-deep-learning baseline (Random Forest or Logistic Regression) is trained on aggregated summary statistics (mean, std, median, min, max) of the exact same sequences across the identical group-safe splits. Metrics are saved to `models/baseline_metrics.json`.

### 10. Model Artifacts Persistence
All serialized artifacts are saved in `models/`:
- `lstm_model.keras`: Keras native model checkpoint (weights, architecture, and optimizer state).
- `feature_scaler.pkl`: Training-fitted feature scaler.
- `label_mapping.json`: Bidirectional class-to-integer mapping.
- `feature_manifest.json`: Feature catalog, dimensions, and ordering.
- `training_config.json`: Hyperparameters and training configuration.
- `training_metrics.json`: Validation and test set performance report.
- `training_history.json`: Epoch-by-epoch loss and accuracy telemetry.

### 11. Command-Line Training Execution
To run the model training pipeline from the terminal:
```bash
python -m src.deep_learning.train
```
If no real dataset is found in `data/raw/`, the system enforces data protocol safety and reports:
`REAL MODEL TRAINING BLOCKED -- NO VALID LABELED DATASET AVAILABLE`

### 12. Non-Diagnostic Research Boundary
The LSTM model detects fine-motor rhythm variations mapped to research condition labels. It **does not diagnose** psychological, neurological, or psychiatric disorders. All model probability outputs are labeled as "class probabilities", not "clinical confidence".

---

## 🚦 Current Dataset & Model Training Status

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ STATUS: REAL DATASET REQUIRED — INTEGRATION PIPELINE READY                   │
│ Training Gate: ENGAGED & LOCKED (Zero Fabrication Guarantee)                │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Critical Distinction: Software Implementation vs. Empirical Model Training

| Dimension | Status | Verified Details |
| :--- | :--- | :--- |
| **Software Implementation** | **100% Complete & Hardened** | Full pipeline verified with 160+ automated unit & integration tests. Canonical event schema, BaseDatasetAdapter interface, 15-point empirical validator, 6-dimension leakage audit, zero-raw-text privacy filters, group-aware 3-way splitters, and stacked LSTM architecture are fully operational. |
| **Empirical Model Training** | **Awaiting Real Research Dataset** | In strict compliance with academic and scientific research ethics, **no fake model weights, fabricated accuracy metrics, or synthetic pseudo-predictions exist**. Production model training will only commence once an approved real research dataset is placed in `data/raw/`. |
| **Current Training Gate** | **Safely Blocked** | Executing `python -m src.deep_learning.train` stages the pre-flight checks and halts at `[1/8] Dataset discovery ........ BLOCKED (REAL DATASET REQUIRED)`. |

### Next Step for Ingestion:
To ingest an approved research dataset (e.g. MobileStress or CMU keystroke dynamics):
1. Review the detailed protocol in [docs/dataset_acquisition_checklist.md](docs/dataset_acquisition_checklist.md).
2. Place the verified dataset file in `data/raw/` (e.g. `data/raw/mobilestress.csv`).
3. Execute the staged training command:
   ```bash
   python -m src.deep_learning.train
   ```
   The engine will automatically execute all 8 pre-flight verification stages and train the production LSTM model.


---

## 🔍 Behavioral Assessment & Model Interpretation Layer

The Assessment subsystem (`src/assessment/`) synthesizes temporal model predictions, personal baseline deviations, and data-quality checks into an interpretable, transparent behavioral evaluation.

### 1. Dual Independent Signals (Zero Score Fusion)
The system strictly separates two fundamentally distinct concepts:
- **Model Prediction**: *"What population behavioral pattern does this sequence resemble based on training data?"*
- **Personal Baseline Deviation (TDI)**: *"How different is this specific session from this individual user's own historical motor rhythm?"*

These two signals are **never** mathematically merged into a single composite "mental health score" or arbitrary weighted formula. Both signals are presented side-by-side with appropriate scientific context.

### 2. Model Uncertainty & Prediction Reliability
The engine calculates empirical uncertainty indicators:
- **Top Probability ($p_{\text{top}}$)**: Maximum class probability assigned by the model.
- **Probability Margin ($\Delta p$)**: Difference between the top and second-highest class probabilities:
  $$\Delta p = p_{\text{top}} - p_{\text{second}}$$
- **Normalized Shannon Entropy ($H_{\text{norm}}$)**: Measures distribution dispersion across classes ($0.0 = \text{certainty}, 1.0 = \text{uniform confusion}$):
  $$H(p) = -\sum_{i=1}^K p_i \log_2(p_i), \quad H_{\text{norm}}(p) = \frac{H(p)}{\log_2(K)}$$
- **Reliability Categorization**:
  - `HIGH_SEPARATION`: $\Delta p \ge 0.30$ (clear class distinction).
  - `MODERATE_SEPARATION`: $0.15 \le \Delta p < 0.30$.
  - `LOW_SEPARATION`: $\Delta p < 0.15$ (competing classes are closely contested).

### 3. Personal Baseline & Feature Deviation Ranking
- Evaluates the **Typing Deviation Index (TDI, 0–100)**:
  - `expected_variance`: $\text{TDI} < 30$
  - `moderate_deviation`: $30 \le \text{TDI} < 60$
  - `higher_deviation`: $\text{TDI} \ge 60$
- Ranks individual features by absolute standardized deviation ($|z\text{-score}|$) to highlight the specific motor dimensions driving deviation (e.g. dwell time elongation, flight latency increases).

### 4. Data-Quality Gate & Explicit Operational States
Before producing an assessment, the system audits input integrity and establishes an unambiguous operational state:
- `ready`: Valid keystrokes, model and personal baseline both available.
- `prediction_available`: Model prediction ready; baseline unavailable or insufficient historical sessions.
- `baseline_available`: Personal baseline ready; deep learning model not yet loaded.
- `model_unavailable`: Neither model nor baseline ready.
- `insufficient_data`: Fewer than 5 keystrokes recorded in the session window.
- `not_ready`: Input tensor contains NaNs, infinite values, or dimensional mismatches.

### 5. Machine-Readable Zero-Text Reports
Assessments are serialized as JSON artifacts in `data/processed/assessments/assessment_YYYYMMDD_HHMMSS.json`. In strict compliance with the project's **Zero-Text Policy**, no character names, typed words, or sentence content are ever retained. Reports contain only pseudonymous participant identifiers (`usr_<hash>`), timing metrics, probabilities, and version metadata.

---

## 🔒 Data Security & Privacy Architecture

The Privacy and Security subsystem (`src/privacy/`) operationalizes privacy-by-design principles across every stage of the data lifecycle. A full specification is available in [docs/privacy_threat_model.md](docs/privacy_threat_model.md).

### 1. Strict Zero-Raw-Text Policy
- **Elimination of Character Content**: Keystroke character names (`key`, `char`, `text`, `word`, `character`, `password`) are quarantined and purged immediately at the boundary of data collection.
- **Audit Verification**: Continuous programmatic scans (`audit_zero_raw_text`) enforce that no text payloads or key labels can enter databases, feature tables, or serialized assessments.

### 2. Keyed HMAC-SHA256 Pseudonymization
- Participant identifiers (usernames, student IDs, emails) are replaced with deterministic, irreversible pseudonyms formatted as `usr_<16-hex>`.
- Hashes utilize server-side environment secrets (`PSEUDONYMIZATION_SECRET`), preventing rainbow table and dictionary re-identification attacks.

### 3. Authenticated Symmetric Encryption at Rest (Fernet)
- Sensitive artifacts—including personal baseline profiles (`data/processed/baselines/`) and behavioral assessment reports (`data/processed/assessments/`)—are encrypted using Fernet (128-bit AES in CBC mode with PKCS7 padding and HMAC-SHA256 authentication).
- Integrity verification guarantees that tampered or corrupted ciphertexts are detected immediately upon decryption, raising a `DecryptionError`.

### 4. Differential Privacy (Laplace Mechanism)
- Aggregate research queries (e.g. population average typing speeds, cohort pause frequencies) inject calibrated noise drawn from the Laplace distribution:
  $$\text{Noise} \sim \text{Laplace}\left(0, \frac{\Delta}{\epsilon}\right), \quad \text{where } \Delta = \frac{b - a}{N}$$
- Enforces bounded $L_1$ sensitivity via interval clipping $[a, b]$ and logs cumulative privacy budget consumption under basic sequential composition.

### 5. Data Retention Schedules & GDPR Article 17 Purge
- Automated data expiration policies prevent indefinite retention:
  - **Raw timing events**: 7-day retention.
  - **Session features & baselines**: 90-day retention.
  - **Behavioral assessment reports**: 180-day retention.
- **Participant Data Purge**: Supports GDPR Article 17 "Right to be Forgotten", enabling complete deletion of a participant's records across all SQLite tables and local files.

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

### 5. Execute Model Training Pipeline
```bash
python -m src.deep_learning.train
```

### 6. Launch the Streamlit Application
```bash
streamlit run app.py
```

---

## 🛡️ Ethical & Privacy Statement

This project adheres strictly to privacy-by-design guidelines:
- **No Keylogging**: The system does not intercept or record key names or textual content.
- **No Medical Claim**: The system evaluates motor timing fluctuations as a proxy for cognitive strain and strictly does **not** provide clinical diagnosis or medical treatment advice.

