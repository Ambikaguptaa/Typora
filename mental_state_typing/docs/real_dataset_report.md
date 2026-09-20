# Real Dataset Validation Report

**Project**: Mental-State Detection System using Typing Behavior  
**Document**: Real Dataset Validation & Readiness Specification  
**Report Generation Status**: Active Pipeline Ready — Awaiting Real Dataset Placement in `data/raw/`  

---

## 1. Dataset Overview

- **Dataset Name**: MobileStress / CMU Stress Keystroke Dynamics Candidate
- **Source**: Academic Research Laboratories (Data Transfer Agreement Required)
- **Current Acquisition Status**: `ACCESS REQUIRED / MISSING IN data/raw/`
- **Primary Data Directory**: `data/raw/`
- **Permitted Formats**: `.csv`, `.xlsx`, `.parquet`, `.json`, `.tsv`

---

## 2. Participant & Session Architecture

- **Total Participants**: 0 (Pending real data ingestion)
- **Total Sessions**: 0
- **Total Keystroke Records**: 0
- **Sessions per Participant**: N/A
- **Validation Criterion**: Minimum 3 distinct participants required for group-aware leakage-safe 3-way partitioning (`Train 70% / Val 15% / Test 15%`).

---

## 3. Behavioral Labels & Target Mapping

- **Observed Unique Labels**: None present in `data/raw/`
- **Target Variable**: `condition` (or equivalent behavioral state indicator)
- **Supported Classes**: Subject to dataset publication taxonomy (e.g. `neutral`, `stressed`, `baseline`, `recovery`).
- **Dynamic Mapping Policy**: Numeric integer classes $[0, \dots, K-1]$ will be assigned dynamically and persisted to `data/processed/label_metadata.json`. No artificial binary bins will be fabricated.

---

## 4. Keystroke Signals & Timing Features

| Signal Category | Status in Engine | Feature Extraction Mapping |
| :--- | :--- | :--- |
| **Dwell Time (Hold Duration)** | Supported | Derived from `release_time - press_time` |
| **Flight Time (Inter-Key Interval)** | Supported | Derived from consecutive keydown transitions |
| **Pause Frequencies** | Supported | Thresholded pauses ($\ge 250\text{ms}, \ge 500\text{ms}, \ge 1000\text{ms}$) |
| **Typing Speed** | Supported | Characters per minute (CPM) and Words per minute (WPM) |
| **Backspace & Correction Events** | Supported | Detected via non-text event markers or backspace counters |
| **Full Word Lexical Timing** | Unavailable | Suppressed under strict Zero-Raw-Text policy |
| **Linguistic Semantic Tokens** | Unavailable | Strictly prohibited under privacy invariants |

---

## 5. Data Quality & Timing Plausibility

The validation engine audits:
- **Timestamp Monotonicity**: Verifies non-decreasing timestamps within participant sessions.
- **Physiological Bounds**:
  - Dwell time: $10\text{ms} \le \text{hold} \le 4000\text{ms}$
  - Flight time: $0\text{ms} \le \text{flight} \le 10000\text{ms}$
- **Missing Value & Duplicate Audit**: Duplicate millisecond rows flagged and removed.

---

## 6. Privacy & Sensitive Field Quarantine

- **Zero-Raw-Text Policy**: All character identities, typed sentences, word strings, and text payloads are intercepted and excluded at intake.
- **Participant Pseudonymization**: Real participant identifiers (student IDs, emails, subject numbers) are transformed into irreversible `usr_<16-hex>` pseudonyms using HMAC-SHA256.
- **Audit Verification**: Every batch is audited with `audit_zero_raw_text()`.

---

## 7. Leakage Audit Specification

Prior to model training, the pipeline enforces strict leakage isolation:
1. **Group Overlap**: $\text{Train}_{\text{users}} \cap \text{Val}_{\text{users}} = \emptyset$ and $\text{Val}_{\text{users}} \cap \text{Test}_{\text{users}} = \emptyset$.
2. **Session Isolation**: Sliding windows are constructed strictly within session boundaries and never bleed across sessions.
3. **Target Cleanliness**: Target labels and condition markers are quarantined from the input feature tensor $X$.

---

## 8. Sequence Configuration Recommendation

- **Default Development Window**: 30 events.
- **Adaptive Sequence Calibration**: When real data is ingested, the engine evaluates the empirical distribution:
  $$\text{Recommended Window} = \min\left(30, \max(10, \lfloor \text{Median Session Event Count} / 2 \rfloor)\right)$$
- Preserves temporal dynamics while avoiding excessive zero-padding or sample truncation.

---

## 9. Decoupled Readiness Verdicts

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ DATASET STATUS:           REAL DATASET REQUIRED -- INTEGRATION PIPELINE READY│
│ MODEL TRAINING READINESS: BLOCKED (Awaiting approved dataset in data/raw/)   │
│ BASELINE READINESS:       BASELINE NOT AVAILABLE FROM DATASET                │
│ TRAINING GATE:            ENGAGED & LOCKED (Zero Fabrication Guarantee)      │
└──────────────────────────────────────────────────────────────────────────────┘
```

- **Model Training Readiness**: Requires $\ge 3$ participants and $\ge 2$ classes.
- **Personal Baseline Readiness**: Decoupled from model training; requires $\ge 5$ longitudinal sessions per participant to establish individual typing deviation norms.
- Real model training is blocked until a valid research dataset is placed in `data/raw/` and successfully passes the automated 15-point validation audit.
- See [docs/dataset_acquisition_checklist.md](dataset_acquisition_checklist.md) for placement and verification instructions.

