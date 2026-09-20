# Real Research Dataset Acquisition Checklist & Ingestion Guide

**Project**: Mental-State Detection System using Typing Behavior  
**Document**: Research Dataset Acquisition, Licensing, and Verification Protocol  
**Current Pipeline Status**: `REAL DATASET REQUIRED -- INTEGRATION PIPELINE READY`

---

## 1. Executive Summary

The entire software pipeline—including schema normalization, Zero-Raw-Text privacy filtering, HMAC-SHA256 pseudonymization, 15-point empirical validation, 6-dimension leakage auditing, feature extraction, sequence generation, group-aware partitioning, and LSTM training orchestration—is **100% complete and hardened**.

In accordance with strict academic and scientific ethics:
- **No real model is trained yet.**
- **No empirical accuracy is claimed.**
- **No fake dataset or predictions have been fabricated.**

The production training gate is **LOCKED** until an approved real research dataset is placed into `data/raw/`. This guide explains the exact protocol to acquire, verify, and ingest an approved dataset.

---

## 2. Supported Dataset Formats & Schemas

### A. Supported Formats
The ingestion engine automatically recognizes and parses:
- Comma-Separated Values (`.csv`)
- Tab-Separated Values (`.tsv`)
- Microsoft Excel (`.xlsx`, `.xls`)
- Apache Parquet (`.parquet`)
- JavaScript Object Notation (`.json`, `.jsonl`)

### B. Standard Canonical Event Schema
If providing a pre-processed or custom dataset, format columns to match the **Canonical Event Schema** (`src/data_engineering/canonical_schema.py`):

| Column Name | Data Type | Required/Optional | Description |
| :--- | :--- | :--- | :--- |
| `participant_id` | String | **Required** | Participant identifier (pseudonymized, e.g. `p_01`, `sub_04`). |
| `session_id` | String | **Required** | Session or trial identifier (e.g. `sess_1`, `trial_02`). |
| `timestamp` | Float | **Required** | Millisecond event timestamp (monotonically increasing per session). |
| `event_type` | String | **Required** | Keystroke action type (`down`, `up`, `press`, `release`). |
| `key_identifier` | String | **Required** | **Abstract key token ONLY** (e.g. `k_01`, `k_enter`, `k_backspace`). **NEVER raw text.** |
| `press_time` | Float | **Required** | Millisecond timestamp when the key was depressed. |
| `release_time` | Float | **Required** | Millisecond timestamp when the key was released (or `NaN` if single-timestamp). |
| `condition` | String | **Required** | Experimental condition label (e.g. `neutral`, `stressed`). |
| `pressure` | Float | Optional | Touchscreen pressure or force sensor reading. |
| `x` | Float | Optional | Horizontal touch coordinate (mobile devices). |
| `y` | Float | Optional | Vertical touch coordinate (mobile devices). |

---

## 3. Strict Scientific & Privacy Invariants

### 1. Zero-Raw-Text Policy
Under no circumstances may raw typed text enter `data/raw/`:
- **Prohibited columns**: `key`, `char`, `text`, `word`, `sentence`, `message`, `typed_text`, `password`, `user_input`, `content`.
- Any file containing raw text strings will be **automatically rejected and quarantined** by the validation pre-flight audit.
- Key identifiers must be hashed (`k_<hash>`) or category-tokenized (`k_alpha`, `k_space`, `k_backspace`).

### 2. Participant Privacy & Pseudonymization
- Real participant names, emails, student IDs, or IP addresses must never be stored.
- Identifiers are pseudonymized via salted HMAC-SHA256 into deterministic tokens (`usr_<16-hex>`).

### 3. Behavioral Target Semantics (Non-Diagnostic)
- Labels must represent **dataset-derived behavioral classification targets** (e.g. `neutral` vs `stressed` rhythm under induced experimental tasks).
- **Prohibited descriptions**: "Clinical depression", "anxiety disorder diagnosis", "mental illness detector". The system is a research behavioral prototype, NOT a clinical diagnostic tool.

---

## 4. Minimum Dataset Requirements

To successfully pass the pre-flight verification gate, the dataset must satisfy:

1. **Participants**: $\ge 3$ distinct participants (required for group-aware 3-way split: 70% train / 15% val / 15% test with zero participant overlap).
2. **Classes**: $\ge 2$ distinct behavioral condition classes (e.g. `neutral` and `stressed`).
3. **Sessions**: $\ge 1$ session per participant for model training. (Note: $\ge 5$ sessions per participant are required if personal baseline deviation is to be computed).
4. **Events per Session**: $\ge 30$ consecutive keystroke events to extract recurrent LSTM sequence windows.
5. **Timestamps**: Monotonically non-decreasing timestamps without large negative timing jumps.

---

## 5. Step-by-Step Placement & Ingestion Protocol

When an approved dataset is obtained:

### Step 1: Place the File in `data/raw/`
Copy the raw dataset file into:
```
data/raw/
```
Example filenames:
- `data/raw/mobilestress.csv`
- `data/raw/cmu_keystroke.csv`
- `data/raw/keystroke_dataset.parquet`

### Step 2: Update Metadata Manifest (Optional but Recommended)
Edit `data/raw/dataset_manifest.json` to record dataset provenance:
```json
{
  "dataset_name": "MobileStress",
  "version": "1.0",
  "source": "Academic Research Study",
  "access_status": "READY",
  "access_type": "Institutional Data Transfer Agreement",
  "labels": ["neutral", "stressed"]
}
```

### Step 3: Run the Data Engineering Pipeline
Execute the pipeline to inspect, validate, clean, and extract features:
```powershell
.\.venv\Scripts\python.exe -m src.data_engineering.pipeline
```

### Step 4: Review Automated Validation Reports
Inspect generated reports under `data/processed/`:
- `data/processed/real_dataset_validation_report.json`
- `data/processed/feature_compatibility_report.json`
- `data/processed/privacy_audit_report.json`
- `data/processed/leakage_audit_report.json`

Verify that all reports conclude with:
```json
"passed": true
```

### Step 5: Execute Staged Production Training
Once all 7 pre-flight validation stages pass, trigger LSTM training:
```powershell
.\.venv\Scripts\python.exe -m src.deep_learning.train
```

The CLI will execute all 8 stages:
```
[1/8] Dataset discovery ........ PASS
[2/8] Schema validation ....... PASS
[3/8] Privacy audit ........... PASS
[4/8] Label validation ....... PASS
[5/8] Leakage audit ........... PASS
[6/8] Feature validation ..... PASS
[7/8] Training readiness ..... PASS
[8/8] Training execution ..... COMPLETE
```

### Step 6: Verify Test Suite
Run the full automated test suite:
```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

---

## 6. What If Access Is Blocked or Pending?

If your academic institution is still reviewing data transfer agreements:
- **Do not download unverified Kaggle re-uploads.**
- **Do not fabricate synthetic records in `data/raw/`.**
- You may safely run developer smoke tests using the firewalled sample data:
  ```powershell
  .\.venv\Scripts\python.exe -m src.deep_learning.train --allow-synthetic-smoke-test
  ```
  *(This allows validating code paths without saving fake production models or making empirical performance claims).*
