# Privacy Threat Model and Data Protection Architecture

**Project**: Mental-State Detection System using Typing Behavior  
**Document**: Architectural Privacy Threat Model & Security Posture  
**Classification**: Academic Research Specification  
**Disclaimer**: *This system provides behavioral typing pattern estimates and is strictly not a medical diagnostic tool. It never records, stores, or processes typed character content.*

---

## 1. Executive Summary

Keystroke dynamics analysis involves capturing fine-grained temporal patterns of human-computer interaction. While powerful for estimating behavioral strain, keystroke analysis introduces substantial privacy risks if improperly handled—most critically, the accidental logging of sensitive typed communications, passwords, or personal identity markers.

This document formalizes the **Threat Model**, **Data Protection Architecture**, and **Residual Risk Profile** for the academic project, establishing strict technical controls that enforce:
1. **Zero-Raw-Text Invariant**: Pure timing metadata extraction; complete elimination of character text, word tokens, and keystroke payloads.
2. **Keyed Pseudonymization**: HMAC-SHA256 irreversible pseudo-identifiers separating behavioral metrics from real-world identities.
3. **Authenticated Encryption at Rest**: Fernet symmetric encryption (AES-128-CBC with SHA-256 HMAC) for persistent baselines and assessment records.
4. **Differential Privacy**: Laplace mechanism noise injection for population-level research statistics.
5. **Data Minimization & Lifecycle Management**: Automated expiration schedules and GDPR Article 17 ("Right to be Forgotten") participant purge capabilities.

---

## 2. Asset Classification & Sensitivity Hierarchy

| Asset Name | Sensitivity Level | Description | Cryptographic Controls |
| :--- | :--- | :--- | :--- |
| **Raw Keystroke Events** | `HIGHLY_SENSITIVE` | Press/release millisecond timestamps | Zero-text quarantine, HMAC user ID, 7-day retention |
| **Participant Identifiers** | `HIGHLY_SENSITIVE` | Plaintext names, student IDs, emails | Immediately transformed into `usr_<16-hex>` via HMAC-SHA256 |
| **Timing Feature Sets** | `SENSITIVE` | Dwell time, flight time, pause durations, typing speed | Pseudonymized, stored in isolated sessions (90-day retention) |
| **Personal Baselines** | `SENSITIVE` | Individual mean, standard deviation, IQR, and TDI | Encrypted at rest (Fernet), pseudonymized |
| **Behavioral Assessments**| `SENSITIVE` | Strain estimates, entropy, prediction margins | Encrypted at rest, pseudonymized (180-day retention) |
| **Model Weights & Config**| `INTERNAL` | LSTM architecture parameters, feature scalers | Versioned, access-controlled |
| **Security Audit Logs** | `INTERNAL` | System events, access control decisions, errors | Redacted, append-only JSONL format |

---

## 3. Threat Model (STRIDE-Aligned)

### Threat 1: Keystroke Reconstruction (Information Disclosure)
- **Threat Description**: An adversary gaining access to the keystroke event stream attempts to reconstruct typed passwords, emails, private messages, or confidential notes.
- **Attack Vector**: Interception of raw event streams, inspection of SQLite tables, or inspection of temporary CSV dumps.
- **Mitigation**: 
  - Strict **Zero-Raw-Text Policy**: The capture layer, dataset adapter, and feature extraction pipeline reject and quarantine columns named `key`, `char`, `text`, `word`, `payload`, `password`, or `raw_input`.
  - Only numeric timestamps (`press_time`, `release_time`) and relative intervals are processed. Reconstructing linguistic content from pure inter-key timings without character identities is mathematically intractable for arbitrary text.

### Threat 2: Participant Re-identification (Spoofing / Tampering)
- **Threat Description**: An external attacker or unauthorized researcher links anonymized typing features back to a specific individual student or employee.
- **Attack Vector**: Dictionary attacks against unsalted or weakly hashed user IDs, or cross-referencing session timestamps with network access logs.
- **Mitigation**:
  - Keyed **HMAC-SHA256 Pseudonymization**: Hashes combine a secret key stored in server-side environment variables (`PSEUDONYMIZATION_SECRET`) with the user ID, producing `usr_<16-hex>`.
  - Without the server secret, rainbow table and precomputation attacks are computationally infeasible.

### Threat 3: Membership & Attribute Inference on ML Models (Information Disclosure)
- **Threat Description**: An adversary issues repeated queries against model endpoints or aggregate statistics to infer whether a specific target was part of the training cohort, or to extract demographic/behavioral attributes.
- **Attack Vector**: Model inversion attacks, differential analysis of population averages before and after a participant joins.
- **Mitigation**:
  - **Differential Privacy (Laplace Mechanism)**: Calibrated Laplace noise $\text{Laplace}(0, \Delta / \epsilon)$ is injected into aggregate queries (such as population typing speed or pause frequencies).
  - Sensitivity bounds are strictly enforced via clipping: $\Delta = (b - a) / N$.
  - Privacy budget tracking logs cumulative $\epsilon$ consumption under composition theorems.

### Threat 4: Unauthorized Local Data Exfiltration (Elevation of Privilege)
- **Threat Description**: An unauthorized local user or insider copies SQLite database files, baseline JSONs, or assessment reports from disk.
- **Attack Vector**: Direct disk read or theft of workstation storage media.
- **Mitigation**:
  - **Fernet Authenticated Encryption at Rest**: Sensitive baselines and assessment outputs are encrypted using 128-bit AES in CBC mode with PKCS7 padding and authenticated via HMAC-SHA256.
  - Corrupted or tampered ciphertexts are detected immediately upon decryption attempt and trigger security audit warnings (`DecryptionError`).

### Threat 5: Indefinite Data Accumulation (Compliance Violation)
- **Threat Description**: Stale participant behavioral records linger indefinitely, increasing exposure surface over time and violating privacy regulations (e.g., GDPR, CCPA).
- **Attack Vector**: Long-term disk persistence and lack of lifecycle pruning.
- **Mitigation**:
  - Automated **Lifecycle Retention Schedules**:
    - Raw event metadata: 7 days.
    - Session timing features and baselines: 90 days.
    - Assessment reports: 180 days.
  - **GDPR Article 17 "Right to be Forgotten"**: The `purge_participant_data` utility purges all associated records across database tables and storage artifacts upon participant withdrawal.

---

## 4. Architectural Controls & Data Flow

```
[Keystroke Timing Sensors]
          │
          ▼
┌────────────────────────────────────────────────┐
│ 1. Zero-Text Sanitization & Ingestion          │
│    - Purge 'key', 'char', 'text', 'word', etc. │
│    - Validate numeric duration integrity      │
└──────────────────────┬─────────────────────────┘
                       │ Sanitized Timestamps
                       ▼
┌────────────────────────────────────────────────┐
│ 2. Data Minimization & Pseudonymization        │
│    - HMAC-SHA256(Secret, UserID) -> usr_xxxx  │
│    - Strip extraneous participant attributes   │
└──────────────────────┬─────────────────────────┘
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────────┐
│ 3. Feature Pipeline     │     │ 4. Storage & Encryption     │
│    - Dwell & flight     │     │    - Fernet AES-128-CBC     │
│    - Pause frequencies  │     │    - Storage Policy Engine  │
│    - Typing speed (CPM) │     │    - Retention & Expiration │
└──────────────┬──────────┘     └──────────────┬──────────────┘
               │                               │
               ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────────┐
│ 5. Behavioral Model     │     │ 6. Research Queries (DP)    │
│    - Uncertainty & TDI  │     │    - Laplace Mechanism      │
│    - Reliability Gate   │     │    - Privacy Budget Tracker │
└─────────────────────────┘     └─────────────────────────────┘
```

---

## 5. Explicit Limitations and Non-Claims

In alignment with academic honesty and rigorous security engineering standards:
1. **No Claim of Unbreakable Anonymity**: Under strong adversary models where an attacker possesses comprehensive ground-truth behavioral profiles of a small closed cohort, biometric stylometry might theoretically facilitate re-identification.
2. **Host OS Boundary**: Physical security of the machine hosting the SQLite database and environment variables (`.env`) is the responsibility of the system administrator.
3. **Non-Diagnostic Scope**: The system measures *behavioral typing variations* (dwell time variance, pause fluctuations). It does not diagnose, treat, or monitor clinical psychological conditions, neurological disorders, or psychiatric illnesses.
