# Live Behavioral Session Implementation Architecture

## 1. Overview
The **Live Behavioral Session** is the core interactive system in the Typing Dynamics platform. It captures temporal keystroke micro-timing telemetry, applies rigorous zero-text privacy filtering, extracts 6 canonical physiological features, aggregates sliding temporal sequence windows (30 timesteps), and processes either personalized baseline calibration or real-time typing deviation analysis (TDI) without storing or transmitting raw typed text.

---

## 2. Authoritative Session State Machine
The session lifecycle is governed by an authoritative state machine (`SessionState` in `src/live_typing/session.py`):

```
       ┌─────────────────────────┐
       │          READY          │◄───────────────────────┐
       └────────────┬────────────┘                        │
                    │ START                               │
                    ▼                                     │
       ┌─────────────────────────┐                        │
  ┌───►│        CAPTURING        │                        │
  │    └──────┬────────────┬─────┘                        │
  │           │ PAUSE      │ STOP                         │
  │ RESUME    ▼            ▼                              │
  │    ┌─────────────┐   ┌───────────────────────────┐    │
  └────┤   PAUSED    │   │         STOPPING          │    │
       └─────────────┘   └─────────────┬─────────────┘    │
                                       │ (compute)        │ RESET
                                       ▼                  │
                         ┌───────────────────────────┐    │
                         │        VALIDATING         │    │
                         └─────────────┬─────────────┘    │
                                       │ (complete)       │
                                       ▼                  │
                         ┌───────────────────────────┐    │
                         │         COMPLETED         │────┘
                         └───────────────────────────┘
```

### State Definitions & Guards
| State | Active Controls | Allowed Next Actions | Description |
| :--- | :--- | :--- | :--- |
| **READY** | START enabled; PAUSE/RESUME/STOP disabled | `START` | System idle, awaiting session initiation. Previous buffers cleared. |
| **CAPTURING** | PAUSE, STOP enabled; START/RESUME disabled | `PAUSE`, `STOP`, `ERROR` | Typing canvas active; browser micro-timing telemetry actively ingested. |
| **PAUSED** | RESUME, STOP enabled; START/PAUSE disabled | `RESUME`, `STOP`, `ERROR` | Timing boundaries suspended; incoming keystrokes strictly ignored. |
| **STOPPING** | All controls locked | `VALIDATING` | Session ended; timestamp frozen; final keystrokes flushed. |
| **VALIDATING** | All controls locked | `COMPLETED`, `ERROR` | Quality gate validation (checks duration, event count, monotonic order). |
| **COMPLETED** | START, RESET enabled; PAUSE/RESUME/STOP disabled | `RESET`, `START` | Assessment, metrics, and report persisted and displayed. |
| **ERROR** | RESET enabled | `RESET` | Exception trapped gracefully without crashing UI. |

---

## 3. Capture Architecture & Frontend Bridge
Keystroke capture is isolated to a dedicated, sandboxed frontend component (`src/live_typing/frontend/index.html` via `st.components.v1.html`):

1. **Isolation**: Standard `st.text_input` and `st.text_area` trigger server reruns and do not provide per-event keydown/keyup timestamps. The capture component runs an in-browser event loop that records precise `performance.now()` microsecond-level offsets alongside epoch timestamps (`Date.now()`).
2. **Buffer Batching**: Keystrokes are buffered in the browser canvas and dispatched in batches every 500ms or on 10 accumulated events to prevent Streamlit rerun overhead.
3. **Session Interlock**: When `session_active` is `false`, keyboard event listeners are completely detached and the canvas input area is disabled.
4. **State Machine UI Binding**: Every button on the Physical Control Deck (`START`, `PAUSE`, `RESUME`, `STOP`, `RESET`) is dynamically bound to `engine.session.state`, preventing invalid transitions.

---

## 4. Sanitized Event Payload Schema
The communication boundary between frontend and Python strictly enforces the sanitized telemetry contract.

### Keydown Event
```json
{
  "event_id": "evt_1789929850123_4a",
  "event_type": "keydown",
  "timestamp": 1789929850123.45,
  "key_category": "alphanumeric"
}
```

### Keyup Event
```json
{
  "event_id": "evt_1789929850215_4b",
  "event_type": "keyup",
  "timestamp": 1789929850215.12,
  "key_category": "alphanumeric"
}
```

### Permitted Categories
- `alphanumeric`, `space`, `backspace`, `delete`, `enter`, `modifier`, `punctuation`, `navigation`
- **Zero-Text Invariant**: The actual character value (`key`, `char`, `text`, `value`, `keyCode`) is NEVER sent across the bridge or persisted.

---

## 5. Zero-Text Privacy Policy Enforcements
The application enforces privacy at multiple distinct levels:
1. **Frontend**: Character codes are intercepted in memory, converted immediately to coarse categories (`alphanumeric`, `backspace`), and the raw character variable is discarded before message packaging.
2. **Bridge Filter (`sanitize_event_batch`)**: Ingested payloads are scanned against `FORBIDDEN_PAYLOAD_FIELDS` (`key`, `text`, `character`, `char`, `raw_text`, `input_text`, etc.). In strict mode, any dirty payload triggers a `PrivacyViolationError` and is immediately quarantined.
3. **Storage Gate**: Neither SQLite nor PostgreSQL databases have text/content columns in `sessions`, `typing_metrics`, `assessments`, or `audit_logs`.
4. **Report Generator**: Automated privacy assertions ensure no text content exists in JSON, Markdown, or tabular exports.

---

## 6. Feature Extraction & Windowing Pipeline
Paired keydown/keyup events are passed into `LiveFeatureBuffer`:

### Physiological Keystroke Features (Canonical 6-D Vector)
1. **Dwell Time (`dwell_time`)**: Time elapsed between `keydown` and matching `keyup` of the same key.
2. **Flight Time (`flight_time`)**: Latency between the release of key $k-1$ and the press of key $k$.
3. **Pause Duration (`pause_duration`)**: Inter-keystroke intervals exceeding the pause threshold (500 ms).
4. **Typing Speed (`typing_speed`)**: Rolling estimate of words-per-minute calculated over sliding 10-event spans.
5. **Backspace Flag (`backspace`)**: Binary flag indicating error-correction keystrokes.
6. **Error Flag (`error_flag`)**: Binary flag indicating correction or backspace activity.

### Temporal Sequence Windowing
- Window Length: 30 events (`[30, 6]` tensor).
- Stride: 10 events.
- Compatibility: Exact dimensional match with the trained LSTM recurrent classifier.

---

## 7. Calibration vs. Analysis Protocols
### Calibration Mode
- Purpose: Learns the participant's natural typing distribution (means and variances across features).
- Threshold: Requires at least 5 completed calibration sessions before establishing an active baseline.
- Model Decoupling: Does not trigger deep learning inference; strictly accumulates baseline profile.

### Analysis Mode
- Purpose: Compares the current session's typing dynamics against the established personal baseline.
- Readiness Check: If fewer than 5 calibration sessions exist, gracefully reports `BASELINE NOT READY`.
- Metric: Calculates the **Typing Deviation Index (TDI)** measuring statistical divergence.
- Clinical Disclaimer: Non-diagnostic wording enforced across all displays and reports.

---

## 8. Deep Learning Model Readiness Gate
The production LSTM executes only when all scientific gates pass:
1. Genuine trained model checkpoint exists (`models/lstm_model.keras`).
2. Fitted training scaler exists (`models/feature_scaler.pkl`).
3. Label mapping artifact exists (`models/label_mapping.json`).
4. At least one valid 30-event window `[30, 6]` is available.
5. If any artifact is missing, the engine returns `MODEL_NOT_READY` with an explanatory reason rather than fabricating predictions.

---

## 9. Database Persistence Architecture
- **Local SQLite**: `database/mental_state.db` initialized with tables: `sessions`, `typing_metrics`, `assessments`, `baseline_profiles`, `audit_logs`.
- **Cloud PostgreSQL**: Supabase Session Pooler detected via `DATABASE_URL` from Streamlit secrets.
- **Fail-Safe Persistence**: If database writing encounters a connection failure, in-memory assessment results remain intact and are presented to the user without crashing the runtime.
