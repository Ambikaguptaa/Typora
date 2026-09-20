# Live Typing Behavior Capture Engine — Technical Architecture & Privacy Manual

## 1. Overview & System Purpose

The **Live Typing Behavior Capture Engine** provides real-time, privacy-preserving keystroke dynamics instrumentation within the Mental-State Detection System. It enables research participants to perform controlled typing tasks inside a specialized Streamlit console component, deriving fine-motor behavioral timing signals while rigorously eliminating the possibility of keystroke logging, character capture, or textual surveillance.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           BROWSER SANDBOX (IFRAME)                              │
│                                                                                 │
│   [ Controlled Focus Input Area ]                                               │
│             │                                                                   │
│             ▼ (Native keydown / keyup events)                                   │
│   [ Token & Micro-Timing Classifier ]                                           │
│   - Extracts performance.now() relative timestamp                              │
│   - Converts key to abstract category: k_alpha, k_backspace, k_enter, k_space   │
│   - STRICTLY OMITS: event.key character string, textarea.value text             │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │ postMessage (JSON event batch)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      PYTHON BACKEND (STREAMLIT RUNTIME)                         │
│                                                                                 │
│   [ Privacy Filter Layer ]                                                      │
│   - Audits for forbidden keys ('key', 'char', 'text', 'word', 'value', etc.)   │
│   - Rejects violations: logs PRIVACY_EVENT_REJECTED or raises PrivacyViolation  │
│             │                                                                   │
│             ▼ Sanitized RawBrowserEvents                                        │
│   [ Stateful Event Normalizer ]                                                 │
│   - Pairs keydown + keyup timestamps into single TypingEvents                   │
│   - Calculates dwell_time = release_time - press_time                          │
│   - Calculates flight_time = press_time[i] - release_time[i-1]                  │
│   - Detects rapid editing bursts (is_correction flag)                           │
│             │                                                                   │
│             ▼ Paired TypingEvents                                               │
│   [ Live Feature Buffer & Telemetry ]                                           │
│   - Rolling FIFO buffer (capacity 1000 events)                                  │
│   - Real-time telemetry: Mean dwell/flight, pause rate, WPM proxy, error rate   │
│   - Canonical 3D temporal sequence arrays (N, 30, 6) matching LSTM input        │
│             │                                                                   │
│             ▼ Session Completion                                                │
│   [ Technical Quality Validator ]                                               │
│   - Min events: 15, Min duration: 3.0s, Min valid dwells: 5                     │
│   - Returns: SESSION_VALID or INSUFFICIENT_DATA                                 │
│             │                                                                   │
│             ▼ Decision Gates                                                    │
│   ┌───────────────────────────┐         ┌─────────────────────────────┐         │
│   │     Model Readiness       │         │     Baseline Readiness      │         │
│   │ 🔒 MODEL_NOT_READY        │         │ 📊 BASELINE_NOT_READY       │         │
│   │ (Real dataset required;   │         │ (Requires >= 5 calibration  │         │
│   │  zero fake predictions)   │         │  sessions before TDI)       │         │
│   └───────────────────────────┘         └─────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Zero-Raw-Text Privacy Invariant & Security Architecture

### 2.1 Complete Absence of Global Keyboard Hooks
- **No OS-level hooks**: Packages like `pynput`, `keyboard`, or Windows API hooks (`SetWindowsHookEx`) are **strictly prohibited** and completely absent from the codebase.
- **Focus Isolation**: Listeners are attached strictly to the dedicated `<textarea>` DOM element inside an isolated `<iframe>` sandbox. Keystrokes typed in other browser tabs, other applications, or other system windows are technically unreachable.

### 2.2 Client-Side Character Stripping
In `src/live_typing/frontend/index.html`, incoming DOM events are mapped immediately to categorical tokens:
```javascript
// Disallowed: e.key, e.target.value
// Allowed: relative timing & abstract token
const classification = classifyKey(event); // returns 'k_alpha', 'k_backspace', etc.
const timingRecord = {
    event_type: eventType,
    timestamp_ms: performance.now(),
    key_token: classification.key_token,
    is_backspace: classification.is_backspace,
    is_enter: classification.is_enter,
    is_space: classification.is_space
};
```

### 2.3 Server-Side Privacy Filtering (`privacy_filter.py`)
Incoming payloads are audited against `FORBIDDEN_EVENT_KEYS`:
`{"key", "char", "character", "text", "word", "sentence", "message", "typed_text", "password", "user_input", "content", "value", "input", "keystring", "raw_input", "raw_text", "val", "str"}`

- If any forbidden field is detected:
  - In strict mode: raises `PrivacyViolationError`.
  - In filtering mode: drops the event and logs a sanitized token `PRIVACY_EVENT_REJECTED` without echoing payload content.
- Key tokens must conform to the regex `^k_[a-zA-Z0-9_]+$`. Any raw single character is immediately rejected.

---

## 3. Micro-Timing Normalization & Event Pairing (`event_normalizer.py`)

A stateful pairing engine pairs independent `down` and `up` browser events:
1. **Dwell Time**:
   $$\text{dwell\_time} = \text{release\_timestamp} - \text{press\_timestamp}$$
   - Clamped to $[10.0\text{ ms}, 4000.0\text{ ms}]$. Implausible or negative values caused by clock jitter are discarded.
2. **Flight Time / Inter-Key Interval (IKI)**:
   $$\text{flight\_time}_i = \max(0.0, \text{press\_timestamp}_i - \text{release\_timestamp}_{i-1})$$
   - Computed consecutively between sequential key releases and presses. First key of a session receives `None` (`np.nan`).
3. **Correction & Editing Detection**:
   - Consecutive backspaces with inter-arrival $< 400\text{ ms}$ are tagged as `is_correction = True` to capture rapid cognitive revision bursts without recording what was deleted.

---

## 4. Canonical Feature Buffer & Temporal Sequences (`feature_buffer.py`)

### 4.1 Canonical Feature Columns
The live buffer formats events to match the offline data engineering feature schema:
1. `dwell_time`: Duration key was depressed (ms).
2. `flight_time`: Transition latency from prior key release (ms).
3. `pause_duration`: Flight latency if $> 500\text{ ms}$, else $0.0\text{ ms}$.
4. `typing_speed`: Rolling cadence estimation proxy (WPM).
5. `backspace`: Indicator flag ($1.0$ if backspace, else $0.0$).
6. `error_flag`: Indicator flag ($1.0$ if backspace or correction burst, else $0.0$).

### 4.2 3D Recurrent Sequence Slicing
- Buffer slices observations into sliding windows of shape:
  $$(N_{\text{windows}}, \text{sequence\_length}, 6)$$
  where default $\text{sequence\_length} = 30$, $\text{sequence\_stride} = 10$.
- If fewer than 30 events are buffered, returns an empty tensor `(0, 30, 6)`.

---

## 5. Technical Quality Validation (`validation.py`)

Every session undergoes technical quality validation upon stopping:
- $\text{total\_paired\_events} \ge 15$
- $\text{active\_duration} \ge 3.0\text{ seconds}$
- $\text{valid\_dwell\_events} \ge 5$
- $\text{valid\_flight\_events} \ge 4$

**Non-Diagnostic Rule**: If quality criteria are not satisfied, the verdict is strictly `INSUFFICIENT_DATA`. The system refuses to evaluate, categorize, or classify partial or erratic sessions.

---

## 6. Honest Gating & Anti-Fabrication Guarantees (`live_pipeline.py`)

### 6.1 Model Readiness Gate
In strict alignment with academic ethics and data engineering integrity:
- If no approved real research dataset exists in `data/raw/`, production LSTM training is gated.
- The live pipeline reports:
  ```json
  {
    "status": "MODEL_NOT_READY",
    "ready": false,
    "reason": "No verified production model trained on real research data."
  }
  ```
- **Zero Fabrication**: The system never outputs fake probabilities, pseudo-random predictions, or fabricated mental state labels.

### 6.2 Personal Baseline Gate
- An individual motor typing baseline requires longitudinal consistency.
- Minimum threshold: **5 completed calibration sessions**.
- For sessions $1$ through $4$, reports:
  ```json
  {
    "status": "BASELINE_NOT_READY",
    "ready": false,
    "completed_sessions": 1,
    "required_sessions": 5,
    "reason": "Minimum 5 calibration sessions required to establish a personal baseline."
  }
  ```
- Once $\ge 5$ sessions are logged, computes the non-diagnostic **Typing Deviation Index (TDI, 0–100)**.

---

## 7. Mandatory Academic & Non-Diagnostic Disclaimer

> ⚠️ **ACADEMIC RESEARCH DISCLAIMER**:
> This system analyzes motor behavioral typing patterns for experimental research purposes. It does not provide medical, clinical, or psychiatric diagnoses, nor does it screen for anxiety, depression, or cognitive disorders. All metrics represent fine-motor timing variations relative to statistical baselines.
