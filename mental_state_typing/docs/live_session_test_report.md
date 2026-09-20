# Live Behavioral Session Automated Test Report

**Execution Environment**: Python 3.12.10 (.venv) / Windows / Headless  
**Testing Methodology**: 100% Automated Terminal / Headless (Zero Browser Testing)  
**Declaration**: Application UI was validated through Streamlit AppTest/headless automated testing.

---

## 1. Test Suite Summary Matrix

| Testing Layer | Command | Status | Details |
| :--- | :--- | :--- | :--- |
| **Pytest Full Suite** | `python -m pytest tests -v` | **343 PASSED** | 0 failed, 0 errors, 54 warnings, 32.51s |
| **Headless Live Session** | `python -m pytest tests/test_live_session_headless.py -v` | **30 PASSED** | 100% coverage of 30 lifecycle & state requirements |
| **Capture Component Contract** | `python -m pytest tests/test_capture_contract.py -v` | **6 PASSED** | Sanitized schema, keydown/keyup pairing, deduplication |
| **Streamlit AppTest** | `python -m pytest tests/test_apptest_navigation.py -v` | **5 PASSED** | All 9 pages, empty state, state-machine deck buttons |
| **Database Persistence & Pooler** | `python -m pytest tests/test_database_persistence.py -v` | **32 PASSED** | SQLite CRUD, Session Pooler, SELECT 1, mask checks |
| **Security & Zero-Text Policy** | `python -m pytest tests/test_privacy_regression.py -v` | **2 PASSED** | Codebase-wide token scan, dirty payload quarantine |
| **Live Smoke Test (CLI)** | `python scripts/live_session_smoke_test.py` | **12/12 PASS** | All 12 lifecycle steps verified sequentially |
| **Subprocess Crash Isolation** | `python -c "import ..."` (4 isolated commands) | **4/4 PASS** | Streamlit, TensorFlow, NumPy, App Import |
| **Static Code Compilation** | `python -m compileall app.py src database tests scripts` | **PASS** | Exit code 0, all files compiled cleanly |
| **Dependency Integrity** | `python -m pip check` | **PASS** | No broken requirements found |
| **Streamlit Headless Process** | `streamlit run app.py --server.headless true --server.port 8599` | **PASS** | Process running, Uvicorn started, zero crashes |

---

## 2. Detailed Execution Logs

### A. Full Pytest Suite Run
```bash
$ .\.venv\Scripts\python.exe -m pytest tests -v
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\Typora\mental_state_typing
plugins: anyio-4.15.1
collected 343 items

tests/test_adapters.py::test_dataset_adapter_protocols PASSED            [  0%]
...
tests/test_live_session_headless.py::test_01_ready_initial_state PASSED   [ 65%]
tests/test_live_session_headless.py::test_02_start_transition PASSED      [ 66%]
...
tests/test_live_session_headless.py::test_30_no_raw_text_persistence PASSED [ 74%]
tests/test_capture_contract.py::test_contract_valid_keydown_and_keyup_payloads PASSED [ 74%]
tests/test_capture_contract.py::test_contract_malformed_payload_rejected_safely PASSED [ 75%]
tests/test_capture_contract.py::test_contract_duplicate_payload_ignored PASSED [ 75%]
tests/test_capture_contract.py::test_contract_raw_text_never_persisted PASSED [ 75%]
tests/test_capture_contract.py::test_contract_timestamps_processed_correctly PASSED [ 76%]
tests/test_capture_contract.py::test_contract_event_ordering_preserved PASSED [ 76%]
tests/test_apptest_navigation.py::test_app_starts_cleanly PASSED         [ 76%]
tests/test_apptest_navigation.py::test_empty_session_state_first_run PASSED [ 77%]
tests/test_apptest_navigation.py::test_navigation_across_all_pages PASSED [ 77%]
tests/test_apptest_navigation.py::test_live_session_deck_state_machine_and_buttons PASSED [ 77%]
tests/test_apptest_navigation.py::test_repeated_reruns_stability PASSED  [ 78%]
...
====================== 343 passed, 54 warnings in 32.51s ======================
```

### B. Live Session Smoke Test (Terminal Simulator)
```bash
$ .\.venv\Scripts\python.exe scripts/live_session_smoke_test.py
LIVE SESSION SMOKE TEST
-----------------------
START: PASS
EVENT CAPTURE: PASS
FEATURE EXTRACTION: PASS
30-EVENT WINDOW: PASS
PAUSE: PASS
RESUME: PASS
STOP: PASS
VALIDATION: PASS
BASELINE/ANALYSIS: PASS
PERSISTENCE: PASS
REPORT: PASS
RESET: PASS
```

### C. Subprocess Crash Isolation
```bash
$ .\.venv\Scripts\python.exe -c "import streamlit; print('OK STREAMLIT')"
OK STREAMLIT (exit code 0)

$ .\.venv\Scripts\python.exe -c "import tensorflow; print('OK TENSORFLOW')"
OK TENSORFLOW (exit code 0)

$ .\.venv\Scripts\python.exe -c "import numpy; print('OK NUMPY')"
OK NUMPY (exit code 0)

$ .\.venv\Scripts\python.exe -c "import app; print('APP IMPORT OK')"
[STARTUP STEP 1] Initializing application environment and standard libraries...
[STARTUP STEP 2] Loading scientific and UI libraries (numpy, pandas, plotly, streamlit)...
[STARTUP STEP 3] Loading database persistence layer (pg8000 pure-python driver)...
[STARTUP STEP 4] Loading behavioral and ML feature pipeline modules...
[STARTUP STEP 5] App module imports completed successfully.
APP IMPORT OK (exit code 0)
```

### D. Static Compilation and Pip Integrity
```bash
$ .\.venv\Scripts\python.exe -m compileall app.py src database tests scripts
Listing 'src'...
Listing 'database'...
Listing 'tests'...
Listing 'scripts'...
Exit code: 0

$ .\.venv\Scripts\python.exe -m pip check
No broken requirements found.
Exit code: 0
```

### E. Database Diagnostics & Pooler Health
```bash
$ .\.venv\Scripts\python.exe -c "from database.database import check_connection, check_database_health; print(check_connection(), check_database_health())"
DB CHECK CONNECTION STATUS: True
DB HEALTH: <DatabaseHealth status=DATABASE_READY ready=True backend=sqlite>
```
Supabase Session Pooler configuration:
- Pooler host detection: PASS
- Direct host avoidance: PASS
- Password masking: PASS
- SELECT 1 health check: PASS
