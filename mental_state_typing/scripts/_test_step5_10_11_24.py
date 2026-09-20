import sys
sys.path.insert(0, '.')

from src.integration import BehavioralEngine, SessionType
from src.live_typing import SessionState

def feed_events(engine, n_pairs=40, base_ts=10000.0):
    events = []
    for i in range(n_pairs):
        p = base_ts + i * 195.0 + (i % 4) * 15.0
        r = p + 92.0 + (i % 6) * 22.0
        is_space = (i + 1) % 8 == 0
        is_bk = i == 20
        tok = "k_space" if is_space else ("k_backspace" if is_bk else "k_alpha")
        events.append({"event_id": f"d{i}", "event_type": "down", "timestamp_ms": p, "key_token": tok,
                       "is_backspace": is_bk, "is_enter": False, "is_space": is_space})
        events.append({"event_id": f"u{i}", "event_type": "up", "timestamp_ms": r, "key_token": tok,
                       "is_backspace": is_bk, "is_enter": False, "is_space": is_space})
    return engine.ingest_raw_events(events, strict_privacy=False)

print("=" * 60)
print("STEP 5: STOPPING/VALIDATING/COMPLETED intermediate transition test")
print("=" * 60)

# We need to observe the intermediate states. Since they occur within stop_session,
# we can monkey-patch transition_to to capture history.
engine = BehavioralEngine()
session = engine.session
transitions = []
_orig_transition = session.transition_to
def capture_transition(target):
    transitions.append((session.state.value, target.value))
    _orig_transition(target)
session.transition_to = capture_transition

print("\n1. Initial state:", session.state.value)
assert session.state == SessionState.READY

engine.start_session()
print("2. After START:", session.state.value)
assert session.state == SessionState.CAPTURING

# Feed events BEFORE the transitions capture
p = feed_events(engine, 40)
print(f"   (Fed events: {p} pairs added)")

transitions.clear()
res = engine.stop_session()
print(f"3. Transitions observed during STOP: {transitions}")

# Verify STOPPING and VALIDATING both appeared (intermediate states before COMPLETED)
observed_states = [t[1] for t in transitions]
print(f"   -> Intermediate states seen: {observed_states}")
assert "STOPPING" in observed_states, "Missing STOPPING intermediate state!"
assert "VALIDATING" in observed_states, "Missing VALIDATING intermediate state!"
assert observed_states[-1] == "COMPLETED", f"Final should be COMPLETED, got {observed_states[-1]}"
print("   ✓ PASS: STOPPING → VALIDATING → COMPLETED transitions all observed")
print("   Final session state:", session.state.value)

# Verify result
print(f"\n4. Result pipeline_status: {res.pipeline_status}")
print(f"   Quality is_valid: {res.data_quality.is_valid}")
print(f"   Feature events: {res.feature_summary.event_count}")
print(f"   Baseline status: {res.baseline_result.status}")
print(f"   Model status: {res.model_result.status}")
assert res.data_quality.is_valid, "Data quality should pass with 40+ events"
assert res.pipeline_status == "COMPLETED"
print("   ✓ PASS: Assessment result is valid")

# Test ERROR handling (introduce an error via monkey-patch)
print("\n" + "=" * 60)
print("STEP 5: ERROR state transition test")
print("=" * 60)

engine2 = BehavioralEngine()
transitions2 = []
_orig_t2 = engine2.session.transition_to
def cap2(target):
    transitions2.append(target.value)
    _orig_t2(target)
engine2.session.transition_to = cap2

# Break something in the pipeline - make validate_session_quality throw
import src.live_typing.validation as validation_mod
_orig_validate = validation_mod.validate_session_quality
def broken_validate(*a, **kw):
    raise RuntimeError("Simulated validation crash")
validation_mod.validate_session_quality = broken_validate

try:
    engine2.start_session()
    feed_events(engine2, 50)
    transitions2.clear()
    res2 = engine2.stop_session()
    print(f"Final state after broken pipeline: {engine2.session.state.value}")
    print(f"Pipeline status in result: {res2.pipeline_status}")
    print(f"Transitions observed: {transitions2}")
    # After error, should end in ERROR
    assert "STOPPING" in transitions2, "Should still hit STOPPING before error"
    assert engine2.session.state == SessionState.ERROR, f"Should be ERROR, got {engine2.session.state}"
    assert res2.pipeline_status == "ERROR", f"Result status should be ERROR, got {res2.pipeline_status}"
    print("   ✓ PASS: Pipeline error caught and transitioned to ERROR state")
finally:
    # Restore
    validation_mod.validate_session_quality = _orig_validate

# Test Pause → Resume → Stop full cycle
print("\n" + "=" * 60)
print("Full Pause/Resume/Stop cycle with state verification")
print("=" * 60)

engine3 = BehavioralEngine()
engine3.start_session()
assert engine3.session.state == SessionState.CAPTURING
pairs = feed_events(engine3, 20)
print(f"Started + {pairs} initial events. State: {engine3.session.state.value}")

engine3.pause_session()
assert engine3.session.state == SessionState.PAUSED
# Try feeding events while paused - should be ignored
pairs_paused = feed_events(engine3, 10, base_ts=50000.0)
print(f"After PAUSE + attempted events: added={pairs_paused}. State={engine3.session.state.value}")
assert pairs_paused == 0, "Events during PAUSE must be discarded!"
count_during_pause = engine3.feature_buffer.event_count
print(f"   (Feature count stayed at {count_during_pause} during PAUSE — correct)")

engine3.resume_session()
assert engine3.session.state == SessionState.CAPTURING
pairs_after = feed_events(engine3, 25, base_ts=70000.0)
print(f"After RESUME + 25 pairs: added={pairs_after}. State={engine3.session.state.value}")
assert count_during_pause < engine3.feature_buffer.event_count, "Should gain events after resume"

res3 = engine3.stop_session()
print(f"After STOP: state={engine3.session.state.value}, pipeline={res3.pipeline_status}")
assert res3.data_quality.is_valid, f"Should have valid quality after 45+ events, got {res3.data_quality.reasons}"
print("   ✓ PASS: Full Pause/Resume/Stop cycle with real events")

# Test Calibration session actually updates baseline count
print("\n" + "=" * 60)
print("STEP 9: Calibration session updates baseline count")
print("=" * 60)

engine4 = BehavioralEngine()
engine4.set_session_type(SessionType.CALIBRATION)
init_baseline_count = len(engine4.user_calibration_history)
print(f"Initial calibration history count: {init_baseline_count}")

engine4.start_session()
feed_events(engine4, 45)
res4 = engine4.stop_session()
after_baseline_count = len(engine4.user_calibration_history)
print(f"After 1 calibration session: baseline count = {after_baseline_count}")
print(f"Baseline result status: {res4.baseline_result.status}")
print(f"Baseline result session_count: {res4.baseline_result.session_count}")

assert after_baseline_count == init_baseline_count + 1, "Calibration session MUST update history count!"
print("   ✓ PASS: Calibration session increments baseline session count")

# Baseline NOT_READY - test Analysis session reports it
print("\n" + "=" * 60)
print("STEP 10: Analysis session with insufficient calibration reports NOT READY")
print("=" * 60)

engine5 = BehavioralEngine()
engine5.set_session_type(SessionType.ANALYSIS)
engine5.start_session()
feed_events(engine5, 50)
res5 = engine5.stop_session()
print(f"Baseline status: {res5.baseline_result.status}")
print(f"Baseline message: {res5.baseline_result.message[:80]}...")
print(f"TDI: {res5.baseline_result.typing_deviation_index}")
# Should be NOT_READY since we only have 0-1 calibration session (less than min 5)
assert res5.baseline_result.status == "NOT_READY", f"Expected NOT_READY, got {res5.baseline_result.status}"
assert res5.baseline_result.typing_deviation_index is None, "TDI must be None when baseline not ready"
print("   ✓ PASS: Analysis without sufficient baseline reports BASELINE NOT READY")

# LSTM gating - should say MODEL_NOT_READY since no artifacts
print("\n" + "=" * 60)
print("STEP 11: LSTM honest gating — MODEL_NOT_READY without artifacts")
print("=" * 60)
print(f"Model status: {res5.model_result.status}")
print(f"Model reason: {res5.model_result.reason[:100]}...")
assert res5.model_result.status == "MODEL_NOT_READY"
assert res5.model_result.predicted_class is None, "Must NOT fabricate predictions!"
print("   ✓ PASS: Model gating correctly reports MODEL_NOT_READY (no fake predictions)")

# RESET test
print("\n" + "=" * 60)
print("STEP RESET: Reset returns to READY and clears buffers")
print("=" * 60)

before_reset_session_id = engine5.session.session_id
before_reset_count = engine5.session.event_count
before_reset_baseline = len(engine5.user_calibration_history)
engine5.reset_session()
after_reset_state = engine5.session.state.value
after_reset_session_id = engine5.session.session_id
after_reset_count = engine5.session.event_count
after_reset_baseline = len(engine5.user_calibration_history)

print(f"Before reset: session={before_reset_session_id[:8]}, events={before_reset_count}, baseline={before_reset_baseline}")
print(f"After  reset: state={after_reset_state}, session={after_reset_session_id[:8]}, events={after_reset_count}, baseline={after_reset_baseline}")

assert after_reset_state == "READY", f"Reset must result in READY state, got {after_reset_state}"
assert before_reset_session_id != after_reset_session_id, "Reset must give NEW session ID!"
assert after_reset_count == 0, f"Reset must clear event count, got {after_reset_count}"
assert after_reset_baseline == before_reset_baseline, "Reset MUST NOT touch historical baseline records!"
print("   ✓ PASS: RESET creates fresh READY state without destroying history")

# Privacy test - raw text / key forbidden
print("\n" + "=" * 60)
print("STEP 24: Privacy test — raw text/key fields cannot pass filter")
print("=" * 60)

engine6 = BehavioralEngine()
engine6.start_session()

bad_payloads = [
    # Has 'text' field
    {"event_id": "bad1", "event_type": "down", "timestamp_ms": 1.0, "key_token": "k_alpha", "text": "hello"},
    # Has 'key' field
    {"event_id": "bad2", "event_type": "down", "timestamp_ms": 2.0, "key_token": "k_alpha", "key": "A"},
    # Has 'value' and 'char'
    {"event_id": "bad3", "event_type": "down", "timestamp_ms": 3.0, "key_token": "k_alpha", "value": "x", "char": "x"},
    # Missing timestamp
    {"event_id": "bad4", "event_type": "down", "key_token": "k_alpha"},
]
added_bad = engine6.ingest_raw_events(bad_payloads, strict_privacy=False)
print(f"Bad payloads added (expected 0): {added_bad}")
assert added_bad == 0, "All bad payloads MUST be rejected!"
print("   ✓ PASS: Privacy filter correctly rejects all bad payloads")

# Verify no forbidden keys anywhere in the result dict
print("\nFinal check: result export contains NO sensitive fields")
from src.live_typing.privacy_filter import FORBIDDEN_PAYLOAD_FIELDS
result_dict = res5.to_dict()

# Search nested recursively
def search_keys(d, forbidden, path=""):
    found = []
    if isinstance(d, dict):
        for k, v in d.items():
            if k in forbidden:
                found.append(f"{path}.{k}")
            found.extend(search_keys(v, forbidden, f"{path}.{k}"))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            found.extend(search_keys(v, forbidden, f"{path}[{i}]"))
    return found

leaks = search_keys(result_dict, FORBIDDEN_PAYLOAD_FIELDS, "result")
if leaks:
    print(f"   FAIL: Found forbidden keys in result: {leaks}")
    assert False, f"Privacy leak! {leaks}"
print("   ✓ PASS: Exported result contains zero forbidden fields")

print("\n" + "=" * 60)
print("ALL CORE FUNCTIONALITY TESTS PASSED")
print("=" * 60)
