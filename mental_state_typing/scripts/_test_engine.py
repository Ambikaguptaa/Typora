import sys, json
sys.path.insert(0, '.')

from src.integration import BehavioralEngine, SessionType

print("Testing BehavioralEngine import and init...")
engine = BehavioralEngine()
print(f"Engine created. session_type={engine.session_type.value}")
print(f"Session state: {engine.session.state.value}")
assert engine.session.state.value == "READY"
print("PASS: engine initializes with READY state")

print("\nTesting START SESSION...")
sid = engine.start_session()
print(f"Started session: {sid[:12]}...")
print(f"Session state: {engine.session.state.value}")
print(f"Pipeline state: {engine.state.value}")
assert engine.session.state.value == "CAPTURING"
print("PASS: START -> CAPTURING state")

print("\nIngesting 40 realistic event pairs...")
base_ts = 10000.0
events = []
for i in range(40):
    p = base_ts + i * 195.0 + (i % 4) * 15.0
    r = p + 92.0 + (i % 6) * 22.0
    is_space = (i + 1) % 8 == 0
    is_bk = i == 20 or i == 21
    tok = "k_space" if is_space else ("k_backspace" if is_bk else "k_alpha")
    events.append({"event_id": f"d{i}", "event_type": "down", "timestamp_ms": p, "key_token": tok, "is_backspace": is_bk, "is_enter": False, "is_space": is_space})
    events.append({"event_id": f"u{i}", "event_type": "up", "timestamp_ms": r, "key_token": tok, "is_backspace": is_bk, "is_enter": False, "is_space": is_space})

added = engine.ingest_raw_events(events, strict_privacy=False)
print(f"Added {added} paired events")
print(f"Session raw events: {engine.session.event_count}")
print(f"Feature buffer count: {engine.feature_buffer.event_count}")
ec = engine.session.event_count
fb_count = engine.feature_buffer.event_count
assert added > 25, f"Expected >25 paired, got {added}"
assert ec >= 70, f"Expected >=70 raw events, got {ec}"
assert fb_count == added, f"Expected {added} in buffer, got {fb_count}"
print("PASS: event ingestion through engine works")

print("\nTesting PAUSE...")
engine.pause_session()
print(f"State after PAUSE: {engine.session.state.value}")
assert engine.session.state.value == "PAUSED"
events_pause = [{"event_id": "pd1", "event_type": "down", "timestamp_ms": 999999, "key_token": "k_alpha", "is_backspace": False, "is_enter": False, "is_space": False}]
paused_added = engine.ingest_raw_events(events_pause, strict_privacy=False)
print(f"Events added while PAUSED (should be 0): {paused_added}")
assert paused_added == 0, f"Expected 0 events during pause, got {paused_added}"
print("PASS: PAUSE state works - events are discarded")

print("\nTesting RESUME...")
engine.resume_session()
print(f"State after RESUME: {engine.session.state.value}")
assert engine.session.state.value == "CAPTURING"
print(f"Count preserved after resume: {engine.feature_buffer.event_count}")
assert engine.feature_buffer.event_count == fb_count, "Lost events after resume!"
print("PASS: RESUME preserves existing data")

print("\nTesting STOP & VALIDATE (Calibration)...")
engine.set_session_type(SessionType.CALIBRATION)
result = engine.stop_session()
print(f"Result type: {type(result).__name__}")
print(f"Session ID: {result.session_id[:12]}...")
print(f"Pipeline status: {result.pipeline_status}")
print(f"Quality verdict: {result.data_quality.verdict}")
print(f"Quality is_valid: {result.data_quality.is_valid}")
print(f"Feature rows: {result.feature_summary.event_count}")
print(f"Baseline status: {result.baseline_result.status}")
print(f"Baseline count: {result.baseline_result.session_count}")
print(f"Model status: {result.model_result.status}")
dqm = result.data_quality.metrics
print(f"Quality metrics keys: {list(dqm.keys())}")
assert result.session_id == sid, "Session ID mismatch!"
print(f"Session state after stop: {engine.session.state.value}")
print("PASS: STOP produces valid assessment result")

print("\nTesting privacy: raw text forbidden...")
from src.live_typing.privacy_filter import sanitize_event_batch, PrivacyViolationError
bad_payload = [{"event_id": "1", "event_type": "down", "timestamp_ms": 1.0, "key_token": "k_alpha", "text": "hello world", "key": "a"}]
clean, rej = sanitize_event_batch(bad_payload, strict_raise=False)
print(f"Rejected bad payload: rejected={rej}")
assert rej == 1, f"Expected 1 rejection, got {rej}"
assert len(clean) == 0, f"Expected 0 clean events, got {len(clean)}"
print("PASS: privacy filter rejects payloads with text/key fields")

print("\nTesting RESET...")
engine.reset_session()
print(f"State after RESET: {engine.session.state.value}")
assert engine.session.state.value == "READY"
assert engine.session.event_count == 0, "Events not cleared after reset!"
assert engine.feature_buffer.event_count == 0, "Feature buffer not cleared after reset!"
print("PASS: RESET returns to READY with fresh counters")

print("\n===== ALL BEHAVIORAL ENGINE TESTS PASSED =====")
