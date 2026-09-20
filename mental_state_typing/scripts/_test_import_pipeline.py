import sys, math
sys.path.insert(0, '.')

from src.live_typing.session import LiveTypingSession, SessionState
from src.live_typing.feature_buffer import LiveFeatureBuffer, CANONICAL_SEQUENCE_FEATURES

print(f"Canonical features: {CANONICAL_SEQUENCE_FEATURES}")
assert CANONICAL_SEQUENCE_FEATURES == ['dwell_time', 'flight_time', 'pause_duration', 'typing_speed', 'backspace', 'error_flag']
print("PASS: feature schema matches expected (6 features)")

s = LiveTypingSession()
s.start()
assert s.state == SessionState.CAPTURING
assert s.event_count == 0

base_ts = 1000.0
events = []
n_pairs = 35
for i in range(n_pairs):
    press_ts = base_ts + i * 200.0 + (i % 3) * 10.0
    rel_ts = press_ts + 80.0 + (i % 5) * 20.0
    events.append({
        "event_id": f"ev_down_{i}",
        "event_type": "down",
        "timestamp_ms": press_ts,
        "key_token": "k_alpha",
        "is_backspace": False,
        "is_enter": False,
        "is_space": i % 7 == 0,
    })
    events.append({
        "event_id": f"ev_up_{i}",
        "event_type": "up",
        "timestamp_ms": rel_ts,
        "key_token": "k_alpha",
        "is_backspace": False,
        "is_enter": False,
        "is_space": i % 7 == 0,
    })

events.append({"event_id": "bk_dn", "event_type": "down", "timestamp_ms": base_ts + 35*200 + 50, "key_token": "k_backspace", "is_backspace": True, "is_enter": False, "is_space": False})
events.append({"event_id": "bk_up", "event_type": "up", "timestamp_ms": base_ts + 35*200 + 140, "key_token": "k_backspace", "is_backspace": True, "is_enter": False, "is_space": False})

added = s.ingest_browser_batch(events, strict_privacy=False)
print(f"Added {added} paired events")
assert added > 25, f"Expected > 25 paired events, got {added}"
assert s.event_count > 50, f"Expected > 50 raw events, got {s.event_count}"

paired = s.get_paired_events()
print(f"Paired events: {len(paired)}")
assert len(paired) == added

fb = LiveFeatureBuffer()
fb.add_events(paired)
print(f"Feature buffer events: {fb.event_count}")
assert fb.event_count == added

df = fb.to_dataframe()
print(f"DataFrame shape: {df.shape}")

telemetry = fb.extract_telemetry()
md = telemetry["mean_dwell_ms"]
mf = telemetry["mean_flight_ms"]
wpm = telemetry["estimated_wpm"]
print(f"Telemetry: mean_dwell={md:.1f}ms, mean_flight={mf:.1f}ms, wpm={wpm:.1f}")
assert telemetry["event_count"] == added

X_seq, meta = fb.generate_sequence_windows(sequence_length=30, sequence_stride=10)
print(f"Sequence windows: X shape = {X_seq.shape}")
if len(X_seq) > 0:
    assert X_seq.shape[1] == 30, f"Expected seq_len=30, got {X_seq.shape[1]}"
    assert X_seq.shape[2] == 6, f"Expected 6 features, got {X_seq.shape[2]}"
    import numpy as np
    assert not np.isnan(X_seq).any(), "NaN values found in sequence"
    assert not np.isinf(X_seq).any(), "Inf values found in sequence"
    print("PASS: sequence windows are valid (30x6, finite)")

print("PASS: end-to-end event ingestion and feature pipeline works")
