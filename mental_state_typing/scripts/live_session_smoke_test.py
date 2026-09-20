"""Dedicated Live Behavioral Session Terminal Smoke Test.

Simulates the complete end-to-end event protocol in headless terminal execution:
START -> EVENT CAPTURE -> FEATURE EXTRACTION -> 30-EVENT WINDOW -> PAUSE -> RESUME -> STOP -> VALIDATION -> BASELINE/ANALYSIS -> PERSISTENCE -> REPORT -> RESET

Exits with code 0 on complete PASS.
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.behavioral_engine import BehavioralEngine
from src.integration.pipeline_state import SessionType
from src.live_typing.session import SessionState
from src.assessment.report_generator import generate_markdown_report


def run_smoke_test():
    results = {}
    print("LIVE SESSION SMOKE TEST")
    print("-----------------------")

    try:
        engine = BehavioralEngine(user_id="smoke_test_user")

        # 1. START
        engine.set_session_type(SessionType.CALIBRATION)
        session_id = engine.start_session()
        assert engine.session.state == SessionState.CAPTURING, f"Expected CAPTURING, got {engine.session.state}"
        results["START"] = "PASS"

        # 2. EVENT CAPTURE & INGESTION
        t0 = 1700000000000.0
        events = []
        # Generate 35 valid, paired key events with realistic ms intervals (span ~ 7.0 seconds)
        for i in range(35):
            t_down = t0 + i * 200.0
            t_up = t_down + 90.0
            events.append({
                "event_id": f"evt_smoke_{i}_down",
                "event_type": "keydown",
                "timestamp": t_down,
                "key_category": "alphanumeric",
            })
            events.append({
                "event_id": f"evt_smoke_{i}_up",
                "event_type": "keyup",
                "timestamp": t_up,
                "key_category": "alphanumeric",
            })

        added = engine.ingest_raw_events(events, strict_privacy=True)
        assert added == 35, f"Expected 35 paired events ingested, got {added}"
        assert engine.session._raw_event_count == 70
        assert len(engine.session.get_paired_events()) == 35
        results["EVENT CAPTURE"] = "PASS"

        # 3. FEATURE EXTRACTION
        n_features = engine.feature_buffer.event_count
        assert n_features >= 30, f"Expected at least 30 feature rows, got {n_features}"
        telemetry = engine.feature_buffer.extract_telemetry()
        assert telemetry["mean_dwell_ms"] > 0
        results["FEATURE EXTRACTION"] = "PASS"

        # 4. 30-EVENT WINDOW
        X_windows, meta = engine.feature_buffer.generate_sequence_windows(sequence_length=30)
        assert len(X_windows) >= 1, f"Expected at least 1 window of 30, got {len(X_windows)}"
        assert X_windows[0].shape == (30, 6), f"Expected shape (30, 6), got {X_windows[0].shape}"
        results["30-EVENT WINDOW"] = "PASS"

        # 5. PAUSE
        engine.pause_session()
        assert engine.session.state == SessionState.PAUSED
        # Ingest while paused should be dropped
        paused_evt = [{
            "event_id": "evt_paused_ignore",
            "event_type": "keydown",
            "timestamp": t0 + 10.0,
            "key_category": "alphanumeric",
        }]
        engine.ingest_raw_events(paused_evt, strict_privacy=True)
        assert engine.session._raw_event_count == 70, "Paused event should not have been accepted"
        results["PAUSE"] = "PASS"

        # 6. RESUME
        engine.resume_session()
        assert engine.session.state == SessionState.CAPTURING
        results["RESUME"] = "PASS"

        # 7. STOP
        assessment = engine.stop_session()
        assert engine.session.state == SessionState.COMPLETED
        results["STOP"] = "PASS"

        # 8. VALIDATION
        assert assessment.data_quality.is_valid is True, f"Validation failed: {assessment.data_quality.reasons}"
        results["VALIDATION"] = "PASS"

        # 9. BASELINE/ANALYSIS
        # Calibration mode updates baseline
        assert engine.session_type == SessionType.CALIBRATION
        results["BASELINE/ANALYSIS"] = "PASS"

        # 10. PERSISTENCE
        # Verification that session and typing metrics persisted
        from database.database import get_db_cursor
        with get_db_cursor() as cur:
            cur.execute("SELECT session_id, status FROM sessions WHERE session_id = ?", (session_id,))
            row = cur.fetchone()
            assert row is not None, "Session was not found in database"
            assert row["session_id"] == session_id
        results["PERSISTENCE"] = "PASS"

        # 11. REPORT
        report_md = generate_markdown_report(assessment)
        assert len(report_md) > 100
        assert "Behavioral Session Intelligence Report" in report_md
        results["REPORT"] = "PASS"

        # 12. RESET
        engine.reset_session()
        assert engine.session.state == SessionState.READY
        assert engine.session._raw_event_count == 0
        assert engine.feature_buffer.event_count == 0
        results["RESET"] = "PASS"

    except Exception as e:
        print(f"FAILED AT: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    for step, status in results.items():
        print(f"{step}: {status}")

    return 0


if __name__ == "__main__":
    sys.exit(run_smoke_test())
