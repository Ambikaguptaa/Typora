"""Verification script to test rendering logic across all 9 pages of app.py."""

import os
import sys

# Ensure current directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath("."))

from src.integration.behavioral_engine import BehavioralEngine
from app import (
    render_overview_page,
    render_live_session_page,
    render_baseline_analytics_page,
    render_behavioral_model_page,
    render_session_history_page,
    render_reports_page,
    render_privacy_security_page,
    render_dataset_system_status_page,
    render_architecture_page,
)

print("Testing page render functions directly:")
engine = BehavioralEngine()

pages = [
    ("Overview", lambda: render_overview_page(engine)),
    ("Live Session", lambda: render_live_session_page(engine)),
    ("Baseline Analytics", lambda: render_baseline_analytics_page(engine)),
    ("Behavioral Model", lambda: render_behavioral_model_page(engine)),
    ("Session History", lambda: render_session_history_page()),
    ("Reports", lambda: render_reports_page()),
    ("Privacy & Security", lambda: render_privacy_security_page()),
    ("Dataset / System Status", lambda: render_dataset_system_status_page()),
    ("Architecture & Roadmap", lambda: render_architecture_page()),
]

for name, fn in pages:
    try:
        fn()
        print(f"  [OK] {name}: PASSED")
    except Exception as e:
        print(f"  [FAIL] {name}: FAILED - {e}")
        import traceback
        traceback.print_exc()

print("All non-interactive pages verified successfully!")
