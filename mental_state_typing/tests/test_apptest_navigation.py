"""Automated headless testing using Streamlit's official AppTest framework.

DO NOT USE BROWSER TESTING.
Validates:
- Script starts cleanly with no exception
- Navigation across all 9 pages
- First-run behavior with empty session_state
- Immunity to 'selected_nav' NameError
- Live session button state machine transitions (READY -> START -> PAUSE -> RESUME -> STOP -> RESET)
- Repeated reruns stability
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_starts_cleanly():
    """Verify app.py loads and executes without unhandled exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=25)
    at.run()
    assert len(at.exception) == 0, f"Exception occurred on startup: {at.exception}"
    assert "engine" in at.session_state


def test_empty_session_state_first_run():
    """Verify first run with completely empty session_state does not trigger NameError or crash."""
    at = AppTest.from_file(APP_PATH, default_timeout=25)
    # Ensure fresh start without manually clearing internal proxy
    at.run()
    assert len(at.exception) == 0
    assert at.session_state.get("selected_nav") is not None


def test_navigation_across_all_pages():
    """Verify selecting each available page renders without exception."""
    nav_options = [
        "1. 📋 Overview",
        "2. ⌨️ Live Session",
        "3. 📊 Baseline Analytics",
        "4. 🧠 Behavioral Model",
        "5. 📜 Session History",
        "6. 📑 Reports",
        "7. 🔒 Privacy & Security",
        "8. 🔬 Dataset / System Status",
        "9. 🧩 Architecture & Roadmap",
    ]

    at = AppTest.from_file(APP_PATH, default_timeout=25)
    at.run()
    assert len(at.exception) == 0

    for option in nav_options:
        # Select navigation option using set_value and rerun
        at.sidebar.radio(key="navigation_radio").set_value(option).run()
        assert len(at.exception) == 0, f"Error rendering page: {option}. Exceptions: {at.exception}"
        assert at.session_state["selected_nav"] == option


def test_live_session_deck_state_machine_and_buttons():
    """Verify live session buttons follow strict authoritative state machine via AppTest."""
    at = AppTest.from_file(APP_PATH, default_timeout=25)
    at.run()

    # Navigate to Live Session
    at.sidebar.radio(key="navigation_radio").set_value("2. ⌨️ Live Session").run()
    assert len(at.exception) == 0

    # 1. State: READY
    engine = at.session_state.get("engine")
    assert engine is not None
    assert engine.session.state.value == "READY"

    btn_start = at.button(key="live_btn_start")
    btn_pause = at.button(key="live_btn_pause")
    btn_resume = at.button(key="live_btn_resume")
    btn_stop = at.button(key="live_btn_stop")

    assert not btn_start.disabled
    assert btn_pause.disabled
    assert btn_resume.disabled
    assert btn_stop.disabled

    # 2. Click START -> State: CAPTURING
    at.button(key="live_btn_start").click().run()
    assert len(at.exception) == 0
    assert engine.session.state.value == "CAPTURING"

    btn_start = at.button(key="live_btn_start")
    btn_pause = at.button(key="live_btn_pause")
    btn_resume = at.button(key="live_btn_resume")
    btn_stop = at.button(key="live_btn_stop")

    assert btn_start.disabled
    assert not btn_pause.disabled
    assert btn_resume.disabled
    assert not btn_stop.disabled

    # 3. Click PAUSE -> State: PAUSED
    at.button(key="live_btn_pause").click().run()
    assert len(at.exception) == 0
    assert engine.session.state.value == "PAUSED"

    btn_start = at.button(key="live_btn_start")
    btn_pause = at.button(key="live_btn_pause")
    btn_resume = at.button(key="live_btn_resume")
    btn_stop = at.button(key="live_btn_stop")

    assert btn_start.disabled
    assert btn_pause.disabled
    assert not btn_resume.disabled
    assert not btn_stop.disabled

    # 4. Click RESUME -> State: CAPTURING
    at.button(key="live_btn_resume").click().run()
    assert len(at.exception) == 0
    assert engine.session.state.value == "CAPTURING"

    # 5. Click STOP -> State: COMPLETED
    at.button(key="live_btn_stop").click().run()
    assert len(at.exception) == 0
    assert engine.session.state.value == "COMPLETED"

    # 6. Click RESET -> State: READY
    at.button(key="live_btn_reset").click().run()
    assert len(at.exception) == 0
    assert engine.session.state.value == "READY"


def test_repeated_reruns_stability():
    """Verify repeated app reruns maintain state and do not crash."""
    at = AppTest.from_file(APP_PATH, default_timeout=25)
    at.run()
    assert len(at.exception) == 0

    for _ in range(3):
        at.run()
        assert len(at.exception) == 0
