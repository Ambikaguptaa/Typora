"""Workstation Theme and Skeuomorphic Styling Module.

Injects custom CSS to transform the Streamlit application into a futuristic,
tactile behavioral analysis console with dark graphite textures, recessed instrument
readouts, and illuminated status indicators.
"""

import streamlit as st

SKEUOMORPHIC_CSS = """
<style>
/* ==========================================================================
   WORKSTATION CONSOLE DESIGN TOKENS
   ========================================================================== */
:root {
    --bg-main: #111318;
    --panel-primary: #1B1F26;
    --panel-recessed: #15181E;
    --panel-border: #282F3A;
    --panel-border-highlight: #3A4454;
    
    --accent-cyan: #35D6FF;
    --accent-violet: #8B7CFF;
    --accent-positive: #45E0A8;
    --accent-warning: #FFB84D;
    --accent-critical: #FF667A;
    
    --text-primary: #E9EEF5;
    --text-secondary: #8993A4;
    --text-muted: #576071;
    
    --shadow-raised: 0 4px 14px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    --shadow-recessed: inset 0 2px 6px rgba(0, 0, 0, 0.65), 0 1px 0 rgba(255, 255, 255, 0.02);
}

/* Base App Background */
.stApp {
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #13161C !important;
    border-right: 1px solid #202630 !important;
    box-shadow: 2px 0 10px rgba(0,0,0,0.3) !important;
}

/* Typography Overrides */
h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary) !important;
    letter-spacing: 0.02em;
}

p, span, label {
    color: var(--text-secondary) !important;
}

/* ==========================================================================
   SKEUOMORPHIC CONSOLE CARDS & RECESSED READOUTS
   ========================================================================== */

/* Raised Console Header */
.console-header-card {
    background: linear-gradient(180deg, #222832 0%, #1B1F26 100%);
    border: 1px solid var(--panel-border);
    border-top: 1px solid var(--panel-border-highlight);
    border-radius: 6px;
    padding: 16px 22px;
    margin-bottom: 20px;
    box-shadow: var(--shadow-raised);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.console-title-group {
    display: flex;
    flex-direction: column;
}

.console-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text-primary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0;
}

.console-subtitle {
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-top: 3px;
}

/* Raised Panel */
.console-card {
    background: linear-gradient(180deg, #20252E 0%, #1A1E25 100%);
    border: 1px solid var(--panel-border);
    border-top: 1px solid var(--panel-border-highlight);
    border-radius: 6px;
    padding: 18px 20px;
    margin-bottom: 16px;
    box-shadow: var(--shadow-raised);
}

.console-card-header {
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--accent-cyan);
    text-transform: uppercase;
    letter-spacing: 0.09em;
    border-bottom: 1px solid #232A35;
    padding-bottom: 8px;
    margin-bottom: 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* Recessed Bay */
.recessed-panel {
    background-color: var(--panel-recessed);
    border: 1px solid #1E232B;
    border-radius: 4px;
    padding: 14px 16px;
    box-shadow: var(--shadow-recessed);
    margin-bottom: 12px;
}

/* Digital Readout Display */
.readout-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 16px;
}

.readout-cell {
    background-color: var(--panel-recessed);
    border: 1px solid #1F252E;
    border-radius: 4px;
    padding: 12px;
    text-align: center;
    box-shadow: var(--shadow-recessed);
    border-top: 2px solid #252D38;
}

.readout-cell.active-cyan {
    border-top: 2px solid var(--accent-cyan);
}

.readout-cell.active-violet {
    border-top: 2px solid var(--accent-violet);
}

.readout-cell.active-green {
    border-top: 2px solid var(--accent-positive);
}

.readout-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}

.readout-value {
    font-family: "Courier New", Courier, monospace, monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.2;
}

/* Status LEDs */
.led-indicator {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 4px 10px;
    border-radius: 20px;
    background: #15181E;
    border: 1px solid #28303C;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.5);
}

.led-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
}

.led-dot.ready, .led-dot.positive {
    background-color: var(--accent-positive);
    box-shadow: 0 0 8px var(--accent-positive);
}

.led-dot.cyan {
    background-color: var(--accent-cyan);
    box-shadow: 0 0 8px var(--accent-cyan);
}

.led-dot.warning {
    background-color: var(--accent-warning);
    box-shadow: 0 0 8px var(--accent-warning);
}

.led-dot.critical {
    background-color: var(--accent-critical);
    box-shadow: 0 0 8px var(--accent-critical);
}

/* Detected Signal Pills */
.signals-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 8px;
}

.signal-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 4px;
    font-size: 0.78rem;
    font-weight: 600;
    background: linear-gradient(180deg, #1C2129 0%, #161A20 100%);
    border: 1px solid #2C3542;
    box-shadow: 0 2px 4px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05);
    color: var(--text-primary);
}

.signal-pill.detected {
    border-color: rgba(53, 214, 255, 0.4);
    color: var(--accent-cyan);
}

.signal-pill.missing {
    border-color: #2D333D;
    color: var(--text-muted);
    opacity: 0.7;
}

/* Privacy Guard Banner */
.privacy-guard-banner {
    background: linear-gradient(180deg, #1F1926 0%, #17131D 100%);
    border: 1px solid #63324C;
    border-left: 4px solid var(--accent-critical);
    border-radius: 4px;
    padding: 12px 18px;
    margin-bottom: 16px;
    box-shadow: var(--shadow-raised);
    display: flex;
    align-items: center;
    gap: 12px;
}

.privacy-guard-text {
    font-size: 0.85rem;
    color: #F8D7DA;
    line-height: 1.4;
}

.privacy-guard-highlight {
    font-weight: 700;
    color: var(--accent-critical);
    text-transform: uppercase;
}

/* Streamlit Tabs Skeuomorphic Override */
.stTabs [data-baseweb="tab-list"] {
    background-color: #14171E !important;
    border-radius: 6px !important;
    padding: 4px !important;
    border: 1px solid #222933 !important;
    gap: 4px !important;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 4px !important;
    padding: 8px 16px !important;
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    border: 1px solid transparent !important;
    background: transparent !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(180deg, #242B36 0%, #1C222B 100%) !important;
    color: var(--accent-cyan) !important;
    border: 1px solid #364152 !important;
    border-top: 2px solid var(--accent-cyan) !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.4) !important;
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 7px;
    height: 7px;
}
::-webkit-scrollbar-track {
    background: #111318;
}
::-webkit-scrollbar-thumb {
    background: #252D38;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #364152;
}
</style>
"""


def apply_workstation_theme() -> None:
    """Inject the permanent skeuomorphic workstation CSS into Streamlit."""
    st.markdown(SKEUOMORPHIC_CSS, unsafe_allow_html=True)
