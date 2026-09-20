"""Workstation Theme and Skeuomorphic Styling Module.

Injects custom CSS to transform the Streamlit application into a futuristic,
tactile behavioral analysis console with dark graphite textures, recessed instrument
readouts, illuminated status indicators, and physical-feeling interactive elements.

Design Language:
    - Dark industrial / skeuomorphic research workstation
    - Physical depth via shadows, bevels, inset panels, and highlights
    - Monospace readout typography (JetBrains Mono) for instrument values
    - Clean UI typography (Inter) for labels and body text
    - Micro-animations: LED breathing pulse, tactile button press, scanline CRT
    - Palette: #111318 base, #1B1F26 panels, #35D6FF cyan, #8B7CFF violet,
      #45E0A8 positive, #FFB84D warning, #FF667A critical
"""

import streamlit as st

SKEUOMORPHIC_CSS = """
<style>
/* ==========================================================================
   TYPOGRAPHY IMPORTS
   ========================================================================== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ==========================================================================
   WORKSTATION CONSOLE DESIGN TOKENS
   ========================================================================== */
:root {
    --bg-main: #111318;
    --bg-deep: #0C0E12;
    --panel-primary: #1B1F26;
    --panel-recessed: #15181E;
    --panel-border: #282F3A;
    --panel-border-highlight: #3A4454;
    --panel-surface: #20252E;

    --accent-cyan: #35D6FF;
    --accent-violet: #8B7CFF;
    --accent-positive: #45E0A8;
    --accent-warning: #FFB84D;
    --accent-critical: #FF667A;

    --text-primary: #E9EEF5;
    --text-secondary: #8993A4;
    --text-muted: #576071;
    --text-dim: #3D4555;

    --shadow-raised: 0 4px 14px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    --shadow-recessed: inset 0 2px 6px rgba(0, 0, 0, 0.65), 0 1px 0 rgba(255, 255, 255, 0.02);
    --shadow-button: 0 2px 8px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    --shadow-button-active: inset 0 2px 5px rgba(0, 0, 0, 0.6), 0 0 0 rgba(0,0,0,0);

    --font-ui: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --font-mono: 'JetBrains Mono', 'Courier New', Courier, monospace;

    --radius-sm: 4px;
    --radius-md: 6px;
    --radius-lg: 8px;

    --transition-fast: 0.15s ease;
    --transition-std: 0.25s ease;
}

/* ==========================================================================
   KEYFRAME ANIMATIONS
   ========================================================================== */

/* LED breathing pulse */
@keyframes ledPulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}

/* Subtle scanline sweep (CRT instrument effect) */
@keyframes scanlineSweep {
    0% { background-position: 0 0; }
    100% { background-position: 0 100%; }
}

/* Glow pulse for progress bar */
@keyframes progressGlow {
    0%, 100% { box-shadow: 0 0 8px rgba(53, 214, 255, 0.4); }
    50% { box-shadow: 0 0 16px rgba(53, 214, 255, 0.7); }
}

/* ==========================================================================
   BASE APPLICATION SHELL
   ========================================================================== */

.stApp {
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-ui) !important;
}

/* Main content area subtle texture */
.stApp > header {
    background-color: var(--bg-deep) !important;
    border-bottom: 1px solid #1A1E25 !important;
}

/* ==========================================================================
   TYPOGRAPHY OVERRIDES
   ========================================================================== */

h1 {
    color: var(--text-primary) !important;
    font-family: var(--font-ui) !important;
    font-weight: 800 !important;
    letter-spacing: -0.01em !important;
}

h2 {
    color: var(--text-primary) !important;
    font-family: var(--font-ui) !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
}

h3, h4, h5, h6 {
    color: var(--text-primary) !important;
    font-family: var(--font-ui) !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
}

p {
    color: var(--text-secondary) !important;
    font-family: var(--font-ui) !important;
    line-height: 1.6 !important;
}

span, label, .stMarkdown {
    font-family: var(--font-ui) !important;
}

code, .stCode, pre {
    font-family: var(--font-mono) !important;
}

/* ==========================================================================
   SIDEBAR — INSTRUMENT CONTROL PANEL
   ========================================================================== */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13161C 0%, #101318 100%) !important;
    border-right: 1px solid #1E232B !important;
    box-shadow: 3px 0 16px rgba(0, 0, 0, 0.5) !important;
}

section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}

/* Sidebar radio navigation — active item accent */
section[data-testid="stSidebar"] .stRadio > div {
    gap: 2px !important;
}

section[data-testid="stSidebar"] .stRadio label {
    background: transparent !important;
    border-left: 3px solid transparent !important;
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0 !important;
    padding: 7px 12px 7px 10px !important;
    transition: all var(--transition-fast) !important;
    cursor: pointer !important;
}

section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(53, 214, 255, 0.05) !important;
    border-left-color: rgba(53, 214, 255, 0.3) !important;
}

section[data-testid="stSidebar"] .stRadio label[data-checked="true"],
section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: linear-gradient(90deg, rgba(53, 214, 255, 0.1) 0%, transparent 100%) !important;
    border-left-color: var(--accent-cyan) !important;
}

section[data-testid="stSidebar"] .stRadio label span {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    transition: color var(--transition-fast) !important;
}

section[data-testid="stSidebar"] .stRadio label:has(input:checked) span {
    color: var(--text-primary) !important;
}

/* Hide default radio circles in sidebar */
section[data-testid="stSidebar"] .stRadio input[type="radio"] {
    display: none !important;
}

/* Sidebar horizontal rules */
section[data-testid="stSidebar"] hr {
    border: none !important;
    border-top: 1px solid #1E232B !important;
    margin: 12px 0 !important;
}

/* ==========================================================================
   SKEUOMORPHIC CONSOLE CARDS & RECESSED READOUTS
   ========================================================================== */

/* Console Header Card — raised panel with CRT scanline texture */
.console-header-card {
    background: linear-gradient(180deg, #222832 0%, #1B1F26 100%);
    border: 1px solid var(--panel-border);
    border-top: 1px solid var(--panel-border-highlight);
    border-radius: var(--radius-md);
    padding: 18px 24px;
    margin-bottom: 22px;
    box-shadow: var(--shadow-raised);
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: relative;
    overflow: hidden;
}

/* Subtle scanline overlay on console headers */
.console-header-card::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 3px,
        rgba(255, 255, 255, 0.008) 3px,
        rgba(255, 255, 255, 0.008) 4px
    );
    pointer-events: none;
    animation: scanlineSweep 12s linear infinite;
}

.console-title-group {
    display: flex;
    flex-direction: column;
    z-index: 1;
}

.console-title {
    font-family: var(--font-ui) !important;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text-primary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0;
}

.console-subtitle {
    font-family: var(--font-ui);
    font-size: 0.78rem;
    color: var(--text-secondary);
    margin-top: 4px;
    letter-spacing: 0.02em;
}

/* Raised Console Card */
.console-card {
    background: linear-gradient(180deg, #20252E 0%, #1A1E25 100%);
    border: 1px solid var(--panel-border);
    border-top: 1px solid var(--panel-border-highlight);
    border-radius: var(--radius-md);
    padding: 20px 22px;
    margin-bottom: 18px;
    box-shadow: var(--shadow-raised);
}

.console-card-header {
    font-family: var(--font-ui);
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--accent-cyan);
    text-transform: uppercase;
    letter-spacing: 0.09em;
    border-bottom: 1px solid #232A35;
    padding-bottom: 10px;
    margin-bottom: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* Recessed Bay */
.recessed-panel {
    background-color: var(--panel-recessed);
    border: 1px solid #1E232B;
    border-radius: var(--radius-sm);
    padding: 14px 16px;
    box-shadow: var(--shadow-recessed);
    margin-bottom: 12px;
}

/* ==========================================================================
   INSTRUMENT READOUT GRID
   ========================================================================== */

.readout-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 16px;
}

.readout-cell {
    background-color: var(--panel-recessed);
    border: 1px solid #1F252E;
    border-radius: var(--radius-sm);
    padding: 14px;
    text-align: center;
    box-shadow: var(--shadow-recessed);
    border-top: 2px solid #252D38;
    transition: border-color var(--transition-std);
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
    font-family: var(--font-ui);
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}

.readout-value {
    font-family: var(--font-mono);
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.2;
}

/* ==========================================================================
   STATUS LED INDICATORS — WITH BREATHING PULSE
   ========================================================================== */

.led-indicator {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 5px 12px;
    border-radius: 20px;
    background: #15181E;
    border: 1px solid #28303C;
    box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.5);
    z-index: 1;
}

.led-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    animation: ledPulse 2.5s ease-in-out infinite;
}

.led-dot.ready, .led-dot.positive {
    background-color: var(--accent-positive);
    box-shadow: 0 0 8px var(--accent-positive), 0 0 3px var(--accent-positive);
}

.led-dot.cyan {
    background-color: var(--accent-cyan);
    box-shadow: 0 0 8px var(--accent-cyan), 0 0 3px var(--accent-cyan);
}

.led-dot.warning {
    background-color: var(--accent-warning);
    box-shadow: 0 0 8px var(--accent-warning), 0 0 3px var(--accent-warning);
}

.led-dot.critical {
    background-color: var(--accent-critical);
    box-shadow: 0 0 8px var(--accent-critical), 0 0 3px var(--accent-critical);
}

/* ==========================================================================
   DETECTED SIGNAL PILLS
   ========================================================================== */

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
    border-radius: var(--radius-sm);
    font-family: var(--font-mono);
    font-size: 0.75rem;
    font-weight: 600;
    background: linear-gradient(180deg, #1C2129 0%, #161A20 100%);
    border: 1px solid #2C3542;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    color: var(--text-primary);
    transition: all var(--transition-fast);
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

/* ==========================================================================
   PRIVACY GUARD BANNER
   ========================================================================== */

.privacy-guard-banner {
    background: linear-gradient(180deg, #1F1926 0%, #17131D 100%);
    border: 1px solid #63324C;
    border-left: 4px solid var(--accent-critical);
    border-radius: var(--radius-sm);
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

/* ==========================================================================
   STREAMLIT BUTTONS — TACTILE BEVELED CONTROLS
   ========================================================================== */

/* Primary buttons */
.stButton > button {
    background: linear-gradient(180deg, #252C38 0%, #1C222C 100%) !important;
    border: 1px solid var(--panel-border) !important;
    border-top: 1px solid var(--panel-border-highlight) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-ui) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.04em !important;
    padding: 10px 18px !important;
    box-shadow: var(--shadow-button) !important;
    transition: all var(--transition-fast) !important;
    cursor: pointer !important;
    text-transform: uppercase !important;
    position: relative !important;
    overflow: hidden !important;
}

.stButton > button:hover {
    background: linear-gradient(180deg, #2C3442 0%, #222936 100%) !important;
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 3px 12px rgba(53, 214, 255, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.12) !important;
    color: var(--accent-cyan) !important;
    transform: translateY(-1px) !important;
}

.stButton > button:active {
    background: linear-gradient(180deg, #181D24 0%, #1A1F28 100%) !important;
    box-shadow: var(--shadow-button-active) !important;
    transform: translateY(1px) !important;
    border-color: var(--panel-border) !important;
    color: var(--text-secondary) !important;
}

.stButton > button:focus {
    box-shadow: 0 0 0 2px rgba(53, 214, 255, 0.25), var(--shadow-button) !important;
    outline: none !important;
}

/* Download buttons */
.stDownloadButton > button {
    background: linear-gradient(180deg, #1B2534 0%, #151C28 100%) !important;
    border: 1px solid #2A3848 !important;
    border-top: 1px solid #374A5E !important;
    border-radius: var(--radius-sm) !important;
    color: var(--accent-cyan) !important;
    font-family: var(--font-ui) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.04em !important;
    padding: 10px 18px !important;
    box-shadow: var(--shadow-button) !important;
    transition: all var(--transition-fast) !important;
    text-transform: uppercase !important;
}

.stDownloadButton > button:hover {
    background: linear-gradient(180deg, #223040 0%, #1A2636 100%) !important;
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 3px 14px rgba(53, 214, 255, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    transform: translateY(-1px) !important;
}

.stDownloadButton > button:active {
    background: linear-gradient(180deg, #121A24 0%, #141C26 100%) !important;
    box-shadow: var(--shadow-button-active) !important;
    transform: translateY(1px) !important;
}

/* ==========================================================================
   STREAMLIT METRICS — INSTRUMENT READOUT STYLE
   ========================================================================== */

[data-testid="stMetric"] {
    background-color: var(--panel-recessed) !important;
    border: 1px solid #1F252E !important;
    border-top: 2px solid #2A3340 !important;
    border-radius: var(--radius-sm) !important;
    padding: 12px 14px !important;
    box-shadow: var(--shadow-recessed) !important;
    transition: border-top-color var(--transition-std) !important;
}

[data-testid="stMetric"]:hover {
    border-top-color: var(--accent-cyan) !important;
}

[data-testid="stMetric"] label {
    font-family: var(--font-ui) !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--text-secondary) !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important;
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
}

[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
}

/* ==========================================================================
   STREAMLIT PROGRESS BAR — GLOWING INSTRUMENT GAUGE
   ========================================================================== */

.stProgress > div {
    background-color: var(--panel-recessed) !important;
    border: 1px solid #1E232B !important;
    border-radius: 3px !important;
    box-shadow: var(--shadow-recessed) !important;
    height: 10px !important;
    overflow: hidden !important;
}

.stProgress > div > div {
    background: linear-gradient(90deg, #1A8FAA, var(--accent-cyan)) !important;
    border-radius: 2px !important;
    animation: progressGlow 2s ease-in-out infinite !important;
    position: relative !important;
}

/* Scanline texture on progress fill */
.stProgress > div > div::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: repeating-linear-gradient(
        90deg,
        transparent,
        transparent 4px,
        rgba(255, 255, 255, 0.06) 4px,
        rgba(255, 255, 255, 0.06) 5px
    );
}

/* ==========================================================================
   STREAMLIT TABLES & DATAFRAMES — DARK INSTRUMENT PANELS
   ========================================================================== */

/* st.table */
.stTable {
    border-radius: var(--radius-sm) !important;
    overflow: hidden !important;
}

.stTable table {
    background-color: var(--panel-recessed) !important;
    border-collapse: separate !important;
    border-spacing: 0 !important;
    border: 1px solid #1E232B !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: var(--shadow-recessed) !important;
    width: 100% !important;
}

.stTable thead th {
    background: linear-gradient(180deg, #1E242E 0%, #1A1F28 100%) !important;
    color: var(--accent-cyan) !important;
    font-family: var(--font-ui) !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    padding: 10px 14px !important;
    border-bottom: 2px solid #2A3340 !important;
    text-align: left !important;
}

.stTable tbody td {
    background-color: transparent !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    padding: 8px 14px !important;
    border-bottom: 1px solid #1A1F27 !important;
    transition: background-color var(--transition-fast) !important;
}

.stTable tbody tr:nth-child(even) td {
    background-color: rgba(21, 24, 30, 0.5) !important;
}

.stTable tbody tr:hover td {
    background-color: rgba(53, 214, 255, 0.04) !important;
}

/* st.dataframe */
.stDataFrame {
    border-radius: var(--radius-sm) !important;
    border: 1px solid #1E232B !important;
    box-shadow: var(--shadow-recessed) !important;
    overflow: hidden !important;
}

[data-testid="stDataFrame"] {
    background-color: var(--panel-recessed) !important;
    border-radius: var(--radius-sm) !important;
}

/* ==========================================================================
   STREAMLIT ALERTS — INSTRUMENT NOTIFICATIONS
   ========================================================================== */

/* Info box */
.stAlert [data-testid="stNotificationMessage"][aria-label*="info"],
div[data-testid="stAlert"]:has([data-baseweb="notification"][kind="info"]),
.stAlert:has(.st-emotion-cache-info) {
    background-color: rgba(53, 214, 255, 0.06) !important;
    border: 1px solid rgba(53, 214, 255, 0.2) !important;
    border-left: 4px solid var(--accent-cyan) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
}

/* Warning box */
.stAlert [data-testid="stNotificationMessage"][aria-label*="warning"],
div[data-testid="stAlert"]:has([data-baseweb="notification"][kind="warning"]) {
    background-color: rgba(255, 184, 77, 0.06) !important;
    border: 1px solid rgba(255, 184, 77, 0.2) !important;
    border-left: 4px solid var(--accent-warning) !important;
    border-radius: var(--radius-sm) !important;
}

/* Error box */
.stAlert [data-testid="stNotificationMessage"][aria-label*="error"],
div[data-testid="stAlert"]:has([data-baseweb="notification"][kind="negative"]) {
    background-color: rgba(255, 102, 122, 0.06) !important;
    border: 1px solid rgba(255, 102, 122, 0.2) !important;
    border-left: 4px solid var(--accent-critical) !important;
    border-radius: var(--radius-sm) !important;
}

/* Success box */
.stAlert [data-testid="stNotificationMessage"][aria-label*="success"],
div[data-testid="stAlert"]:has([data-baseweb="notification"][kind="positive"]) {
    background-color: rgba(69, 224, 168, 0.06) !important;
    border: 1px solid rgba(69, 224, 168, 0.2) !important;
    border-left: 4px solid var(--accent-positive) !important;
    border-radius: var(--radius-sm) !important;
}

/* General alert text styling */
[data-testid="stAlert"] p,
.stAlert p {
    font-family: var(--font-ui) !important;
    color: var(--text-primary) !important;
}

/* ==========================================================================
   STREAMLIT SELECTBOX & TEXT INPUT — RECESSED INSTRUMENT CONTROLS
   ========================================================================== */

/* Selectbox */
.stSelectbox [data-baseweb="select"] > div {
    background-color: var(--panel-recessed) !important;
    border: 1px solid #1E232B !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: var(--shadow-recessed) !important;
    color: var(--text-primary) !important;
    transition: border-color var(--transition-fast) !important;
}

.stSelectbox [data-baseweb="select"] > div:hover {
    border-color: #364152 !important;
}

.stSelectbox [data-baseweb="select"] > div:focus-within {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 0 2px rgba(53, 214, 255, 0.15), var(--shadow-recessed) !important;
}

/* Selectbox dropdown menu */
[data-baseweb="popover"] {
    border-radius: var(--radius-sm) !important;
    border: 1px solid #2A3340 !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5) !important;
}

[data-baseweb="menu"] {
    background-color: #1A1F28 !important;
}

[data-baseweb="menu"] li {
    background-color: transparent !important;
    color: var(--text-secondary) !important;
    font-family: var(--font-ui) !important;
    transition: all var(--transition-fast) !important;
}

[data-baseweb="menu"] li:hover {
    background-color: rgba(53, 214, 255, 0.08) !important;
    color: var(--text-primary) !important;
}

[data-baseweb="menu"] li[aria-selected="true"] {
    background-color: rgba(53, 214, 255, 0.12) !important;
    color: var(--accent-cyan) !important;
}

/* Text input */
.stTextInput input {
    background-color: var(--panel-recessed) !important;
    border: 1px solid #1E232B !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: var(--shadow-recessed) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    padding: 10px 14px !important;
    transition: border-color var(--transition-fast) !important;
}

.stTextInput input:focus {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 0 2px rgba(53, 214, 255, 0.15), var(--shadow-recessed) !important;
}

/* Text area */
.stTextArea textarea {
    background-color: var(--panel-recessed) !important;
    border: 1px solid #1E232B !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: var(--shadow-recessed) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    transition: border-color var(--transition-fast) !important;
}

.stTextArea textarea:focus {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 0 2px rgba(53, 214, 255, 0.15), var(--shadow-recessed) !important;
}

/* ==========================================================================
   STREAMLIT RADIO — INSTRUMENT SWITCH PANEL (MAIN CONTENT)
   ========================================================================== */

.main .stRadio > label {
    font-family: var(--font-ui) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    color: var(--text-secondary) !important;
}

.main .stRadio > div > label {
    background: linear-gradient(180deg, #1D222B 0%, #181C24 100%) !important;
    border: 1px solid #252D38 !important;
    border-radius: var(--radius-sm) !important;
    padding: 8px 14px !important;
    margin-bottom: 4px !important;
    transition: all var(--transition-fast) !important;
}

.main .stRadio > div > label:hover {
    border-color: #3A4454 !important;
    background: linear-gradient(180deg, #222832 0%, #1C2029 100%) !important;
}

.main .stRadio > div > label:has(input:checked) {
    border-color: var(--accent-cyan) !important;
    background: linear-gradient(180deg, #1C2A34 0%, #172430 100%) !important;
    box-shadow: 0 0 8px rgba(53, 214, 255, 0.1) !important;
}

/* ==========================================================================
   STREAMLIT EXPANDER — COLLAPSIBLE INSTRUMENT BAYS
   ========================================================================== */

.stExpander {
    background: linear-gradient(180deg, #1E242E 0%, #1A1E26 100%) !important;
    border: 1px solid var(--panel-border) !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: var(--shadow-raised) !important;
    overflow: hidden !important;
}

.stExpander > summary,
.stExpander [data-testid="stExpanderToggleDetails"] {
    font-family: var(--font-ui) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    color: var(--accent-cyan) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    padding: 10px 16px !important;
    background: transparent !important;
    border-bottom: 1px solid #232A35 !important;
    transition: all var(--transition-fast) !important;
}

.stExpander > summary:hover,
.stExpander [data-testid="stExpanderToggleDetails"]:hover {
    background: rgba(53, 214, 255, 0.03) !important;
}

/* ==========================================================================
   STREAMLIT TABS — INSTRUMENT PANEL SWITCHER
   ========================================================================== */

.stTabs [data-baseweb="tab-list"] {
    background-color: #14171E !important;
    border-radius: var(--radius-md) !important;
    padding: 4px !important;
    border: 1px solid #222933 !important;
    gap: 4px !important;
    box-shadow: var(--shadow-recessed) !important;
}

.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm) !important;
    padding: 8px 16px !important;
    color: var(--text-secondary) !important;
    font-family: var(--font-ui) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    border: 1px solid transparent !important;
    background: transparent !important;
    transition: all var(--transition-fast) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.03em !important;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(53, 214, 255, 0.05) !important;
    color: var(--text-primary) !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(180deg, #242B36 0%, #1C222B 100%) !important;
    color: var(--accent-cyan) !important;
    border: 1px solid #364152 !important;
    border-top: 2px solid var(--accent-cyan) !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.4) !important;
}

/* Tab highlight bar */
.stTabs [data-baseweb="tab-highlight"] {
    background-color: transparent !important;
}

/* ==========================================================================
   STREAMLIT CODE BLOCKS — TERMINAL READOUT
   ========================================================================== */

.stCodeBlock, .stCode {
    border-radius: var(--radius-sm) !important;
    border: 1px solid #1A1F27 !important;
    box-shadow: var(--shadow-recessed) !important;
}

.stCodeBlock pre, .stCode pre {
    background-color: #0D0F13 !important;
    color: #7AE0A5 !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    line-height: 1.5 !important;
    padding: 16px !important;
    border-radius: var(--radius-sm) !important;
}

/* ==========================================================================
   STREAMLIT MARKDOWN — REFINED ELEMENTS
   ========================================================================== */

/* Markdown tables */
.stMarkdown table {
    background-color: var(--panel-recessed) !important;
    border: 1px solid #1E232B !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: var(--shadow-recessed) !important;
    border-collapse: separate !important;
    border-spacing: 0 !important;
    width: 100% !important;
    overflow: hidden !important;
}

.stMarkdown th {
    background: linear-gradient(180deg, #1E242E 0%, #1A1F28 100%) !important;
    color: var(--accent-cyan) !important;
    font-family: var(--font-ui) !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    padding: 10px 14px !important;
    border-bottom: 2px solid #2A3340 !important;
    text-align: left !important;
}

.stMarkdown td {
    color: var(--text-primary) !important;
    font-family: var(--font-ui) !important;
    font-size: 0.82rem !important;
    padding: 8px 14px !important;
    border-bottom: 1px solid #1A1F27 !important;
}

.stMarkdown tr:nth-child(even) td {
    background-color: rgba(21, 24, 30, 0.5) !important;
}

.stMarkdown tr:hover td {
    background-color: rgba(53, 214, 255, 0.04) !important;
}

/* Inline code in markdown */
.stMarkdown code {
    background-color: #15181E !important;
    color: var(--accent-cyan) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    padding: 2px 6px !important;
    border-radius: 3px !important;
    border: 1px solid #222933 !important;
}

/* Markdown strong text */
.stMarkdown strong {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
}

/* Markdown emphasis */
.stMarkdown em {
    color: var(--text-secondary) !important;
}

/* Markdown horizontal rules */
.stMarkdown hr {
    border: none !important;
    border-top: 1px solid #232A35 !important;
    margin: 18px 0 !important;
}

/* Markdown list items */
.stMarkdown li {
    color: var(--text-secondary) !important;
    font-size: 0.88rem !important;
    line-height: 1.7 !important;
}

/* ==========================================================================
   STREAMLIT CAPTION
   ========================================================================== */

.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--text-muted) !important;
    font-family: var(--font-ui) !important;
    font-size: 0.75rem !important;
    font-style: italic !important;
}

/* ==========================================================================
   CUSTOM SCROLLBAR
   ========================================================================== */

::-webkit-scrollbar {
    width: 7px;
    height: 7px;
}
::-webkit-scrollbar-track {
    background: var(--bg-main);
}
::-webkit-scrollbar-thumb {
    background: #252D38;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #364152;
}

/* ==========================================================================
   CUSTOM HELPER CLASSES (used by dashboard.py helpers)
   ========================================================================== */

/* System masthead banner */
.system-masthead {
    background: linear-gradient(180deg, #1A2030 0%, #141820 100%);
    border: 1px solid #252D3A;
    border-bottom: 2px solid var(--accent-cyan);
    border-radius: 0;
    padding: 16px 20px;
    margin: -16px -16px 20px -16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.system-masthead-title {
    font-family: var(--font-ui);
    font-size: 0.68rem;
    font-weight: 700;
    color: var(--accent-cyan);
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.system-masthead-subtitle {
    font-family: var(--font-ui);
    font-size: 1.1rem;
    font-weight: 800;
    color: var(--text-primary);
    margin-top: 2px;
    letter-spacing: -0.005em;
}

.system-masthead-badge {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    font-weight: 600;
    color: var(--text-muted);
    background: #15181E;
    border: 1px solid #222933;
    border-radius: 3px;
    padding: 3px 8px;
    letter-spacing: 0.05em;
}

/* Metric readout cell (standalone) */
.metric-readout {
    background-color: var(--panel-recessed);
    border: 1px solid #1F252E;
    border-radius: var(--radius-sm);
    padding: 14px 16px;
    text-align: center;
    box-shadow: var(--shadow-recessed);
}

.metric-readout-label {
    font-family: var(--font-ui);
    font-size: 0.68rem;
    font-weight: 700;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}

.metric-readout-value {
    font-family: var(--font-mono);
    font-size: 1.4rem;
    font-weight: 700;
    line-height: 1.2;
}

.metric-readout-unit {
    font-family: var(--font-ui);
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-muted);
    margin-top: 2px;
}

/* Status badge inline */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 3px;
    background: #15181E;
    border: 1px solid #222933;
    box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.4);
}

.status-badge.ready {
    border-color: rgba(69, 224, 168, 0.3);
    color: var(--accent-positive);
}

.status-badge.warning {
    border-color: rgba(255, 184, 77, 0.3);
    color: var(--accent-warning);
}

.status-badge.critical {
    border-color: rgba(255, 102, 122, 0.3);
    color: var(--accent-critical);
}

.status-badge.cyan {
    border-color: rgba(53, 214, 255, 0.3);
    color: var(--accent-cyan);
}

/* Section divider with label */
.section-divider {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 14px 0 8px 0;
}

.section-divider-line {
    flex: 1;
    height: 1px;
    background: #1E232B;
}

.section-divider-label {
    font-family: var(--font-ui);
    font-size: 0.65rem;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    white-space: nowrap;
}

/* ==========================================================================
   PLOTLY CHART CONTAINER REFINEMENT
   ========================================================================== */

[data-testid="stPlotlyChart"] {
    border-radius: var(--radius-sm) !important;
    overflow: hidden !important;
}

/* ==========================================================================
   RESPONSIVE REFINEMENTS
   ========================================================================== */

@media (max-width: 768px) {
    .readout-grid {
        grid-template-columns: repeat(2, 1fr);
    }

    .console-header-card {
        flex-direction: column;
        align-items: flex-start;
        gap: 10px;
    }
}

</style>
"""


def apply_workstation_theme() -> None:
    """Inject the permanent skeuomorphic workstation CSS into Streamlit."""
    st.markdown(SKEUOMORPHIC_CSS, unsafe_allow_html=True)
