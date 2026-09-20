"""Phase 2 Isolated Test 2: Streamlit + PostgreSQL Driver Combination.

Tests the co-existence of Streamlit and the PostgreSQL driver:
- Streamlit
- pg8000
- Then exit cleanly.
"""

import sys

print("[TEST 2/3] Testing Streamlit + PostgreSQL driver co-import...", flush=True)

try:
    import streamlit as st
    print(f"[TEST 2/3] Streamlit imported successfully. Version: {st.__version__}", flush=True)
except Exception as e:
    print(f"[TEST 2/3] FAILED: Streamlit import failed: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

try:
    import pg8000.dbapi
    print(f"[TEST 2/3] pg8000 imported successfully alongside Streamlit. Version: {pg8000.__version__}", flush=True)
except Exception as e:
    print(f"[TEST 2/3] FAILED: pg8000 import failed: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

print("[TEST 2/3] SUCCESS: Streamlit + PostgreSQL driver combination is stable with zero segfaults.", flush=True)
sys.exit(0)
