"""Phase 2 Isolated Test 3: TensorFlow + Streamlit + PostgreSQL Driver Combination.

Tests the co-existence of all three major runtime components:
- TensorFlow
- Streamlit
- pg8000
- Then exit cleanly.
"""

import sys

print("[TEST 3/3] Testing TensorFlow + Streamlit + PostgreSQL driver co-import...", flush=True)

try:
    import tensorflow as tf
    print(f"[TEST 3/3] TensorFlow imported successfully. Version: {tf.__version__}", flush=True)
except Exception as e:
    print(f"[TEST 3/3] FAILED: TensorFlow import failed: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

try:
    import streamlit as st
    print(f"[TEST 3/3] Streamlit imported successfully. Version: {st.__version__}", flush=True)
except Exception as e:
    print(f"[TEST 3/3] FAILED: Streamlit import failed: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

try:
    import pg8000.dbapi
    print(f"[TEST 3/3] pg8000 imported successfully. Version: {pg8000.__version__}", flush=True)
except Exception as e:
    print(f"[TEST 3/3] FAILED: pg8000 import failed: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

print("[TEST 3/3] SUCCESS: TensorFlow + Streamlit + pg8000 combination is completely stable with zero segfaults.", flush=True)
sys.exit(0)
