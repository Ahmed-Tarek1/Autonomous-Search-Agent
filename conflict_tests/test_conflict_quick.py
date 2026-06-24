"""
conflict_tests/test_conflict_quick.py
======================================
Quick smoke test for P4 conflict detector.
Tests the import and basic flow WITHOUT needing an API key.
Passes when the agent gracefully handles the no-API-key case.

Run from repo root:
    C:/Users/20121/anaconda3/python.exe conflict_tests/test_conflict_quick.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from agents.conflict import detect_conflicts, route_on_conflict
    from state import mock_state
    print("[OK] Imports successful")

    s = mock_state()
    r = detect_conflicts(s)
    print("[OK] detect_conflicts() ran successfully")
    print("[OK] conflict_report:", r["conflict_report"])
    print("[OK] Route ->", route_on_conflict(r))
    print("[OK] Trace:", r["reasoning_trace"][-1])
    print("\n=== SMOKE TEST PASSED ===")

except RuntimeError as e:
    # Expected if GEMINI_API_KEY is set but invalid
    print(f"[INFO] RuntimeError (likely API key issue): {e}")
    print("=== Smoke test skipped - set GEMINI_API_KEY in .env to run live test ===")

except Exception as e:
    print(f"[FAIL] {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
