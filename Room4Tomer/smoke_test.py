"""
Headless smoke test for Room 4's Streamlit app, using Streamlit's official
AppTest harness. Mirrors Room1Tomer/smoke_test.py's structure.

Run with:  python smoke_test.py
"""

import sys
from streamlit.testing.v1 import AppTest


def show(at, label):
    print(f"\n=== {label} ===")
    print(f"  exceptions: {len(at.exception)}")
    for e in at.exception:
        print("  !! ", e.value)
        if getattr(e, "stack_trace", None):
            print("".join(e.stack_trace[-12:]))
    print(f"  warnings  : {[w.value for w in at.warning]}")
    return len(at.exception) == 0


def main():
    ok = True

    # ---- 1. app boots -------------------------------------------------------
    at = AppTest.from_file("app.py", default_timeout=180)
    at.run()
    ok &= show(at, "1. initial load")

    # ---- 2. sidebar changes propagate ---------------------------------------
    at.session_state.wind_scale = 2.0
    at.session_state.preset = "Wind Corridor"
    at.run()
    ok &= show(at, "2. sidebar changes (preset + wind scale)")
    print("  preset:", at.session_state.preset, "| wind_scale:", at.session_state.wind_scale)

    # ---- 3. train with small overrides (fast) -------------------------------
    at.session_state.episodes = 40
    at.session_state.max_steps = 60
    at.session_state.hidden = 16
    at.session_state.buffer_capacity = 2000
    at.run()

    train = [b for b in at.button if "Train" in (b.label or "")]
    if not train:
        print("  !! Train button not found")
        ok = False
    else:
        train[0].click().run(timeout=180)
        ok &= show(at, "3. train (15 episodes, small network)")
        res = at.session_state.results.get(4)
        if res is None:
            print("  !! results[4] missing after training")
            ok = False
        else:
            print("  reward_history length:", len(res["info"]["reward_history"]))
            print("  before frames:", len(res["before"]["frames"]), "| after frames:", len(res["after"]["frames"]))
            print("  before solved/crashed:", res["before"]["solved"], res["before"]["crashed"])
            print("  after  solved/crashed:", res["after"]["solved"], res["after"]["crashed"])
            ok &= len(res["info"]["reward_history"]) == 40
            ok &= len(res["before"]["frames"]) > 1
            ok &= len(res["after"]["frames"]) > 1

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES FOUND"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
