"""
Headless smoke test for the Streamlit app, using Streamlit's official AppTest
harness. It really executes app.py (same code path as the browser), so any
exception, bad widget key or crash shows up here before we open a browser.

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
    print(f"  buttons   : {len(at.button)}   metrics: {len(at.metric)}")
    return len(at.exception) == 0


def main():
    ok = True

    # ---- 1. app boots -----------------------------------------------------
    at = AppTest.from_file("app.py", default_timeout=120)
    at.run()
    ok &= show(at, "1. initial load (Room 1)")

    # ---- 2. paint a cell (grid editor buttons) ----------------------------
    at.session_state.paint_tool = "wall"
    at.run()
    cell = [b for b in at.button if b.key == "cell_3_3"]
    if not cell:
        print("  !! grid button cell_3_3 not found")
        ok = False
    else:
        cell[0].click().run()
        ok &= show(at, "2. paint wall at (3,3)")
        print("  grid_cells:", at.session_state.grid_cells.get((3, 3)))

    # ---- 3. train Room 1 (DP) --------------------------------------------
    train = [b for b in at.button if "Train" in (b.label or "")]
    if not train:
        print("  !! Train button not found")
        ok = False
    else:
        train[0].click().run()
        ok &= show(at, "3. train Room 1 (DP)")
        res = at.session_state.results.get(1)
        print("  solved:", res and res["solved"], "| steps:", res and len(res["path"]) - 1)
        print("  unlocked_room:", at.session_state.unlocked_room)

    # ---- 4. navigate to Room 2 -------------------------------------------
    nav2 = [b for b in at.button if b.key == "room_nav_2"]
    if not nav2:
        print("  !! Room 2 nav button not found")
        ok = False
    else:
        nav2[0].click().run()
        ok &= show(at, "4. navigate to Room 2")
        print("  current_room:", at.session_state.current_room)

    # ---- 5. train Room 2 (SARSA), small episode count for speed ----------
    at.session_state.episodes = 300
    at.run()
    train = [b for b in at.button if "Train" in (b.label or "")]
    if train:
        train[0].click().run()
        ok &= show(at, "5. train Room 2 (SARSA, 300 episodes)")
        res = at.session_state.results.get(2)
        print("  solved:", res and res["solved"], "| steps:", res and len(res["path"]) - 1)

    # ---- 6. resize the grid ----------------------------------------------
    resize = [b for b in at.button if b.label == "Resize"]
    if resize:
        resize[0].click().run()
        ok &= show(at, "6. resize grid")

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES FOUND"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
