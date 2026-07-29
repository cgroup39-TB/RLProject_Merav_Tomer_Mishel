"""
Headless smoke test for the Streamlit app, using Streamlit's official AppTest
harness. It really executes app.py (same code path as the browser), so any
exception, bad widget key or crash shows up here before we open a browser.

Run with:  python smoke_test.py
"""

import sys
from streamlit.testing.v1 import AppTest

from grid_env import GridWorldEnv, Cell, default_config


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


def check_determinism():
    print("\n=== standalone determinism check (GridWorldEnv seed=0) ===")
    ok = True

    def run(seed):
        cfg = default_config(10, 10)
        env = GridWorldEnv(cfg, slip_prob=0.3, seed=seed)
        s = env.reset()
        out = []
        for a in ([0] * 0 + [3] * 9 + [1] * 9):  # RIGHT*9 then DOWN*9 (perimeter route)
            s, r, done, trunc, _ = env.step(a)
            out.append((s, r, done))
            if done or trunc:
                break
        return out

    same = run(0) == run(0)
    print(f"  same seed -> identical sequence: {'OK' if same else 'FAIL'}")
    ok &= same

    differs = any(run(seed) != run(0) for seed in range(1, 6))
    print(f"  different seeds -> can differ  : {'OK' if differs else 'FAIL'}")
    ok &= differs
    return ok


def main():
    ok = True

    # ---- 1. app boots -----------------------------------------------------
    at = AppTest.from_file("app.py", default_timeout=120)
    at.run()
    ok &= show(at, "1. initial load (Room 1, laser-room default map)")

    # ---- 2. paint a wall ----------------------------------------------------

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

    # ---- 3. paint a laser cell + verify per-cell slip UI appears ----------
    at.session_state.paint_tool = "laser"
    at.run()
    cell = [b for b in at.button if b.key == "cell_4_4"]
    if not cell:
        print("  !! grid button cell_4_4 not found")
        ok = False
    else:
        cell[0].click().run()
        ok &= show(at, "3. paint laser at (4,4)")
        print("  grid_cells:", at.session_state.grid_cells.get((4, 4)))

    at.session_state.paint_tool = "slippery"
    at.run()
    has_slip_widgets = any("Slip probability" in (s.label or "") for s in at.slider) and \
        any("Slip direction" in (s.label or "") for s in at.selectbox)
    print(f"  paint-time slip prob/direction widgets present: {'OK' if has_slip_widgets else 'FAIL'}")
    ok &= has_slip_widgets

    cell = [b for b in at.button if b.key == "cell_5_5"]
    if cell:
        cell[0].click().run()
        ok &= show(at, "3b. paint slippery at (5,5) with custom prob/dir")
        print("  cell_slip_prob:", at.session_state.cell_slip_prob.get((5, 5)))
        print("  cell_slip_dir :", at.session_state.cell_slip_dir.get((5, 5)))

    # ---- 4. train Room 1 (DP) against the new laser-room default map -------

    train = [b for b in at.button if "Train" in (b.label or "")]
    if not train:
        print("  !! Train button not found")
        ok = False
    else:
        train[0].click().run()
        ok &= show(at, "4. train Room 1 (DP)")

        res = at.session_state.results.get(1)
        print("  solved:", res and res["solved"], "| steps:", res and len(res["path"]) - 1)
        print("  unlocked_room:", at.session_state.unlocked_room)

    # ---- 5. resize the grid -------------------------------------------------
    resize = [b for b in at.button if b.label == "Resize"]
    if resize:
        resize[0].click().run()
        ok &= show(at, "5. resize grid (overrides reset)")
        print("  cell_slip_prob after resize:", at.session_state.cell_slip_prob)
        print("  cell_slip_dir after resize :", at.session_state.cell_slip_dir)

    ok &= check_determinism()

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

    # ---- 7. navigate to Room 3 and paint a battery cell --------------------
    nav3 = [b for b in at.button if b.key == "room_nav_3"]
    if not nav3:
        print("  !! Room 3 nav button not found")
        ok = False
    else:
        nav3[0].click().run()
        ok &= show(at, "7. navigate to Room 3")
        print("  current_room:", at.session_state.current_room)

        at.session_state.room3_paint_tool = "battery"
        at.run()
        cell = [b for b in at.button if b.key == "r3cell_2_2"]
        if not cell:
            print("  !! grid button r3cell_2_2 not found")
            ok = False
        else:
            cell[0].click().run()
            ok &= show(at, "7b. paint battery at (2,2)")
            print("  room3_grid_cells:", at.session_state.room3_grid_cells.get((2, 2)))

        # guard-patrol tool: click two cells, verify they're recorded in order
        at.session_state.room3_paint_tool = "guard"
        at.run()
        for key in ("r3cell_1_1", "r3cell_1_2"):
            cell = [b for b in at.button if b.key == key]
            if cell:
                cell[0].click().run()
        ok &= show(at, "7c. paint guard patrol at (1,1) -> (1,2)")
        print("  room3_guard_path:", at.session_state.room3_guard_path)

        # reset back to the hand-designed default layout before training,
        # since the ad-hoc battery/guard edits above would otherwise leave
        # two batteries and a stray guard path on the actual training grid
        reset_btn = [b for b in at.button if "Reset to default layout" in (b.label or "")]
        if reset_btn:
            reset_btn[0].click().run()

    # ---- 8. train Room 3 (Q-Learning), small episode count for speed -------
    at.session_state.episodes = 300
    at.run()
    train = [b for b in at.button if "Train" in (b.label or "")]
    if train:
        train[0].click().run()
        ok &= show(at, "8. train Room 3 (Q-Learning, 300 episodes)")
        res = at.session_state.results.get(3)
        print("  solved:", res and res["solved"], "| steps:", res and len(res["path"]) - 1)
        print("  unlocked_room:", at.session_state.unlocked_room)

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES FOUND"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
