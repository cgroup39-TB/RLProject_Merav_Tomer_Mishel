"""
Escape Room RL — unified launcher.

Each room is its own standalone Streamlit app in its own folder
(Room1Tomer/, Room2Merav/, Room4Tomer/ ...) built independently by
different teammates, with their own modules -- some sharing filenames
(e.g. grid_env.py) or session_state key names (e.g. "seed", "paint_tool")
with unrelated meanings in different rooms. This router runs each room's
app.py in place via runpy, giving it an isolated session_state bucket and a
clean sys.modules namespace so nothing leaks across rooms, and adds the
cross-room progression (unlock next room on solve, skipping any room not
built yet).

Run locally with:
    streamlit run app.py
"""

import re
import runpy
import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Escape Room RL", page_icon="🗝️", layout="wide")

ROOT = Path(__file__).parent

ROOMS = [
    {"id": 1, "name": "The Laser Room", "subtitle": "Dynamic Programming", "dir": "Room1Tomer", "icon": "🔴"},
    {"id": 2, "name": "The Collapsing Bridge", "subtitle": "SARSA", "dir": "Room2Merav", "icon": "🌉"},
    {"id": 3, "name": "The Energy Room", "subtitle": "Q-Learning", "dir": "Room1Tomer", "icon": "⚡"},
    {"id": 4, "name": "The Drone Room", "subtitle": "Function Approximation (DQN)", "dir": "Room4Tomer", "icon": "🚁"},
    {"id": 5, "name": "The Shifting Warehouse", "subtitle": "Dynamic Obstacles (bonus)", "dir": "Room1Tomer", "icon": "🏭"},
]
ROOMS_BY_ID = {r["id"]: r for r in ROOMS}

# Rooms 1, 3 and 5 all live inside Room1Tomer/app.py (its own internal
# multi-room experience, sharing one grid engine/UI); enter_room() tells it
# which of its own rooms to show via "_embedded_room" rather than pointing
# three different ids at three different folders.

# Module names that different rooms define independently (e.g. both
# Room1Tomer and Room2Merav have their own unrelated grid_env.py) --
# purged from sys.modules before entering a room so each room's `import X`
# re-resolves against its own folder instead of a previous room's cached
# module object.
VOLATILE_MODULES = [
    "grid_env", "dp_solver", "sarsa_solver", "qlearning_solver",
    "continuous_env", "linear_qlearning_solver", "smoke_test",
    "room2_env", "sarsa_agent", "train_room2", "viz",
    "drone_env", "dqn_solver", "flight_canvas",
]

# session_state keys owned by this router; everything else belongs to
# whichever room is currently active and gets stashed/restored per room.
ROUTER_KEYS = {"current_room", "unlocked_rooms", "_snapshots", "_active_room"}

# Button-widget keys (grid-editor cells) can't be programmatically restored
# via session_state -- Streamlit rejects that at widget-instantiation time,
# not at assignment time, so these must be filtered out before restoring
# rather than just wrapped in try/except. Harmless to skip: buttons don't
# carry meaningful state across reruns anyway.
_SKIP_RESTORE_PATTERNS = [re.compile(r"^cell_\d+_\d+$"), re.compile(r"^paint_\d+_\d+$")]


def _should_skip_restore(key):
    return any(p.match(key) for p in _SKIP_RESTORE_PATTERNS)


def next_available_room(room_id):
    for r in ROOMS:
        if r["id"] > room_id and r["dir"] is not None:
            return r["id"]
    return None


def room_solved(room_id):
    """Read each room's own existing state to decide if it's been solved --
    no changes needed inside the room's own code for this."""
    if room_id == 1:
        res = st.session_state.get("results", {}).get(1)
        return bool(res and res.get("solved"))
    if room_id == 2:
        hist = st.session_state.get("history")
        return bool(hist and hist[-1].get("success"))
    if room_id == 4:
        res = st.session_state.get("results", {}).get(4)
        return bool(res and res.get("after", {}).get("solved"))
    if room_id in (3, 5):
        res = st.session_state.get("results", {}).get(room_id)
        return bool(res and res.get("solved"))
    return False


def enter_room(room_id):
    """Swap session_state to this room's own isolated bucket, stashing
    whatever the previously-active room had."""
    snapshots = st.session_state.setdefault("_snapshots", {})
    prev = st.session_state.get("_active_room")
    if prev != room_id:
        if prev is not None:
            snapshots[prev] = {
                k: v for k, v in st.session_state.items()
                if k not in ROUTER_KEYS
            }
        for k in [k for k in st.session_state.keys() if k not in ROUTER_KEYS]:
            del st.session_state[k]
        for k, v in snapshots.get(room_id, {}).items():
            if _should_skip_restore(k):
                continue
            try:
                st.session_state[k] = v
            except Exception:
                pass  # any other transient widget-instance key -- harmless to drop
    st.session_state["_active_room"] = room_id
    st.session_state["_embedded"] = True  # tells the room's app.py to skip its own set_page_config
    # For rooms sharing a folder with others (Room1Tomer hosts ids 1/3/5),
    # tells that app.py which of its own internal rooms to render.
    st.session_state["_embedded_room"] = room_id


def run_room(room_id):
    room = ROOMS_BY_ID[room_id]
    room_dir = ROOT / room["dir"]
    for m in VOLATILE_MODULES:
        sys.modules.pop(m, None)
    old_path = sys.path[:]
    sys.path.insert(0, str(room_dir))
    try:
        runpy.run_path(str(room_dir / "app.py"), run_name="__main__")
    finally:
        sys.path[:] = old_path


def render_room_nav():
    unlocked = st.session_state.setdefault("unlocked_rooms", {1})
    cols = st.columns(len(ROOMS))
    for i, room in enumerate(ROOMS):
        with cols[i]:
            is_built = room["dir"] is not None
            is_unlocked = room["id"] in unlocked and is_built
            is_active = room["id"] == st.session_state.get("current_room", 1)
            if not is_built:
                icon = "🚧"
            elif is_active:
                icon = "🟢"
            elif is_unlocked:
                icon = "🔓"
            else:
                icon = "🔒"
            label = f"{icon} Room {room['id']}\n{room['icon']} {room['name']}"
            if st.button(label, key=f"nav_{room['id']}", disabled=not is_unlocked, use_container_width=True):
                st.session_state["current_room"] = room["id"]
                st.rerun()


def main():
    st.title("🗝️ Escape Room RL")
    st.caption("Five rooms, five algorithms. Solve one to unlock the next.")
    render_room_nav()
    st.divider()

    current = st.session_state.get("current_room", 1)
    room = ROOMS_BY_ID[current]

    if room["dir"] is None:
        st.info(f"Room {current} — {room['name']} — 🚧 coming soon.")
        return

    enter_room(current)
    run_room(current)

    if room_solved(current):
        nxt = next_available_room(current)
        if nxt is not None:
            unlocked = st.session_state.setdefault("unlocked_rooms", {1})
            st.divider()
            if nxt not in unlocked:
                # first time this room is seen solved: show the call-to-action
                # button. On later revisits, the button key + rerun cycle can
                # collide with Streamlit's widget-state tracking, so instead
                # of re-rendering the same button, just show a plain notice.
                st.success(f"🎉 Room {current} solved! Room {nxt} is now unlocked.")
                if st.button(f"➡️ Continue to Room {nxt}", type="primary", key=f"continue_{current}"):
                    unlocked.add(nxt)
                    st.session_state["current_room"] = nxt
                    st.rerun()
            else:
                st.info(f"✅ Already solved — use the room bar above to jump to Room {nxt}.")


if __name__ == "__main__":
    main()
