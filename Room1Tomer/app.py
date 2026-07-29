"""
Escape Room RL — unified Streamlit app.

One app, five rooms. You configure the grid/parameters for the active room,
train the agent, and if it solves the room, the next one unlocks — exactly
like a real escape room.

Run locally with:
    streamlit run app.py
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from grid_env import (
    GridConfig, GridWorldEnv, EnergyRoomEnv, Cell, ACTION_ARROWS, default_config,
    default_room3_config, SLIP_DIR_MODES, SLIP_DIR_LABELS,
)
from dp_solver import value_iteration, policy_iteration
from sarsa_solver import sarsa
from qlearning_solver import q_learning
from continuous_env import WarehouseRoom, N_ACTIONS
from linear_qlearning_solver import Features, train_linear_qlearning, greedy_action


st.set_page_config(page_title="Escape Room RL", page_icon="🗝️", layout="wide")

TOOL_LABELS = {
    Cell.EMPTY: "· Empty",
    Cell.WALL: "🧱 Metal Wall",
    Cell.SLIPPERY: "≈ Ice Floor",
    Cell.LASER: "🔴 Laser",
    Cell.GOAL: "🖥️ Control Panel (Goal)",
    Cell.TRAP: "☠ Trap",
    Cell.START: "🚦 Start",
}
CELL_ICON = {
    Cell.EMPTY: "",
    Cell.WALL: "🧱",
    Cell.SLIPPERY: "≈",
    Cell.LASER: "🔴",
    Cell.GOAL: "🖥️",
    Cell.TRAP: "☠",
    Cell.START: "🚦",
}

# Room 3 (Energy Room) has its own grid, its own paint palette (battery,
# switches, guard patrol, ...) and its own fixed 10x10 layout, so it keeps a
# separate tool/icon table rather than overloading Room 1/2's shared one.
ROOM3_TOOL_LABELS = {
    Cell.EMPTY: "· Empty",
    Cell.WALL: "🧱 Wall",
    Cell.START: "🚦 Start",
    Cell.GOAL: "🚪 Exit (locked)",
    Cell.BATTERY: "🔋 Battery",
    Cell.SWITCH_BLUE: "🔵 Blue Switch",
    Cell.SWITCH_RED: "🔴 Red Switch",
    Cell.DOOR: "🚧 Door (opens after blue switch)",
    Cell.SHORTCUT: "⚡ Shortcut (needs battery)",
    Cell.CHARGER: "🔌 Charging Station",
    Cell.ELECTRIC_TRAP: "🌩️ Electric Trap (live on even steps)",
    Cell.TRAP: "☠ Trap",
    "guard": "🤖 Guard Patrol (click cells in order)",
}
ROOM3_CELL_ICON = {
    Cell.EMPTY: "",
    Cell.WALL: "🧱",
    Cell.START: "🚦",
    Cell.GOAL: "🚪",
    Cell.BATTERY: "🔋",
    Cell.SWITCH_BLUE: "🔵",
    Cell.SWITCH_RED: "🔴",
    Cell.DOOR: "🚧",
    Cell.SHORTCUT: "⚡",
    Cell.CHARGER: "🔌",
    Cell.ELECTRIC_TRAP: "🌩️",
    Cell.TRAP: "☠",
}
ROOM3_LETTERS = {
    Cell.BATTERY: ("K", "gold"),
    Cell.SWITCH_BLUE: ("B", "deepskyblue"),
    Cell.SWITCH_RED: ("R", "red"),
    Cell.DOOR: ("D", "orange"),
    Cell.SHORTCUT: ("H", "aquamarine"),
    Cell.CHARGER: ("C", "lime"),
    Cell.ELECTRIC_TRAP: ("E", "violet"),
    Cell.TRAP: ("X", "orangered"),
}

ROOMS = [
    {"id": 1, "name": "Dynamic Programming", "subtitle": "מודל הסביבה ידוע"},
    {"id": 2, "name": "SARSA", "subtitle": "מודל לא ידוע, On-Policy"},
    {"id": 3, "name": "Q-Learning", "subtitle": "מודל לא ידוע, Off-Policy"},
    {"id": 4, "name": "Function Approximation", "subtitle": "מרחב מצבים רציף"},
    {"id": 5, "name": "Dynamic Obstacles", "subtitle": "בונוס — Observation מוגבל"},
]


# ---------------------------------------------------------------------------
# session state initialization
# ---------------------------------------------------------------------------
def init_state():
    if "initialized" in st.session_state:
        return
    st.session_state.initialized = True
    st.session_state.rows = 10
    st.session_state.cols = 10
    st.session_state.paint_tool = Cell.EMPTY
    st.session_state.paint_slip_prob = 0.3
    st.session_state.paint_slip_dir = "cw"
    default = default_config(10, 10)
    st.session_state.grid_cells = dict(default.cells)
    st.session_state.cell_rewards = {}
    st.session_state.cell_slip_prob = dict(default.cell_slip_prob)
    st.session_state.cell_slip_dir = dict(default.cell_slip_dir)
    st.session_state.start = default.start
    st.session_state.goal = default.goal
    st.session_state.current_room = 1
    st.session_state.unlocked_room = 1
    st.session_state.results = {}

    r3_default = default_room3_config()
    st.session_state.room3_grid_cells = dict(r3_default.cells)
    st.session_state.room3_start = r3_default.start
    st.session_state.room3_goal = r3_default.goal
    st.session_state.room3_guard_path = list(r3_default.guard_path)
    st.session_state.room3_paint_tool = Cell.EMPTY


init_state()


# ---------------------------------------------------------------------------
# grid editing
# ---------------------------------------------------------------------------
def paint_cell(r, c):
    tool = st.session_state.paint_tool
    if tool == Cell.START:
        old = st.session_state.start
        st.session_state.grid_cells.pop(old, None)
        st.session_state.start = (r, c)
        st.session_state.grid_cells.pop((r, c), None)
    elif tool == Cell.GOAL:
        old = st.session_state.goal
        st.session_state.grid_cells.pop(old, None)
        st.session_state.goal = (r, c)
        st.session_state.grid_cells[(r, c)] = Cell.GOAL
    elif tool == Cell.EMPTY:
        st.session_state.grid_cells.pop((r, c), None)
    else:
        if (r, c) not in (st.session_state.start, st.session_state.goal):
            st.session_state.grid_cells[(r, c)] = tool

    if tool == Cell.SLIPPERY and (r, c) not in (st.session_state.start, st.session_state.goal):
        st.session_state.cell_slip_prob[(r, c)] = st.session_state.paint_slip_prob
        st.session_state.cell_slip_dir[(r, c)] = st.session_state.paint_slip_dir
    else:
        st.session_state.cell_slip_prob.pop((r, c), None)
        st.session_state.cell_slip_dir.pop((r, c), None)


def build_config():
    cfg = GridConfig(
        rows=st.session_state.rows,
        cols=st.session_state.cols,
        cells=dict(st.session_state.grid_cells),
        cell_rewards=dict(st.session_state.cell_rewards),
        cell_slip_prob=dict(st.session_state.cell_slip_prob),
        cell_slip_dir=dict(st.session_state.cell_slip_dir),
        start=st.session_state.start,
        goal=st.session_state.goal,
    )
    return cfg


def cell_label(r, c):
    if (r, c) == st.session_state.start:
        return "🚦"
    if (r, c) == st.session_state.goal:
        return "🖥️"
    t = st.session_state.grid_cells.get((r, c), Cell.EMPTY)
    return CELL_ICON.get(t, "")


def render_grid_editor():
    st.subheader("📖 Grid Editor")
    st.caption(f"Active tool: **{TOOL_LABELS[st.session_state.paint_tool]}** — click a cell to paint")
    for r in range(st.session_state.rows):
        cols = st.columns(st.session_state.cols)
        for c in range(st.session_state.cols):
            with cols[c]:
                with st.container(key=f"cellwrap_{r}_{c}"):
                    st.button(
                        cell_label(r, c) or " ",
                        key=f"cell_{r}_{c}",
                        on_click=paint_cell,
                        args=(r, c),
                        use_container_width=True,
                    )


# ---------------------------------------------------------------------------
# grid editing — Room 3 (its own grid, its own palette, fixed 10x10)
# ---------------------------------------------------------------------------
def paint_room3_cell(r, c):
    tool = st.session_state.room3_paint_tool
    if tool == "guard":
        path = st.session_state.room3_guard_path
        if (r, c) in path:
            path.remove((r, c))
        else:
            path.append((r, c))
        return
    if tool == Cell.START:
        old = st.session_state.room3_start
        st.session_state.room3_grid_cells.pop(old, None)
        st.session_state.room3_start = (r, c)
        st.session_state.room3_grid_cells.pop((r, c), None)
    elif tool == Cell.GOAL:
        old = st.session_state.room3_goal
        st.session_state.room3_grid_cells.pop(old, None)
        st.session_state.room3_goal = (r, c)
        st.session_state.room3_grid_cells[(r, c)] = Cell.GOAL
    elif tool == Cell.EMPTY:
        st.session_state.room3_grid_cells.pop((r, c), None)
    else:
        if (r, c) not in (st.session_state.room3_start, st.session_state.room3_goal):
            st.session_state.room3_grid_cells[(r, c)] = tool


def build_room3_config():
    return GridConfig(
        rows=10,
        cols=10,
        cells=dict(st.session_state.room3_grid_cells),
        start=st.session_state.room3_start,
        goal=st.session_state.room3_goal,
        guard_path=list(st.session_state.room3_guard_path),
    )


def room3_cell_label(r, c):
    if (r, c) == st.session_state.room3_start:
        return "🚦"
    if (r, c) == st.session_state.room3_goal:
        return "🚪"
    if (r, c) in st.session_state.room3_guard_path:
        return f"🤖{st.session_state.room3_guard_path.index((r, c)) + 1}"
    t = st.session_state.room3_grid_cells.get((r, c), Cell.EMPTY)
    return ROOM3_CELL_ICON.get(t, "")


def render_room3_grid_editor():
    st.subheader("📖 Grid Editor")
    tool = st.session_state.room3_paint_tool
    st.caption(
        f"Active tool: **{ROOM3_TOOL_LABELS[tool]}** — click a cell to paint. "
        "Guard tool: click cells in patrol order, click a numbered cell again to remove it."
    )
    for r in range(10):
        cols = st.columns(10)
        for c in range(10):
            with cols[c]:
                with st.container(key=f"r3cellwrap_{r}_{c}"):
                    st.button(
                        room3_cell_label(r, c) or " ",
                        key=f"r3cell_{r}_{c}",
                        on_click=paint_room3_cell,
                        args=(r, c),
                        use_container_width=True,
                    )


# ---------------------------------------------------------------------------
# lab theme — CSS injected per-cell (no external image assets)
# ---------------------------------------------------------------------------
CHROME_CSS = """
.stApp {
    background: linear-gradient(160deg, #060a10 0%, #0d141c 55%, #0a1a14 100%);
}
[data-testid="stSidebar"] {
    background: #0f1720;
    border-right: 1px solid rgba(56, 209, 255, 0.25);
}
.stButton > button {
    font-size: 1.15rem;
    border-radius: 8px;
    border: 1px solid rgba(120, 140, 160, 0.35);
    background: #182531;
    transition: transform 0.08s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 10px rgba(56, 209, 255, 0.45);
}
@keyframes laserpulse {
    0%, 100% { box-shadow: 0 0 6px 1px #ff2e2ea0; }
    50% { box-shadow: 0 0 16px 4px #ff2e2ee0; }
}
@keyframes goalpulse {
    0%, 100% { box-shadow: 0 0 6px 1px #35ff8aa0; }
    50% { box-shadow: 0 0 18px 4px #35ff8ae0; }
}
"""

CELL_TYPE_CSS = {
    Cell.WALL: (
        "background: repeating-linear-gradient(135deg, #3a4048, #3a4048 4px, "
        "#2b3036 4px, #2b3036 8px) !important; border-color: #52585f !important;"
    ),
    Cell.LASER: (
        "background: radial-gradient(circle, #ff5b5b, #7a0000) !important; "
        "border-color: #ff8080 !important; animation: laserpulse 1.4s ease-in-out infinite;"
    ),
    Cell.SLIPPERY: (
        "background: linear-gradient(160deg, #bfe9ff, #4fa8d8) !important; "
        "border-color: #d8f4ff !important;"
    ),
    Cell.TRAP: (
        "background: radial-gradient(circle, #6e1414, #2a0505) !important; "
        "border-color: #a83232 !important;"
    ),
}
GOAL_CSS = (
    "background: radial-gradient(circle, #35ff8a, #0a5c2e) !important; "
    "border-color: #7dffb8 !important; animation: goalpulse 1.6s ease-in-out infinite;"
)
START_CSS = (
    "background: radial-gradient(circle, #ffb84d, #a15b00) !important; "
    "border-color: #ffd699 !important;"
)


def inject_grid_css():
    rules = [CHROME_CSS]
    for (r, c), cell_type in st.session_state.grid_cells.items():
        if (r, c) in (st.session_state.start, st.session_state.goal):
            continue
        style = CELL_TYPE_CSS.get(cell_type)
        if style:
            rules.append(f".st-key-cellwrap_{r}_{c} button {{ {style} }}")
    sr, sc = st.session_state.start
    gr, gc = st.session_state.goal
    rules.append(f".st-key-cellwrap_{sr}_{sc} button {{ {START_CSS} }}")
    rules.append(f".st-key-cellwrap_{gr}_{gc} button {{ {GOAL_CSS} }}")
    st.html("<style>\n" + "\n".join(rules) + "\n</style>")


ROOM3_CELL_TYPE_CSS = {
    Cell.WALL: CELL_TYPE_CSS[Cell.WALL],
    Cell.TRAP: CELL_TYPE_CSS[Cell.TRAP],
    Cell.BATTERY: (
        "background: radial-gradient(circle, #ffe066, #a67c00) !important; "
        "border-color: #fff2b0 !important;"
    ),
    Cell.SWITCH_BLUE: (
        "background: radial-gradient(circle, #4fa8ff, #0b3d91) !important; "
        "border-color: #9fd0ff !important;"
    ),
    Cell.SWITCH_RED: (
        "background: radial-gradient(circle, #ff5b5b, #7a0000) !important; "
        "border-color: #ff8080 !important;"
    ),
    Cell.DOOR: (
        "background: repeating-linear-gradient(45deg, #b8860b, #b8860b 4px, "
        "#8a6508 4px, #8a6508 8px) !important; border-color: #ffd27f !important;"
    ),
    Cell.SHORTCUT: (
        "background: linear-gradient(160deg, #7dffe0, #0aa88a) !important; "
        "border-color: #baffef !important;"
    ),
    Cell.CHARGER: (
        "background: radial-gradient(circle, #7dff8a, #0a5c2e) !important; "
        "border-color: #b8ffc2 !important;"
    ),
    Cell.ELECTRIC_TRAP: (
        "background: radial-gradient(circle, #d9a6ff, #4b0082) !important; "
        "border-color: #e8caff !important; animation: laserpulse 1.1s ease-in-out infinite;"
    ),
}
GUARD_PATH_CSS = "outline: 3px solid #ff2e2e !important; outline-offset: -3px;"


def inject_room3_css():
    rules = [CHROME_CSS]
    for (r, c), cell_type in st.session_state.room3_grid_cells.items():
        if (r, c) in (st.session_state.room3_start, st.session_state.room3_goal):
            continue
        style = ROOM3_CELL_TYPE_CSS.get(cell_type)
        if style:
            rules.append(f".st-key-r3cellwrap_{r}_{c} button {{ {style} }}")
    for (r, c) in st.session_state.room3_guard_path:
        rules.append(f".st-key-r3cellwrap_{r}_{c} button {{ {GUARD_PATH_CSS} }}")
    sr, sc = st.session_state.room3_start
    gr, gc = st.session_state.room3_goal
    rules.append(f".st-key-r3cellwrap_{sr}_{sc} button {{ {START_CSS} }}")
    rules.append(f".st-key-r3cellwrap_{gr}_{gc} button {{ {GOAL_CSS} }}")
    st.html("<style>\n" + "\n".join(rules) + "\n</style>")


def inject_chrome_css():
    """Room 5 has no per-cell grid to style, just the shared lab theme."""
    st.html(f"<style>\n{CHROME_CSS}\n</style>")


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------
def render_room5_sidebar():
    """
    Room 5 has no grid to paint (the room is continuous, and its obstacles
    are randomised every episode by design), so its whole params shape is
    different from the grid rooms' -- it gets its own sidebar rather than
    threading a second params dict through render_sidebar's grid-room path.
    """
    with st.sidebar:
        st.header("🎮 Game Setup")

        st.subheader("🏭 Warehouse parameters")
        n_obstacles = st.number_input("Number of crates", 0, 20, 6, key="r5_n_obstacles")
        lookahead_m = st.slider("Radar range (m)", 0.5, 5.0, 2.0, key="r5_lookahead_m")
        n_rays = st.slider("Radar rays", 4, 32, 16, key="r5_n_rays")
        decision_interval = st.slider("Ticks per decision", 5, 50, 25, key="r5_decision_interval")
        max_seconds = st.slider("Max episode time (s)", 5.0, 60.0, 25.0, key="r5_max_seconds")

        st.divider()
        st.subheader("Rewards")
        time_cost = st.number_input("Time cost (per second)", value=-1.0, step=0.5, key="r5_time_cost")
        progress_reward = st.number_input("Progress bonus (per metre closer)", value=4.0, step=0.5, key="r5_progress_reward")
        reverse_cost = st.number_input("Reversal penalty", value=-2.0, step=0.5, key="r5_reverse_cost")
        wall_reward = st.number_input("Wall bump penalty", value=-1.0, step=0.5, key="r5_wall_reward")
        crash_reward = st.number_input("Crash penalty", value=-50.0, step=5.0, key="r5_crash_reward")
        goal_reward = st.number_input("Goal reward", value=100.0, step=5.0, key="r5_goal_reward")

        st.divider()
        st.subheader("Function approximation (tile coding)")
        n_tilings = st.slider("Tilings", 1, 8, 4, key="r5_n_tilings")
        tiles = st.slider("Tiles per side", 4, 20, 10, key="r5_tiles")
        ray_bins = st.slider("Radar bins per ray", 2, 8, 4, key="r5_ray_bins")

        st.divider()
        st.subheader("Training Hyperparameters")
        alpha = st.slider("Learning rate α", 0.01, 1.0, 0.3, key="r5_alpha")
        epsilon = st.slider("Initial ε (exploration)", 0.0, 1.0, 1.0, key="r5_epsilon")
        epsilon_min = st.slider("Min ε", 0.0, 0.5, 0.05, key="r5_epsilon_min")
        epsilon_decay = st.slider("ε decay per episode", 0.90, 0.9999, 0.997, key="r5_epsilon_decay", format="%.4f")
        episodes = st.number_input("Episodes", min_value=100, max_value=5000, value=1500, step=100, key="r5_episodes")
        seed = st.number_input("Random seed", min_value=0, value=0, step=1, key="r5_seed")

        st.divider()
        train_clicked = st.button("▶ Train Agent", type="primary", use_container_width=True)

        params = dict(
            n_obstacles=int(n_obstacles), lookahead_m=lookahead_m, n_rays=int(n_rays),
            decision_interval=int(decision_interval), max_seconds=max_seconds,
            time_cost=time_cost, progress_reward=progress_reward, reverse_cost=reverse_cost,
            wall_reward=wall_reward, crash_reward=crash_reward, goal_reward=goal_reward,
            n_tilings=int(n_tilings), tiles=int(tiles), ray_bins=int(ray_bins),
            alpha=alpha, epsilon=epsilon, epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay, episodes=int(episodes), seed=int(seed),
        )
        return params, train_clicked


def render_sidebar(current_room):
    with st.sidebar:
        st.header("🎮 Game Setup")

        if current_room == 3:
            st.subheader("Grid")
            st.caption("10×10, fixed layout (hand-designed for the Energy Room).")
            if st.button("↺ Reset to default layout", use_container_width=True):
                r3_default = default_room3_config()
                st.session_state.room3_grid_cells = dict(r3_default.cells)
                st.session_state.room3_start = r3_default.start
                st.session_state.room3_goal = r3_default.goal
                st.session_state.room3_guard_path = list(r3_default.guard_path)
                st.rerun()

            st.divider()
            st.subheader("Paint tool")
            st.radio(
                "Active tool",
                options=list(ROOM3_TOOL_LABELS.keys()),
                format_func=lambda t: ROOM3_TOOL_LABELS[t],
                key="room3_paint_tool",
                label_visibility="collapsed",
            )
            if st.session_state.room3_guard_path:
                order = " → ".join(f"({r},{c})" for r, c in st.session_state.room3_guard_path)
                st.caption(f"🤖 Patrol order: {order}")
        else:
            st.subheader("Grid size")
            col1, col2 = st.columns(2)
            rows = col1.number_input("Rows", min_value=3, max_value=20, value=st.session_state.rows)
            cols = col2.number_input("Cols", min_value=3, max_value=20, value=st.session_state.cols)
            b1, b2 = st.columns(2)
            if b1.button("Resize", use_container_width=True):
                st.session_state.rows, st.session_state.cols = int(rows), int(cols)
                st.session_state.start = (0, 0)
                st.session_state.goal = (int(rows) - 1, int(cols) - 1)
                st.session_state.grid_cells = {st.session_state.goal: Cell.GOAL}
                st.session_state.cell_rewards = {}
                st.session_state.cell_slip_prob = {}
                st.session_state.cell_slip_dir = {}
                st.rerun()
            if b2.button("Clear", use_container_width=True):
                st.session_state.grid_cells = {st.session_state.goal: Cell.GOAL}
                st.session_state.cell_rewards = {}
                st.session_state.cell_slip_prob = {}
                st.session_state.cell_slip_dir = {}
                st.rerun()

            st.divider()
            st.subheader("Paint tool & rewards")
            tool = st.radio(
                "Active tool",
                options=list(TOOL_LABELS.keys()),
                format_func=lambda t: TOOL_LABELS[t],
                key="paint_tool",
                label_visibility="collapsed",
            )
            if tool == Cell.SLIPPERY:
                st.slider(
                    "Slip probability for this cell", 0.0, 1.0,
                    key="paint_slip_prob",
                )
                st.selectbox(
                    "Slip direction for this cell",
                    options=SLIP_DIR_MODES,
                    format_func=lambda m: SLIP_DIR_LABELS[m],
                    key="paint_slip_dir",
                )

        st.markdown("**Step reward**")
        step_reward = st.number_input("Step", value=-1.0, step=0.1, key="step_reward")
        shaping = st.checkbox("🧭 Potential-based goal shaping", value=True, key="potential_shaping")

        st.divider()
        st.subheader("Dynamics")
        slip_prob = st.slider("Default slip probability", 0.0, 1.0, 0.3, key="slip_prob")
        gamma = st.slider("Discount γ", 0.0, 0.999, 0.95, key="gamma")
        seed = st.number_input("Random seed", min_value=0, value=0, step=1, key="seed")

        room3_extra = {}
        if current_room == 3:
            st.divider()
            st.subheader("⚡ Energy Room rewards")
            room3_extra["wall_penalty"] = st.number_input("🧱 Wall bump penalty", value=-2.0, step=0.5, key="r3_wall_penalty")
            room3_extra["guard_penalty"] = st.number_input("🤖 Guard collision penalty", value=-30.0, step=5.0, key="r3_guard_penalty")
            room3_extra["electric_trap_reward"] = st.number_input("🌩️ Electric trap penalty", value=-20.0, step=5.0, key="r3_electric_trap_reward")
            room3_extra["battery_reward"] = st.number_input("🔋 Battery pickup bonus", value=15.0, step=5.0, key="r3_battery_reward")
            room3_extra["switch_blue_reward"] = st.number_input("🔵 Blue switch bonus", value=20.0, step=5.0, key="r3_switch_blue_reward")
            room3_extra["switch_red_reward"] = st.number_input("🔴 Red switch bonus", value=25.0, step=5.0, key="r3_switch_red_reward")
            room3_extra["charger_reward"] = st.number_input("🔌 Charging station bonus (one-time)", value=5.0, step=1.0, key="r3_charger_reward")
            room3_extra["incomplete_exit_reward"] = st.number_input("🚪 Premature-exit penalty", value=-10.0, step=5.0, key="r3_incomplete_exit_reward")
            goal_reward = st.number_input("✅ Successful-exit reward", value=150.0, step=10.0, key="r3_goal_reward")
            trap_reward = st.number_input("☠ Trap penalty", value=-100.0, step=5.0, key="r3_trap_reward")
            laser_reward = -20.0
        else:
            st.divider()
            st.subheader("Hazard penalty")
            laser_reward = st.number_input("🔴 Laser penalty", value=-20.0, step=5.0, key="laser_reward")

            st.divider()
            st.subheader("Terminal rewards")
            goal_reward = st.number_input("Goal reward", value=100.0, step=5.0, key="goal_reward")
            trap_reward = st.number_input("Trap reward", value=-100.0, step=5.0, key="trap_reward")

        extra = {}
        if current_room in (2, 3):
            st.divider()
            st.subheader("Training Hyperparameters")
            extra["alpha"] = st.slider("Learning rate α", 0.01, 1.0, 0.1, key="alpha")
            extra["epsilon"] = st.slider("Initial ε (exploration)", 0.0, 1.0, 1.0, key="epsilon")
            extra["epsilon_min"] = st.slider("Min ε", 0.0, 0.5, 0.05, key="epsilon_min")
            extra["epsilon_decay"] = st.slider(
                "ε decay per episode", 0.90, 0.9999, 0.998, key="epsilon_decay", format="%.4f"
            )
            extra["episodes"] = st.number_input(
                "Episodes", min_value=100, max_value=20000, value=2000, step=100, key="episodes"
            )

        st.divider()
        train_clicked = st.button("▶ Train Agent", type="primary", use_container_width=True)

        params = dict(
            step_reward=step_reward, shaping=shaping, slip_prob=slip_prob,
            gamma=gamma, goal_reward=goal_reward, trap_reward=trap_reward,
            laser_reward=laser_reward, seed=int(seed),
        )
        params.update(extra)
        params.update(room3_extra)
        return params, train_clicked


# ---------------------------------------------------------------------------
# training + results (Room 1 — DP)
# ---------------------------------------------------------------------------
def train_room1(params):
    cfg = build_config()
    env = GridWorldEnv(
        cfg,
        slip_prob=params["slip_prob"],
        step_reward=params["step_reward"],
        goal_reward=params["goal_reward"],
        trap_reward=params["trap_reward"],
        laser_reward=params["laser_reward"],
        gamma=params["gamma"],
        potential_shaping=params["shaping"],
        seed=params["seed"],
    )
    P = env.get_transition_model()
    V_vi, policy_vi, info_vi = value_iteration(P, env.n_states, env.n_actions, gamma=params["gamma"])
    V_pi, policy_pi, info_pi = policy_iteration(P, env.n_states, env.n_actions, gamma=params["gamma"])

    s = env.reset()
    path = [env.i2s(s)]
    total_reward = 0.0
    solved = False
    for _ in range(env.max_steps):
        a = int(policy_vi[s])
        s, r, done, truncated, _ = env.step(a)
        total_reward += r
        path.append(env.i2s(s))
        if done:
            solved = env.cfg.cell_type(*env.i2s(s)) == Cell.GOAL
            break
        if truncated:
            break

    st.session_state.results[1] = dict(
        env=env, V=V_vi, policy=policy_vi, info_vi=info_vi, info_pi=info_pi,
        policies_match=bool(np.array_equal(policy_vi, policy_pi)),
        path=path, total_reward=total_reward, solved=solved,
    )
    if solved:
        st.session_state.unlocked_room = max(st.session_state.unlocked_room, 2)


# ---------------------------------------------------------------------------
# training + results (Room 2 — SARSA)
# ---------------------------------------------------------------------------
def train_room2(params):
    cfg = build_config()
    env = GridWorldEnv(
        cfg,
        slip_prob=params["slip_prob"],
        step_reward=params["step_reward"],
        goal_reward=params["goal_reward"],
        trap_reward=params["trap_reward"],
        laser_reward=params["laser_reward"],
        gamma=params["gamma"],
        potential_shaping=params["shaping"],
        seed=params["seed"],
    )
    Q, policy, info = sarsa(
        env,
        env.n_states,
        env.n_actions,
        episodes=params["episodes"],
        alpha=params["alpha"],
        gamma=params["gamma"],
        epsilon=params["epsilon"],
        epsilon_min=params["epsilon_min"],
        epsilon_decay=params["epsilon_decay"],
        max_steps=env.max_steps,
        seed=params["seed"],
    )

    s = env.reset()
    path = [env.i2s(s)]
    total_reward = 0.0
    solved = False
    for _ in range(env.max_steps):
        a = int(policy[s])
        s, r, done, truncated, _ = env.step(a)
        total_reward += r
        path.append(env.i2s(s))
        if done:
            solved = env.cfg.cell_type(*env.i2s(s)) == Cell.GOAL
            break
        if truncated:
            break

    st.session_state.results[2] = dict(
        env=env, V=Q.max(axis=1), policy=policy, info=info,
        path=path, total_reward=total_reward, solved=solved,
    )
    if solved:
        st.session_state.unlocked_room = max(st.session_state.unlocked_room, 3)


# ---------------------------------------------------------------------------
# training + results (Room 3 — Q-Learning, Energy Room)
# ---------------------------------------------------------------------------
def train_room3(params):
    cfg = build_room3_config()
    env = EnergyRoomEnv(
        cfg,
        guard_path=cfg.guard_path,
        slip_prob=params["slip_prob"],
        step_reward=params["step_reward"],
        goal_reward=params["goal_reward"],
        trap_reward=params["trap_reward"],
        laser_reward=params["laser_reward"],
        gamma=params["gamma"],
        potential_shaping=params["shaping"],
        seed=params["seed"],
        battery_reward=params["battery_reward"],
        switch_blue_reward=params["switch_blue_reward"],
        switch_red_reward=params["switch_red_reward"],
        guard_penalty=params["guard_penalty"],
        wall_penalty=params["wall_penalty"],
        electric_trap_reward=params["electric_trap_reward"],
        charger_reward=params["charger_reward"],
        incomplete_exit_reward=params["incomplete_exit_reward"],
    )
    Q, policy, info = q_learning(
        env,
        env.n_states,
        env.n_actions,
        episodes=params["episodes"],
        alpha=params["alpha"],
        gamma=params["gamma"],
        epsilon=params["epsilon"],
        epsilon_min=params["epsilon_min"],
        epsilon_decay=params["epsilon_decay"],
        max_steps=env.max_steps,
        seed=params["seed"],
    )

    s = env.reset()
    path = [env.decode_flags(s)[0]]
    total_reward = 0.0
    solved = False
    for _ in range(env.max_steps):
        a = int(policy[s])
        s, r, done, truncated, _ = env.step(a)
        total_reward += r
        path.append(env.decode_flags(s)[0])
        if done:
            pos, has_battery, blue_on, red_on = env.decode_flags(s)
            solved = pos == env.cfg.goal and has_battery and blue_on and red_on
            break
        if truncated:
            break

    st.session_state.results[3] = dict(
        env=env, Q=Q, policy=policy, info=info,
        path=path, total_reward=total_reward, solved=solved,
    )
    if solved:
        # Room 4 (Function Approximation) isn't built yet in this app, so
        # solving the Energy Room unlocks the next room that actually
        # exists -- Room 5 -- rather than dead-ending on a "coming soon" gate.
        st.session_state.unlocked_room = max(st.session_state.unlocked_room, 5)


# ---------------------------------------------------------------------------
# training + results (Room 5 — semi-gradient Q-Learning, Shifting Warehouse)
# ---------------------------------------------------------------------------
def train_room5(params):
    env = WarehouseRoom(
        n_obstacles=params["n_obstacles"],
        lookahead_m=params["lookahead_m"],
        n_rays=params["n_rays"],
        decision_interval=params["decision_interval"],
        max_seconds=params["max_seconds"],
        time_cost=params["time_cost"],
        goal_reward=params["goal_reward"],
        wall_reward=params["wall_reward"],
        crash_reward=params["crash_reward"],
        progress_reward=params["progress_reward"],
        reverse_cost=params["reverse_cost"],
        seed=params["seed"],
    )
    features = Features(
        arena_size=10.0,
        n_rays=env.n_rays,
        lookahead_m=env.lookahead_m,
        n_tilings=params["n_tilings"],
        tiles=params["tiles"],
        ray_bins=params["ray_bins"],
    )
    weights, info = train_linear_qlearning(
        env, features, N_ACTIONS,
        episodes=params["episodes"], alpha=params["alpha"], gamma=0.995,
        epsilon=params["epsilon"], epsilon_min=params["epsilon_min"],
        epsilon_decay=params["epsilon_decay"], seed=params["seed"],
    )

    # The whole point of this room is generalisation, so the reported run
    # is evaluated greedily on a layout the training loop never saw --
    # a different obstacle seed, not one of the per-episode training draws.
    eval_env = WarehouseRoom(
        n_obstacles=params["n_obstacles"],
        lookahead_m=params["lookahead_m"],
        n_rays=params["n_rays"],
        decision_interval=params["decision_interval"],
        max_seconds=params["max_seconds"],
        time_cost=params["time_cost"],
        goal_reward=params["goal_reward"],
        wall_reward=params["wall_reward"],
        crash_reward=params["crash_reward"],
        progress_reward=params["progress_reward"],
        reverse_cost=params["reverse_cost"],
        seed=params["seed"] + 999_983,
    )
    obs = eval_env.reset()
    active = features.active(obs)
    path = [(eval_env.x, eval_env.y)]
    total_reward = 0.0
    solved = False
    for _ in range(eval_env.max_decisions):
        a = greedy_action(weights, active)
        obs, r, done, _ = eval_env.step(a)
        active = features.active(obs)
        total_reward += r
        path.append((eval_env.x, eval_env.y))
        if done:
            solved = eval_env.escaped()
            break

    st.session_state.results[5] = dict(
        env=eval_env, weights=weights, info=info,
        path=path, total_reward=total_reward, solved=solved,
    )


def plot_result(res):
    env = res["env"]
    V = res["V"].reshape(env.rows, env.cols)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(V, cmap="viridis")
    fig.colorbar(im, ax=ax, label="V(s)", fraction=0.046)

    for i in range(env.n_states):
        r, c = env.i2s(i)
        t = env.cfg.cell_type(r, c)
        if (r, c) == env.cfg.goal:
            ax.text(c, r, "G", ha="center", va="center", color="red", fontweight="bold")
        elif (r, c) == env.cfg.start:
            ax.text(c, r, "S", ha="center", va="center", color="lime", fontweight="bold")
        elif t == Cell.WALL:
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="black"))
        elif t == Cell.TRAP:
            ax.text(c, r, "X", ha="center", va="center", color="orangered", fontweight="bold")
        elif t == Cell.LASER:
            # Laser cells are non-terminal — the agent can still be routed through
            # one, so its policy arrow is meaningful and shown on top of the marker.
            ax.add_patch(plt.Circle((c, r), 0.42, color="red", alpha=0.45))
            a = int(res["policy"][i])
            ax.text(c, r, ACTION_ARROWS[a], ha="center", va="center", color="yellow", fontweight="bold", fontsize=9)
        else:
            a = int(res["policy"][i])
            ax.text(c, r, ACTION_ARROWS[a], ha="center", va="center", color="white", fontsize=9)

    path_r = [p[0] for p in res["path"]]
    path_c = [p[1] for p in res["path"]]
    ax.plot(path_c, path_r, color="orange", linewidth=2, marker="o", markersize=3)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    return fig


def plot_learning_curve(info, window=50):
    rewards = info["reward_history"]
    roll = []
    for i in range(len(rewards)):
        lo = max(0, i - window + 1)
        chunk = rewards[lo:i + 1]
        roll.append(sum(chunk) / len(chunk))

    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.plot(rewards, color="gray", alpha=0.3, linewidth=0.7, label="episode reward")
    ax.plot(roll, color="orange", linewidth=2, label=f"rolling mean ({window})")
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    return fig


def render_results(room):
    st.subheader("📊 Results")
    res = st.session_state.results.get(room)
    if res is None:
        st.info("Design your grid and press **Train Agent**.")
        return

    c1, c2 = st.columns([2, 1])
    with c1:
        fig = plot_result(res)
        st.pyplot(fig)
    with c2:
        st.metric("Reached goal?", "✅ Yes" if res["solved"] else "❌ No")
        st.metric("Episode reward", f"{res['total_reward']:.1f}")
        st.metric("Steps to exit", len(res["path"]) - 1)
        st.divider()
        if "info_vi" in res:
            st.write("**Value Iteration**:", res["info_vi"]["iterations"], "iterations")
            st.write("**Policy Iteration**:", res["info_pi"]["iterations"], "iterations")
            st.write("Policies match:", "✅" if res["policies_match"] else "⚠️")
        else:
            info = res["info"]
            st.write(f"**{info['algorithm']}**:", info["episodes"], "episodes trained")
            st.write("Final ε (exploration):", f"{info['final_epsilon']:.3f}")
            last = info["reward_history"][-50:]
            st.write("Avg reward (last 50 ep):", f"{sum(last) / len(last):.1f}")
        next_room = room + 1
        if res["solved"] and st.session_state.unlocked_room >= next_room:
            st.success(f"🔓 Room {next_room} unlocked!")

    if "info" in res and res["info"].get("reward_history"):
        st.divider()
        st.subheader("📈 Learning curve")
        st.pyplot(plot_learning_curve(res["info"]))


ROOM3_FLAG_COMBOS = [
    (False, False, False),
    (True, False, False),
    (True, True, False),
    (True, True, True),
]
ROOM3_FLAG_LABELS = {
    (False, False, False): "○ Nothing collected yet",
    (True, False, False): "🔋 Battery only",
    (True, True, False): "🔋🔵 Battery + Blue switch",
    (True, True, True): "🔋🔵🔴 All three (ready to exit)",
}


def plot_room3_result(res, flags):
    env = res["env"]
    Q = res["Q"]
    bits = (int(flags[0]) << 2) | (int(flags[1]) << 1) | int(flags[2])
    rows, cols = env.rows, env.cols
    V = np.zeros((rows, cols))
    policy = np.zeros((rows, cols), dtype=int)
    for r in range(rows):
        for c in range(cols):
            idx = (r * cols + c) * 8 + bits
            V[r, c] = Q[idx].max()
            policy[r, c] = int(Q[idx].argmax())

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(V, cmap="viridis")
    fig.colorbar(im, ax=ax, label="V(s)", fraction=0.046)

    for r in range(rows):
        for c in range(cols):
            t = env.cfg.cell_type(r, c)
            if (r, c) == env.cfg.goal:
                ax.text(c, r, "G", ha="center", va="center", color="red", fontweight="bold")
            elif (r, c) == env.cfg.start:
                ax.text(c, r, "S", ha="center", va="center", color="lime", fontweight="bold")
            elif t == Cell.WALL:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="black"))
            elif t in ROOM3_LETTERS:
                letter, color = ROOM3_LETTERS[t]
                ax.text(c, r, letter, ha="center", va="center", color=color, fontweight="bold")
            else:
                a = policy[r, c]
                ax.text(c, r, ACTION_ARROWS[a], ha="center", va="center", color="white", fontsize=9)

    for (gr, gc) in set(env._guard_cycle):
        ax.add_patch(plt.Circle((gc, gr), 0.14, facecolor="none", edgecolor="magenta", linewidth=1.6, zorder=6))

    path_r = [p[0] for p in res["path"]]
    path_c = [p[1] for p in res["path"]]
    ax.plot(path_c, path_r, color="orange", linewidth=2, marker="o", markersize=3, zorder=5)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    return fig


def render_room3_results():
    st.subheader("📊 Results")
    res = st.session_state.results.get(3)
    if res is None:
        st.info("Design your grid and press **Train Agent**.")
        return

    c1, c2 = st.columns([2, 1])
    with c1:
        flags = st.selectbox(
            "Value/policy view",
            options=ROOM3_FLAG_COMBOS,
            format_func=lambda f: ROOM3_FLAG_LABELS[f],
        )
        st.pyplot(plot_room3_result(res, flags))
        st.caption("Magenta rings mark the guard robot's patrol cells.")
    with c2:
        st.metric("Reached goal?", "✅ Yes" if res["solved"] else "❌ No")
        st.metric("Episode reward", f"{res['total_reward']:.1f}")
        st.metric("Steps to exit", len(res["path"]) - 1)
        st.divider()
        info = res["info"]
        st.write(f"**{info['algorithm']}**:", info["episodes"], "episodes trained")
        st.write("Final ε (exploration):", f"{info['final_epsilon']:.3f}")
        last = info["reward_history"][-50:]
        st.write("Avg reward (last 50 ep):", f"{sum(last) / len(last):.1f}")
        if res["solved"] and st.session_state.unlocked_room >= 5:
            st.success("🔓 Room 5 unlocked!")

    st.divider()
    st.subheader("📈 Learning curve")
    st.pyplot(plot_learning_curve(res["info"]))


def plot_room5_result(res):
    env = res["env"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(plt.Rectangle((0, 0), 10, 10, fill=False, edgecolor="black", linewidth=1.5))

    for ox, oy in env.obstacles:
        half = env.obstacle_size / 2
        ax.add_patch(plt.Rectangle((ox - half, oy - half), env.obstacle_size, env.obstacle_size, color="black"))

    goal_x, goal_y = env.goal
    ax.add_patch(plt.Circle((goal_x, goal_y), env.goal_radius, color="#35ff8a", alpha=0.35))
    ax.text(goal_x, goal_y, "G", ha="center", va="center", color="darkgreen", fontweight="bold")

    start_x, start_y = env.start
    ax.text(start_x, start_y, "S", ha="center", va="center", color="lime", fontweight="bold",
            bbox=dict(boxstyle="circle", facecolor="black", edgecolor="lime"))

    xs = [p[0] for p in res["path"]]
    ys = [p[1] for p in res["path"]]
    ax.plot(xs, ys, color="orange", linewidth=2, zorder=4)
    ax.plot(xs[-1], ys[-1], marker="o", markersize=10, color="orange", markeredgecolor="black", zorder=5)

    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    return fig


def render_room5_results():
    st.subheader("📊 Results")
    res = st.session_state.results.get(5)
    if res is None:
        st.info("Set your parameters and press **Train Agent**.")
        return

    c1, c2 = st.columns([2, 1])
    with c1:
        st.pyplot(plot_room5_result(res))
        st.caption(
            "Evaluated greedily on a crate layout the training loop never saw -- "
            "this is the generalisation test the room is built around."
        )
    with c2:
        st.metric("Reached goal?", "✅ Yes" if res["solved"] else "❌ No")
        st.metric("Episode reward", f"{res['total_reward']:.1f}")
        st.metric("Decisions to exit", len(res["path"]) - 1)
        st.divider()
        info = res["info"]
        st.write(f"**{info['algorithm']}**:", info["episodes"], "episodes trained")
        st.write("Final ε (exploration):", f"{info['final_epsilon']:.3f}")
        last = info["reward_history"][-50:]
        st.write("Avg reward (last 50 ep):", f"{sum(last) / len(last):.1f}")
        if res["solved"]:
            st.success("🏆 Every built room cleared!")

    st.divider()
    st.subheader("📈 Learning curve")
    st.pyplot(plot_learning_curve(res["info"]))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def render_room_nav():
    cols = st.columns(len(ROOMS))
    for i, room in enumerate(ROOMS):
        with cols[i]:
            unlocked = room["id"] <= st.session_state.unlocked_room
            active = room["id"] == st.session_state.current_room
            icon = "🟢" if active else ("🔓" if unlocked else "🔒")
            label = f"{icon} Room {room['id']}\n{room['name']}"
            if st.button(label, key=f"room_nav_{room['id']}", disabled=not unlocked, use_container_width=True):
                st.session_state.current_room = room["id"]
                st.rerun()


ROOM_HEADERS = {
    1: (
        "### 🔴 Room 1 — The Laser Room",
        "A security lab has locked you in. Cameras dead, alarms silent — only the "
        "**control panel** at the far end can shut the system down. A direct line of "
        "**lasers** cuts straight across the floor: fast, but every beam you cross "
        "burns you. A longer corridor hugs the walls, mostly safe — but the floor "
        "there is **slick with coolant**, and it won't always take you where you meant "
        "to go. Plan the value function. Choose your route.",
    ),
    2: (
        "### 🌉 Room 2 — The Collapsing Bridge",
        "Whatever got you through the laser room is gone — nobody handed you a "
        "blueprint of this factory floor. All you can do is walk in, try a route, and "
        "learn from what happens. The same **walls**, **coolant floors** and hazards "
        "you can paint on the grid are still here, but this time there's no known "
        "model to plan against: **SARSA** has to feel its way to the control panel "
        "through trial and error, on-policy, living with the consequences of every "
        "risk it takes along the way.",
    ),
    3: (
        "### ⚡ Room 3 — The Energy Room",
        "The blast door won't budge on batteries alone. This is a machine room — "
        "cable-strewn floors, humming **generators** in blue and red, and a "
        "**guard robot** pacing a fixed beat down the only corridor south. Reaching "
        "the exit isn't the job: you need to find the **battery** tucked away from "
        "the direct route, throw the **blue switch** it powers, and only then can "
        "the **red switch** — and the door behind it — do anything at all. An "
        "**electric trap** flickers along the guard's corridor, live only every "
        "other step, and a battery-powered **shortcut** cuts through the wall for "
        "whoever's already found one. Q-Learning always assumes the best possible "
        "continuation, so it plans the full battery → blue → red → exit route even "
        "while it's still bumping into walls.",
    ),
    5: (
        "### 🏭 Room 5 — The Shifting Warehouse",
        "The last built sector is the loading warehouse, and its stacking robots "
        "never stopped working: 0.5-metre **crates** stand somewhere new at the "
        "start of every single run. There's no grid to memorise here, and no fixed "
        "map to plan against — the room is 10x10 **metres** of open floor, and you "
        "can't even see it: all you get is a short-range **radar**, a handful of "
        "distance readings around you. A table of Q-values has no row for "
        "'standing at (x=3.7, y=8.1)', so a **linear model** takes its place — "
        "weights over tile-coded position and speed, plus radar-distance bins, "
        "learned with semi-gradient Q-Learning. Learning a *route* is useless "
        "here; the only thing worth learning is a *rule* about what the radar "
        "says — and that's what lets it reach the exit in a crate layout it has "
        "never seen before.",
    ),
}


def render_room_header(room):
    header = ROOM_HEADERS.get(room)
    if header is None:
        return
    title, body = header
    st.markdown(title)
    st.caption(body)


def main():
    room = st.session_state.current_room
    if room == 3:
        inject_room3_css()
    elif room in (1, 2):
        inject_grid_css()
    else:
        inject_chrome_css()
    st.title("🗝️ Escape Room RL")
    st.caption("Design your environment, choose an algorithm, and watch the agent escape.")
    render_room_nav()
    st.divider()

    if room == 5:
        params, train_clicked = render_room5_sidebar()
        render_room_header(room)
        if train_clicked:
            with st.spinner(f"Training for {params['episodes']} episodes..."):
                train_room5(params)
            st.rerun()
        render_room5_results()
        return

    params, train_clicked = render_sidebar(room)

    if room in (1, 2):
        render_room_header(room)
        render_grid_editor()
        if train_clicked:
            (train_room1 if room == 1 else train_room2)(params)
            st.rerun()
        render_results(room)
    elif room == 3:
        render_room_header(room)
        render_room3_grid_editor()
        if train_clicked:
            train_room3(params)
            st.rerun()
        render_room3_results()
    else:
        st.info(f"Room {room} — {ROOMS[room-1]['name']} — 🚧 coming soon, we'll build it next.")


if __name__ == "__main__":
    main()
