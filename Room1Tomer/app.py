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
    GridConfig, GridWorldEnv, Cell, ACTION_ARROWS, default_config,
    SLIP_DIR_MODES, SLIP_DIR_LABELS,
)
from dp_solver import value_iteration, policy_iteration


if not st.session_state.get("_embedded"):
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


init_state()


# ---------------------------------------------------------------------------
# grid editing
# ---------------------------------------------------------------------------
def paint_cell(r, c):
    if "start" not in st.session_state:
        return  # stale on_click firing before this room's state is (re)initialized
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
[class*="st-key-cellwrap_"] button {
    font-size: 1.75rem !important;
    line-height: 1.1 !important;
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


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------
def render_sidebar(current_room):
    with st.sidebar:
        st.header("🎮 Game Setup")

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


def render_results():
    st.subheader("📊 Results")
    res = st.session_state.results.get(1)
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
        st.write("**Value Iteration**:", res["info_vi"]["iterations"], "iterations")
        st.write("**Policy Iteration**:", res["info_pi"]["iterations"], "iterations")
        st.write("Policies match:", "✅" if res["policies_match"] else "⚠️")
        if res["solved"] and st.session_state.unlocked_room >= 2:
            st.success("🔓 Room 2 unlocked!")


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


def render_room1_header():
    st.markdown("### 🔴 Room 1 — The Laser Room")
    st.caption(
        "A security lab has locked you in. Cameras dead, alarms silent — only the "
        "**control panel** at the far end can shut the system down. A direct line of "
        "**lasers** cuts straight across the floor: fast, but every beam you cross "
        "burns you. A longer corridor hugs the walls, mostly safe — but the floor "
        "there is **slick with coolant**, and it won't always take you where you meant "
        "to go. Plan the value function. Choose your route."
    )


def main():
    inject_grid_css()
    embedded = st.session_state.get("_embedded", False)
    if not embedded:
        st.title("🗝️ Escape Room RL")
        st.caption("Design your environment, choose an algorithm, and watch the agent escape.")
        render_room_nav()
        st.divider()

    # When embedded in the unified router, the router already knows this is
    # room 1 -- avoid touching "current_room"/"unlocked_room" at all here,
    # since those names collide with the router's own navigation state.
    room = 1 if embedded else st.session_state.current_room
    is_training = st.session_state.get("is_training", False)

    if is_training:
        with st.sidebar:
            st.header("🎮 Game Setup")
            st.warning("🔒 Training in progress — parameters are locked until it finishes.")
        params = st.session_state.get("_locked_params", {})
    else:
        params, train_clicked = render_sidebar(room)
        if train_clicked:
            st.session_state["is_training"] = True
            st.session_state["_locked_params"] = params
            st.rerun()

    if room == 1:
        render_room1_header()
        render_grid_editor()
        if is_training:
            train_room1(params)
            st.session_state["is_training"] = False
            st.rerun()
        render_results()
    else:
        st.info(f"Room {room} — {ROOMS[room-1]['name']} — 🚧 coming soon, we'll build it next.")


if __name__ == "__main__":
    main()
