"""
Escape Room RL — unified Streamlit app.

One app, five rooms. You configure the grid/parameters for the active room,
train the agent, and if it solves the room, the next one unlocks — exactly
like a real escape room.

Run locally with:
    streamlit run app.py
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from grid_env import GridConfig, GridWorldEnv, Cell, ACTION_ARROWS, default_config
from dp_solver import value_iteration, policy_iteration


st.set_page_config(page_title="Escape Room RL", page_icon="🗝️", layout="wide")

TOOL_LABELS = {
    Cell.EMPTY: "· Empty",
    Cell.WALL: "■ Wall",
    Cell.SLIPPERY: "≈ Slippery",
    Cell.GOAL: "🏁 Goal",
    Cell.TRAP: "☠ Trap",
    Cell.START: "🚦 Start",
}
CELL_ICON = {
    Cell.EMPTY: "",
    Cell.WALL: "🧱",
    Cell.SLIPPERY: "≈",
    Cell.GOAL: "🏁",
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
    st.session_state.grid_cells = {}
    st.session_state.cell_rewards = {}
    default = default_config(10, 10)
    st.session_state.grid_cells = dict(default.cells)
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


def build_config():
    cfg = GridConfig(
        rows=st.session_state.rows,
        cols=st.session_state.cols,
        cells=dict(st.session_state.grid_cells),
        cell_rewards=dict(st.session_state.cell_rewards),
        start=st.session_state.start,
        goal=st.session_state.goal,
    )
    return cfg


def cell_label(r, c):
    if (r, c) == st.session_state.start:
        return "🚦"
    if (r, c) == st.session_state.goal:
        return "🏁"
    t = st.session_state.grid_cells.get((r, c), Cell.EMPTY)
    return CELL_ICON.get(t, "")


def render_grid_editor():
    st.subheader("📖 Grid Editor")
    st.caption(f"Active tool: **{TOOL_LABELS[st.session_state.paint_tool]}** — click a cell to paint")
    for r in range(st.session_state.rows):
        cols = st.columns(st.session_state.cols)
        for c in range(st.session_state.cols):
            with cols[c]:
                st.button(
                    cell_label(r, c) or " ",
                    key=f"cell_{r}_{c}",
                    on_click=paint_cell,
                    args=(r, c),
                    use_container_width=True,
                )


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
            st.rerun()
        if b2.button("Clear", use_container_width=True):
            st.session_state.grid_cells = {st.session_state.goal: Cell.GOAL}
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

        st.markdown("**Step reward**")
        step_reward = st.number_input("Step", value=-1.0, step=0.1, key="step_reward")
        shaping = st.checkbox("🧭 Potential-based goal shaping", value=True, key="potential_shaping")

        st.divider()
        st.subheader("Dynamics")
        slip_prob = st.slider("Slip probability", 0.0, 1.0, 0.3, key="slip_prob")
        gamma = st.slider("Discount γ", 0.0, 0.999, 0.95, key="gamma")

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
        gamma=params["gamma"],
        potential_shaping=params["shaping"],
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


def main():
    st.title("🗝️ Escape Room RL")
    st.caption("Design your environment, choose an algorithm, and watch the agent escape.")
    render_room_nav()
    st.divider()

    room = st.session_state.current_room
    params, train_clicked = render_sidebar(room)

    if room == 1:
        st.caption(f"**Room 1 — {ROOMS[0]['name']}** ({ROOMS[0]['subtitle']})")
        render_grid_editor()
        if train_clicked:
            train_room1(params)
        render_results()
    else:
        st.info(f"Room {room} — {ROOMS[room-1]['name']} — 🚧 coming soon, we'll build it next.")


if __name__ == "__main__":
    main()
