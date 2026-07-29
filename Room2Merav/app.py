"""
Room 2 — SARSA, unknown model, slippery grid.

Standalone Streamlit app for Room 2: configure hyperparameters, train the
SARSA agent, inspect learning curves, and replay individual episodes.

Run locally with:
    streamlit run app.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import streamlit as st

from grid_env import CellOverride
from room2_env import ROOM2_LAYOUT
from train_room2 import TrainRoom2Config, train
from viz import plot_learning_curves, render_grid, render_episode_step

st.set_page_config(page_title="Room 2 — The Collapsing Bridge (SARSA)", page_icon="🌉", layout="wide")

DEFAULTS = TrainRoom2Config()


def init_state():
    if "initialized" in st.session_state:
        return
    st.session_state.initialized = True
    st.session_state.history = None
    st.session_state.trajectories = None
    st.session_state.trained_cfg = None
    st.session_state.q_table = None
    st.session_state.cell_overrides = {}
    st.session_state.paint_tool = "none"


init_state()

# S/K/G are structurally tied to the state machine (start position, the
# has_key flag, the key-gated exit) -- painting over them would silently
# break the room, so the grid editor refuses to touch them.
PROTECTED_CELLS = {
    (r, c) for r, row in enumerate(ROOM2_LAYOUT) for c, ch in enumerate(row) if ch in ("S", "K", "G")
}

PAINT_TOOLS = {
    "none": "🚫 None (inspect only)",
    "slippery": "≈ Slippery (custom %)",
    "reward": "⭐ Reward (custom value)",
    "terminal": "☠ Terminal (custom value)",
    "clear": "🧹 Clear override",
}


def paint_cell(r: int, c: int) -> None:
    if (r, c) in PROTECTED_CELLS:
        st.toast("Can't override the start, key, or door cells.", icon="🚫")
        return
    tool = st.session_state.paint_tool
    if tool == "none":
        return
    if tool == "clear":
        st.session_state.cell_overrides.pop((r, c), None)
        return
    existing = st.session_state.cell_overrides.get((r, c), CellOverride())
    if tool == "slippery":
        existing = CellOverride(
            slip_prob=st.session_state.paint_slip_prob, reward=existing.reward, terminal_reward=existing.terminal_reward
        )
    elif tool == "reward":
        existing = CellOverride(
            slip_prob=existing.slip_prob, reward=st.session_state.paint_reward_value, terminal_reward=existing.terminal_reward
        )
    elif tool == "terminal":
        existing = CellOverride(
            slip_prob=existing.slip_prob, reward=existing.reward, terminal_reward=st.session_state.paint_terminal_value
        )
    st.session_state.cell_overrides[(r, c)] = existing


LEGEND_ITEMS = [
    ("🚦", "Start — where you wake up"),
    ("🔑", "Key — required before the door opens, no way around it"),
    ("🏁", "Door — the exit, locked without the key"),
    ("🧱", "Wall — factory machinery, impassable"),
    ("≈", "Slippery floor — a move here may slip sideways"),
    ("☠", "The abyss — one step in, episode over"),
    ("🌉", "The bridge — slippery too, every crossing risks falling in"),
]


def render_legend() -> None:
    st.markdown("#### 🗺️ Legend")
    for icon, text in LEGEND_ITEMS:
        st.markdown(f"- {icon} {text}")


BASE_SYMBOL_ICON = {"S": "🚦", "K": "🔑", "G": "🏁", "#": "🧱", "P": "☠", "B": "🌉", "~": "≈", ".": "·"}


def cell_button_label(r: int, c: int) -> str:
    label = BASE_SYMBOL_ICON.get(ROOM2_LAYOUT[r][c], "·")
    override = st.session_state.cell_overrides.get((r, c))
    if override is None:
        return label
    parts = [label]
    if override.slip_prob is not None:
        parts.append(f"≈{override.slip_prob:.0%}")
    if override.reward is not None:
        parts.append(f"+{override.reward:g}" if override.reward >= 0 else f"{override.reward:g}")
    if override.terminal_reward is not None:
        parts.append(f"☠{override.terminal_reward:g}")
    return "\n".join(parts)


def render_grid_editor() -> None:
    st.markdown("### 🎨 Grid editor")
    st.caption(
        "Customize any cell (except start/key/door): make it slippery with its own "
        "probability, give it a repeatable reward, or make it a terminal state with "
        "its own reward — independent of the room's base layout."
    )

    tool = st.radio(
        "Paint tool",
        options=list(PAINT_TOOLS.keys()),
        format_func=lambda t: PAINT_TOOLS[t],
        key="paint_tool",
        horizontal=True,
    )
    if tool == "slippery":
        st.slider("Slip probability to paint", 0.0, 1.0, 0.2, step=0.05, key="paint_slip_prob")
    elif tool == "reward":
        st.number_input("Reward value to paint (every visit)", value=5.0, step=1.0, key="paint_reward_value")
    elif tool == "terminal":
        st.number_input("Terminal reward to paint (ends the episode)", value=-20.0, step=5.0, key="paint_terminal_value")

    for r in range(10):
        cols = st.columns(10)
        for c in range(10):
            with cols[c]:
                st.button(
                    cell_button_label(r, c),
                    key=f"paint_{r}_{c}",
                    on_click=paint_cell,
                    args=(r, c),
                    use_container_width=True,
                    disabled=(r, c) in PROTECTED_CELLS,
                )

    if st.session_state.cell_overrides:
        c1, c2 = st.columns([3, 1])
        c1.caption(f"{len(st.session_state.cell_overrides)} custom cell(s) painted.")
        if c2.button("Clear all", use_container_width=True):
            st.session_state.cell_overrides = {}
            st.rerun()


def render_sidebar() -> TrainRoom2Config:
    with st.sidebar:
        st.header("🎮 Game Setup")

        st.subheader("SARSA hyperparameters")
        alpha = st.slider("α · learning rate", 0.01, 1.0, DEFAULTS.alpha, step=0.01)
        gamma = st.slider("γ · discount factor", 0.0, 0.999, DEFAULTS.gamma, step=0.01)
        epsilon_start = st.slider("ε₀ · initial exploration", 0.0, 1.0, DEFAULTS.epsilon_start, step=0.05)
        epsilon_min = st.slider("ε_min · minimum exploration", 0.0, 0.5, DEFAULTS.epsilon_min, step=0.01)
        epsilon_decay = st.slider(
            "ε decay · exploration decay/episode", 0.90, 0.9999, DEFAULTS.epsilon_decay, step=0.0005, format="%.4f"
        )

        st.divider()
        st.subheader("Environment")
        slip_prob = st.slider("Slip probability", 0.0, 1.0, DEFAULTS.slip_prob, step=0.05)
        max_steps = st.number_input("Max steps / episode", 20, 1000, DEFAULTS.max_steps, step=10)

        st.divider()
        st.subheader("Training run")
        episodes = st.number_input("Episodes", 100, 20000, DEFAULTS.episodes, step=100)
        seed = st.number_input("Seed", 0, 10_000, DEFAULTS.seed, step=1)

        train_clicked = st.button("▶ Train Agent", type="primary", use_container_width=True)

        st.divider()
        render_legend()

        cfg = TrainRoom2Config(
            episodes=int(episodes),
            max_steps=int(max_steps),
            alpha=alpha,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            slip_prob=slip_prob,
            seed=int(seed),
            results_dir="results/room2",
        )
        return cfg, train_clicked


def render_metrics(history: list[dict], window: int = 50):
    last_window = history[-window:] if len(history) >= window else history
    success_rate = sum(h["success"] for h in last_window) / len(last_window)
    successful_steps = [h["steps"] for h in last_window if h["success"]]
    avg_steps = sum(successful_steps) / len(successful_steps) if successful_steps else float("nan")

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Success rate (last {len(last_window)} ep)", f"{success_rate:.0%}")
    c2.metric("Avg steps to exit (successes)", f"{avg_steps:.1f}" if successful_steps else "—")
    c3.metric("Final ε", f"{history[-1]['epsilon']:.3f}")


def render_room_briefing() -> None:
    st.markdown("### 📋 Mission briefing")
    st.markdown("- **משימה:** קח את המפתח ואז פתח את הדלת.")
    st.markdown("- **סכנות:** תהום, רצפה חלקלקה, וגשר חלקלק שעלול לקרוס בכל חצייה.")
    st.markdown("- **עיקרון:** SARSA לומד להימנע מסיכונים בסביבה חלקלקה.")


def render_replay(trajectories: dict, q_table, has_key: bool):
    st.subheader("Episode replay")
    if not trajectories:
        st.info("No recorded episodes.")
        return
    ep_keys = sorted(trajectories.keys())
    labels = {ep: f"Episode {ep}" for ep in ep_keys}
    chosen = st.selectbox("Pick a recorded episode", ep_keys, format_func=lambda e: labels[e])
    trajectory = trajectories[chosen]
    step = st.slider("Step", 0, len(trajectory) - 1, len(trajectory) - 1)
    st.pyplot(
        render_episode_step(
            ROOM2_LAYOUT,
            trajectory,
            step,
            cell_overrides=st.session_state.cell_overrides,
            q_table=q_table,
            has_key=has_key,
        )
    )


def main():
    st.title("🌉 Room 2 — The Collapsing Bridge (SARSA)")
    st.caption("Model unknown, on-policy TD control, slippery grid.")

    cfg, train_clicked = render_sidebar()

    if train_clicked:
        with st.spinner(f"Training SARSA for {cfg.episodes} episodes..."):
            agent, history, trajectories = train(cfg, cell_overrides=st.session_state.cell_overrides)
        st.session_state.history = history
        st.session_state.trajectories = trajectories
        st.session_state.trained_cfg = cfg
        st.session_state.q_table = agent.q

    has_key_view = False
    if st.session_state.q_table is not None:
        has_key_view = st.radio(
            "Value/policy view", options=[False, True], format_func=lambda k: "Before key" if not k else "After key",
            horizontal=True,
        )

    col_grid, col_info = st.columns([2, 1])
    with col_grid:
        st.subheader("Room layout")
        st.pyplot(
            render_grid(
                ROOM2_LAYOUT,
                title="Room 2: The Collapsing Bridge",
                cell_overrides=st.session_state.cell_overrides,
                q_table=st.session_state.q_table,
                has_key=has_key_view,
            )
        )
        st.caption("See the 🗺️ Legend in the sidebar for what each icon means.")

    with st.expander("🎨 Grid editor — customize any cell", expanded=False):
        render_grid_editor()

    with col_info:
        if st.session_state.history is None:
            st.info("Set your hyperparameters and press **Train Agent**.")
        else:
            render_metrics(st.session_state.history)

        st.divider()
        render_room_briefing()

    if st.session_state.history is not None:
        st.divider()
        st.subheader("Learning curves")
        st.pyplot(plot_learning_curves(st.session_state.history))

        st.divider()
        render_replay(st.session_state.trajectories, st.session_state.q_table, has_key_view)


if __name__ == "__main__":
    main()
