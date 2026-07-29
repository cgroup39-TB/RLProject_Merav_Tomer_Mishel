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

UNIQUE_SYMBOLS = ("S", "K", "G", "B")  # exactly one of each required by GridWorld

if not st.session_state.get("_embedded"):
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
    st.session_state.layout_grid = [list(row) for row in ROOM2_LAYOUT]
    st.session_state.trained_layout = None


init_state()

# Base layout symbols paintable via dedicated tools (move S/K/G/B, or set
# a plain terrain symbol), separate from the per-cell override tools below
# (which layer custom slip/reward/terminal behavior on top of any symbol).
LAYOUT_TOOLS = {
    "wall": "🧱 Wall",
    "pit": "☠ Pit (abyss)",
    "slippery_floor": "≈ Slippery floor",
    "empty": "· Empty floor",
    "start": "🚦 Start",
    "key": "🔑 Key",
    "goal": "🏁 Door (goal)",
    "bridge": "🌉 Bridge",
}
LAYOUT_SYMBOL = {
    "wall": "#", "pit": "P", "slippery_floor": "~", "empty": ".",
    "start": "S", "key": "K", "goal": "G", "bridge": "B",
}
PAINT_TOOLS = {
    "none": "🚫 None (inspect only)",
    **LAYOUT_TOOLS,
    "slippery": "≈ Slippery (custom %)",
    "reward": "⭐ Reward (custom value)",
    "clear": "🧹 Clear override (this cell)",
}
# A bridge's "orientation" isn't a stored property (unlike Room 4's moving
# gates, a bridge is just a slippery cell) -- it only determines which two
# neighbor cells get auto-painted as pits at paint time, matching whichever
# direction you're actually crossing it.
BRIDGE_ORIENTATIONS = {
    "vertical": "↕ Vertical crossing (pits left/right)",
    "horizontal": "↔ Horizontal crossing (pits above/below)",
}
BRIDGE_NEIGHBOR_DELTAS = {
    "vertical": [(0, -1), (0, 1)],
    "horizontal": [(-1, 0), (1, 0)],
}


def paint_cell(r: int, c: int) -> None:
    if "cell_overrides" not in st.session_state:
        return  # stale on_click firing before this room's state is (re)initialized
    tool = st.session_state.paint_tool
    if tool == "none":
        return

    grid = st.session_state.layout_grid
    current_symbol = grid[r][c]

    if tool in LAYOUT_TOOLS:
        new_symbol = LAYOUT_SYMBOL[tool]
        if current_symbol in UNIQUE_SYMBOLS and new_symbol != current_symbol:
            st.toast(
                f"That's the room's only '{current_symbol}' cell — move it with its own "
                f"tool instead of painting over it.", icon="🚫",
            )
            return
        if new_symbol in UNIQUE_SYMBOLS:
            # relocate: clear whichever cell currently holds this unique symbol
            for rr in range(len(grid)):
                for cc in range(len(grid[0])):
                    if grid[rr][cc] == new_symbol:
                        grid[rr][cc] = "."
        grid[r][c] = new_symbol

        if tool == "bridge":
            # a bridge with nothing but floor beside it is just a slippery
            # tile with no real risk -- auto-paint pits on both sides (per
            # the chosen orientation) so the danger is real without you
            # having to know to place them yourself
            orientation = st.session_state.get("paint_bridge_orientation", "vertical")
            skipped = []
            for dr, dc in BRIDGE_NEIGHBOR_DELTAS[orientation]:
                rr, cc = r + dr, c + dc
                if not (0 <= rr < len(grid) and 0 <= cc < len(grid[0])):
                    continue
                if grid[rr][cc] in UNIQUE_SYMBOLS:
                    skipped.append((rr, cc))
                    continue
                grid[rr][cc] = "P"
            if skipped:
                st.toast(f"Left {skipped} alone (unique cell) — pit not auto-placed there.", icon="⚠️")
        return

    # per-cell override tools (unchanged behavior, layered on top of the symbol)
    if tool == "clear":
        had_override = st.session_state.cell_overrides.pop((r, c), None) is not None
        st.toast(
            f"Cleared override at ({r},{c})." if had_override else f"({r},{c}) had no override to clear.",
            icon="🧹" if had_override else "ℹ️",
        )
        return
    existing = st.session_state.cell_overrides.get((r, c), CellOverride())
    if tool == "slippery":
        existing = CellOverride(slip_prob=st.session_state.paint_slip_prob, reward=existing.reward)
    elif tool == "reward":
        existing = CellOverride(slip_prob=existing.slip_prob, reward=st.session_state.paint_reward_value)
    st.session_state.cell_overrides[(r, c)] = existing


def current_layout() -> tuple[str, ...]:
    return tuple("".join(row) for row in st.session_state.layout_grid)


LEGEND_ITEMS = [
    ("🚦", "Start — where you wake up"),
    ("🔑", "Key — required before the door opens, no way around it"),
    ("🏁", "Door — the exit, locked without the key"),
    ("🧱", "Wall — factory machinery, impassable"),
    ("≈", "Slippery floor — a move here may slip sideways"),
    ("☠", "The abyss — one step in, episode over"),
    ("🌉", "The bridge — really just a slippery floor tile (≈) sitting over the abyss, not a "
           "structure with two ends: crossing it normally moves you one cell at a time like any "
           "floor, but on this particular tile, slipping sideways always drops you straight into "
           "an abyss cell right next to it — no need to mark that separately, it's automatic"),
]


def render_legend() -> None:
    st.markdown("#### 🗺️ Legend")
    for icon, text in LEGEND_ITEMS:
        st.markdown(f"- {icon} {text}")


BASE_SYMBOL_ICON = {"S": "🚦", "K": "🔑", "G": "🏁", "#": "🧱", "P": "☠", "B": "🌉", "~": "≈", ".": "·"}


def cell_button_label(r: int, c: int) -> str:
    label = BASE_SYMBOL_ICON.get(st.session_state.layout_grid[r][c], "·")
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
        "Paint the room's layout (wall / pit / slippery / start / key / door / bridge — "
        "moving a unique cell like start or key relocates it), or layer a per-cell "
        "override (custom slip % or a repeatable reward) on top of any cell's base symbol. "
        "Pit ('☠') is the only terminal (episode-ending) hazard."
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
    elif tool == "bridge":
        st.radio(
            "Bridge orientation",
            options=list(BRIDGE_ORIENTATIONS.keys()),
            format_func=lambda o: BRIDGE_ORIENTATIONS[o],
            key="paint_bridge_orientation",
            horizontal=True,
        )
        st.caption("Placing the bridge automatically paints a Pit ('☠') on both sides, "
                   "so it's a real risky crossing without having to place them yourself.")

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
                )

    n_overrides = len(st.session_state.cell_overrides)
    st.caption(f"{n_overrides} custom slip%/reward override(s) painted.")
    if st.button("🔄 Reset room to default", use_container_width=True):
        st.session_state.layout_grid = [list(row) for row in ROOM2_LAYOUT]
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
        st.subheader("Reward shaping")
        pit_reward = st.number_input("Pit (abyss) penalty", value=DEFAULTS.pit_reward, step=5.0)
        key_bonus = st.number_input("🔑 Key bonus", value=DEFAULTS.key_bonus, step=1.0)
        goal_base_reward = st.number_input("Door base reward", value=DEFAULTS.goal_base_reward, step=5.0)
        goal_decay_per_step = st.number_input(
            "Door reward decay / step", value=DEFAULTS.goal_decay_per_step, step=0.1,
            help="The door's reward shrinks by this much per step taken -- faster escapes score higher.",
        )
        goal_min_reward = st.number_input("Door minimum reward", value=DEFAULTS.goal_min_reward, step=5.0)

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
            pit_reward=pit_reward,
            key_bonus=key_bonus,
            goal_base_reward=goal_base_reward,
            goal_decay_per_step=goal_decay_per_step,
            goal_min_reward=goal_min_reward,
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
    # default to the LAST recorded episode, not the first -- early episodes
    # are still near-random and usually never reach the key, which made it
    # look like the "after key" path was missing/broken
    chosen = st.selectbox(
        "Pick a recorded episode", ep_keys, index=len(ep_keys) - 1, format_func=lambda e: labels[e]
    )
    trajectory = trajectories[chosen]
    step = st.slider("Step", 0, len(trajectory) - 1, len(trajectory) - 1)
    st.pyplot(
        render_episode_step(
            st.session_state.trained_layout or current_layout(),
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

    is_training = st.session_state.get("is_training", False)
    if is_training:
        with st.sidebar:
            st.header("🎮 Game Setup")
            st.warning("🔒 Training in progress — parameters are locked until it finishes.")
        cfg = st.session_state.get("_locked_cfg")
    else:
        cfg, train_clicked = render_sidebar()
        if train_clicked:
            st.session_state["is_training"] = True
            st.session_state["_locked_cfg"] = cfg
            st.session_state["_locked_layout"] = current_layout()
            st.rerun()

    if is_training and cfg is not None:
        layout = st.session_state.get("_locked_layout")
        with st.spinner(f"Training SARSA for {cfg.episodes} episodes..."):
            agent, history, trajectories = train(cfg, cell_overrides=st.session_state.cell_overrides, layout=layout)
        st.session_state.history = history
        st.session_state.trajectories = trajectories
        st.session_state.trained_cfg = cfg
        st.session_state.trained_layout = layout
        st.session_state.q_table = agent.q
        st.session_state["is_training"] = False
        st.rerun()

    with st.expander("🎨 Grid editor — design the room", expanded=True):
        render_grid_editor()
    st.caption("See the 🗺️ Legend in the sidebar for what each icon means.")

    col_info, col_briefing = st.columns([1, 1])
    with col_info:
        if st.session_state.history is None:
            st.info("Set your hyperparameters and press **Train Agent**.")
        else:
            render_metrics(st.session_state.history)
    with col_briefing:
        render_room_briefing()

    has_key_view = False
    if st.session_state.q_table is not None:
        # only worth a separate image once it shows something the grid
        # editor above doesn't: the value heatmap + learned policy arrows --
        # shown below the editor, once training has actually produced one
        st.divider()
        st.subheader("Value function + learned policy")
        has_key_view = st.radio(
            "Value/policy view", options=[False, True], format_func=lambda k: "Before key" if not k else "After key",
            horizontal=True,
        )
        display_layout = st.session_state.trained_layout or current_layout()
        st.pyplot(
            render_grid(
                display_layout,
                title="Room 2: The Collapsing Bridge",
                cell_overrides=st.session_state.cell_overrides,
                q_table=st.session_state.q_table,
                has_key=has_key_view,
            )
        )

        st.divider()
        st.subheader("Learning curves")
        st.pyplot(plot_learning_curves(st.session_state.history))

        st.divider()
        render_replay(st.session_state.trajectories, st.session_state.q_table, has_key_view)


if __name__ == "__main__":
    main()
