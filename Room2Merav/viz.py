"""Plotting helpers for Room 2: learning-curve charts and grid rendering.

Styled to match Room 1's DP visualization, so the merged multi-room app
looks like one consistent product rather than a patchwork: a light
background, a viridis heatmap of the learned value function (Room 1's
V(s), here SARSA's V(s) = max_a Q(s,a)), plain bold colored letters for
special cells, black squares for walls, and an orange path line with
small circle markers. Room 2 has two things Room 1 doesn't -- a has_key
state split (so the heatmap/policy are shown per key-state) and a couple
of extra cell types (the key, the bridge) -- those get the same "bold
colored letter" treatment for consistency, in colors that don't collide
with Room 1's existing S/G/trap palette.

Kept as plain matplotlib functions (no Streamlit dependency) so they can
be unit-tested/used standalone; app.py just calls these and hands the
returned Figure to st.pyplot().
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Room 1's exact conventions (grid_env.py / app.py in Room1Tomer)
WALL_COLOR = "black"
PATH_COLOR = "orange"
ARROW_COLOR_ON_HEATMAP = "white"
ARROW_COLOR_PLAIN = "dimgray"

# (letter, color) for cells Room 1 already has (S/G/trap) plus Room 2's
# own additions (key, bridge), all in the same bold-colored-letter style.
CELL_LETTER = {
    "S": ("S", "lime"),
    "G": ("G", "red"),
    "P": ("X", "orangered"),   # abyss -- rendered as Room 1's trap letter
    "K": ("K", "gold"),
    "B": ("B", "deepskyblue"),
}

# Room 2's action encoding (grid_env.ACTIONS): 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
ACTION_ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}

OVERRIDE_BADGE_COLORS = {
    "slip_prob": "#1f77b4",       # blue -- matches the viridis family
    "reward": "#2ca02c",          # green -- a bonus
    "terminal_reward": "#d62728",  # red -- ends the episode
}


def _draw_override_badges(ax, cx: float, cy: float, override) -> None:
    """Small corner squares marking which per-cell overrides are active.

    Room 1 has no equivalent (its cell_rewards dict isn't visualized on
    the grid either), so there's no existing convention to match here --
    these just need to read clearly against the light background.
    """
    active = [
        key
        for key in ("slip_prob", "reward", "terminal_reward")
        if getattr(override, key, None) is not None
    ]
    for i, key in enumerate(active):
        ax.add_patch(
            plt.Rectangle(
                (cx - 0.48 + i * 0.14, cy + 0.34),
                0.12,
                0.12,
                facecolor=OVERRIDE_BADGE_COLORS[key],
                edgecolor="white",
                linewidth=0.5,
                zorder=7,
            )
        )


def _value_and_policy(q_table: np.ndarray, n_rows: int, n_cols: int, has_key: bool):
    """V(s) = max_a Q(s,a) and greedy policy, reshaped to the grid, for one has_key slice."""
    value = np.zeros((n_rows, n_cols))
    policy = np.zeros((n_rows, n_cols), dtype=int)
    for r in range(n_rows):
        for c in range(n_cols):
            idx = (r * n_cols + c) * 2 + int(has_key)
            value[r, c] = q_table[idx].max()
            policy[r, c] = int(q_table[idx].argmax())
    return value, policy


def _rolling_mean(values: list[float], window: int) -> list[float]:
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def plot_learning_curves(history: list[dict], window: int = 50) -> Figure:
    """4-panel figure: return, steps, epsilon, and rolling success rate per episode."""
    episodes = [h["episode"] for h in history]
    returns = [h["return"] for h in history]
    steps = [h["steps"] for h in history]
    epsilons = [h["epsilon"] for h in history]
    successes = [float(h["success"]) for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    ax = axes[0, 0]
    ax.plot(episodes, returns, color="gray", alpha=0.3, linewidth=0.7, label="return")
    ax.plot(episodes, _rolling_mean(returns, window), color="orange", linewidth=2, label=f"rolling mean ({window})")
    ax.set_title("Episode return")
    ax.set_xlabel("episode")
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[0, 1]
    ax.plot(episodes, steps, color="gray", alpha=0.3, linewidth=0.7, label="steps")
    ax.plot(episodes, _rolling_mean(steps, window), color="orange", linewidth=2, label=f"rolling mean ({window})")
    ax.set_title("Steps to terminal state")
    ax.set_xlabel("episode")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1, 0]
    ax.plot(episodes, epsilons, color="orange")
    ax.set_title("Epsilon (exploration rate)")
    ax.set_xlabel("episode")

    ax = axes[1, 1]
    ax.plot(episodes, _rolling_mean(successes, window), color="green")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Rolling success rate (window={window})")
    ax.set_xlabel("episode")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def trajectory_to_positions(trajectory: list[dict], n_cols: int = 10) -> list[tuple[int, int]]:
    """Convert a recorded trajectory's flat state indices to (row, col) coords.

    State indices encode (row, col, has_key) as (row*n_cols+col)*2+has_key
    (see grid_env.GridWorld.encode_state) -- the has_key bit is dropped
    here since rendering only needs position.
    """
    positions = []
    for step in trajectory:
        pos_index = step["state"] // 2
        positions.append(divmod(pos_index, n_cols))
    return positions


def render_grid(
    layout: tuple[str, ...],
    path: list[tuple[int, int]] | None = None,
    agent_pos: tuple[int, int] | None = None,
    title: str = "",
    cell_overrides: dict | None = None,
    q_table: np.ndarray | None = None,
    has_key: bool = False,
) -> Figure:
    """Render the grid, matching Room 1's plot_result style.

    Without q_table: a plain light-background preview -- colored letters
    for special cells, black wall squares -- for showing the room before
    training. With q_table: a viridis heatmap of V(s) = max_a Q(s,a) for
    the given has_key slice, plus a greedy-policy arrow on every ordinary
    cell, exactly like Room 1's post-training plot.

    cell_overrides (optional) marks cells with a custom slip probability,
    reward, and/or terminal state (see grid_env.CellOverride) with small
    corner badges, independent of the cell's base layout symbol.
    """
    n_rows = len(layout)
    n_cols = len(layout[0])
    cell_overrides = cell_overrides or {}

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    value = policy = None
    if q_table is not None:
        value, policy = _value_and_policy(q_table, n_rows, n_cols, has_key)
        im = ax.imshow(value, cmap="viridis")
        fig.colorbar(im, ax=ax, label="V(s)", fraction=0.046)

    for r in range(n_rows):
        for c in range(n_cols):
            symbol = layout[r][c]
            if symbol == "#":
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=WALL_COLOR))
            elif symbol in CELL_LETTER:
                letter, color = CELL_LETTER[symbol]
                ax.text(c, r, letter, ha="center", va="center", color=color, fontweight="bold")
            elif value is not None:
                arrow_color = ARROW_COLOR_ON_HEATMAP if q_table is not None else ARROW_COLOR_PLAIN
                ax.text(c, r, ACTION_ARROWS[policy[r, c]], ha="center", va="center", color=arrow_color, fontsize=9)
            override = cell_overrides.get((r, c))
            if override is not None:
                _draw_override_badges(ax, c, r, override)

    if path:
        xs = [c for _, c in path]
        ys = [r for r, _ in path]
        ax.plot(xs, ys, color=PATH_COLOR, linewidth=2, marker="o", markersize=3, zorder=4)

    if agent_pos:
        r, c = agent_pos
        ax.plot(c, r, marker="o", markersize=12, color=PATH_COLOR, markeredgecolor="black", markeredgewidth=1.2, zorder=5)

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def render_episode_step(
    layout: tuple[str, ...],
    trajectory: list[dict],
    step_index: int,
    n_cols: int = 10,
    cell_overrides: dict | None = None,
    q_table: np.ndarray | None = None,
    has_key: bool = False,
) -> Figure:
    """Render the grid with the path walked up to (and agent at) step_index."""
    positions = trajectory_to_positions(trajectory, n_cols=n_cols)
    step_index = max(0, min(step_index, len(positions) - 1))
    return render_grid(
        layout,
        path=positions[: step_index + 1],
        agent_pos=positions[step_index],
        title=f"Step {step_index} / {len(positions) - 1}",
        cell_overrides=cell_overrides,
        q_table=q_table,
        has_key=has_key,
    )
