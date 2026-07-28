"""Plotting helpers for Room 2: learning-curve charts and episode replay.

Kept as plain matplotlib functions (no Streamlit dependency) so they can be
unit-tested/used standalone; app.py just calls these and hands the returned
Figure to st.pyplot().

Styled as a dark, abandoned-factory escape room: near-black stone floors
and machinery, an orange lamp glow at the start, teal leaking pipes, a
bottomless-black abyss, a warm wooden bridge plank, a glowing gold access
card, and an emergency-green exit door.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

BG_COLOR = "#0d0f12"        # near-black factory gloom
PANEL_COLOR = "#14171b"     # slightly lighter panel background
GRID_LINE_COLOR = "#2a2e34"
TEXT_COLOR = "#d8dee6"

CELL_COLORS = {
    ".": "#1b1e23",  # bare factory floor
    "#": "#24272c",  # wall / machinery (darker than floor, with hatch texture)
    "~": "#1e5f66",  # leaking pipes -- dim teal slick
    "P": "#050505",  # the abyss -- bottomless black
    "B": "#9a6a35",  # bridge plank, warm wood under lamp light
    "K": "#f2b705",  # access card -- glowing gold
    "S": "#ff7a1a",  # start -- orange lamp glow
    "G": "#28a862",  # door / exit -- emergency-green glow
}
CELL_TEXT_COLORS = {
    "P": "#8a1f1f",   # dim ember-red label, barely visible in the void
    "B": "#241505",
    "K": "#241505",
    "S": "#241505",
    "G": "#0a2916",
}
CELL_LABELS = {"P": "P", "B": "B", "K": "K", "S": "S", "G": "G"}
WALL_HATCH_COLOR = "#3a3f46"


def _style_dark_axes(ax) -> None:
    ax.set_facecolor(PANEL_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_LINE_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)


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

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), facecolor=BG_COLOR)

    ax = axes[0, 0]
    ax.plot(episodes, returns, color="#4a5568", linewidth=0.8, label="return")
    ax.plot(episodes, _rolling_mean(returns, window), color="#ff7a1a", linewidth=2, label=f"rolling mean ({window})")
    ax.set_title("Episode return")
    ax.set_xlabel("episode")
    ax.legend(loc="lower right", fontsize=8, facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR, edgecolor=GRID_LINE_COLOR)

    ax = axes[0, 1]
    ax.plot(episodes, steps, color="#4a5568", linewidth=0.8, label="steps")
    ax.plot(episodes, _rolling_mean(steps, window), color="#f2b705", linewidth=2, label=f"rolling mean ({window})")
    ax.set_title("Steps to terminal state")
    ax.set_xlabel("episode")
    ax.legend(loc="upper right", fontsize=8, facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR, edgecolor=GRID_LINE_COLOR)

    ax = axes[1, 0]
    ax.plot(episodes, epsilons, color="#e8590c")
    ax.set_title("Epsilon (exploration rate)")
    ax.set_xlabel("episode")

    ax = axes[1, 1]
    ax.plot(episodes, _rolling_mean(successes, window), color="#28a862")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Rolling success rate (window={window})")
    ax.set_xlabel("episode")

    for ax in axes.flat:
        _style_dark_axes(ax)
        ax.grid(True, color=GRID_LINE_COLOR, linewidth=0.5)

    fig.tight_layout()
    return fig


def trajectory_to_positions(trajectory: list[dict], n_cols: int = 10) -> list[tuple[int, int]]:
    """Convert a recorded trajectory's flat state indices to (row, col) coords.

    State indices encode (row, col, has_key, bridge_collapsed) as
    ((row*n_cols+col)*2+has_key)*2+bridge_collapsed (see
    grid_env.GridWorld.encode_state) -- both flag bits are dropped here
    since rendering only needs position.
    """
    positions = []
    for step in trajectory:
        pos_index = step["state"] // 4
        positions.append(divmod(pos_index, n_cols))
    return positions


def render_grid(
    layout: tuple[str, ...],
    path: list[tuple[int, int]] | None = None,
    agent_pos: tuple[int, int] | None = None,
    title: str = "",
) -> Figure:
    """Static render of the grid, with an optional traversed path and agent marker."""
    n_rows = len(layout)
    n_cols = len(layout[0])

    fig, ax = plt.subplots(figsize=(5, 5), facecolor=BG_COLOR)
    for r in range(n_rows):
        for c in range(n_cols):
            symbol = layout[r][c]
            color = CELL_COLORS.get(symbol, CELL_COLORS["."])
            is_wall = symbol == "#"
            ax.add_patch(
                plt.Rectangle(
                    (c, n_rows - 1 - r),
                    1,
                    1,
                    facecolor=color,
                    edgecolor=WALL_HATCH_COLOR if is_wall else GRID_LINE_COLOR,
                    hatch="///" if is_wall else None,
                    linewidth=0.6,
                )
            )
            label = CELL_LABELS.get(symbol)
            if label:
                ax.text(
                    c + 0.5,
                    n_rows - 1 - r + 0.5,
                    label,
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color=CELL_TEXT_COLORS.get(symbol, TEXT_COLOR),
                )

    if path:
        xs = [c + 0.5 for _, c in path]
        ys = [n_rows - 1 - r + 0.5 for r, _ in path]
        ax.plot(xs, ys, color="#5fd0ff", linewidth=2, alpha=0.9, zorder=3)

    if agent_pos:
        r, c = agent_pos
        ax.plot(c + 0.5, n_rows - 1 - r + 0.5, marker="o", markersize=14, color="#ff2d2d", zorder=4)

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title(title, color=TEXT_COLOR)
    ax.set_facecolor(PANEL_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_LINE_COLOR)
    fig.tight_layout()
    return fig


def render_episode_step(layout: tuple[str, ...], trajectory: list[dict], step_index: int, n_cols: int = 10) -> Figure:
    """Render the grid with the path walked up to (and agent at) step_index."""
    positions = trajectory_to_positions(trajectory, n_cols=n_cols)
    step_index = max(0, min(step_index, len(positions) - 1))
    return render_grid(
        layout,
        path=positions[: step_index + 1],
        agent_pos=positions[step_index],
        title=f"Step {step_index} / {len(positions) - 1}",
    )
