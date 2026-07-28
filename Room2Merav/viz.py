"""Plotting helpers for Room 2: learning-curve charts and episode replay.

Kept as plain matplotlib functions (no Streamlit dependency) so they can be
unit-tested/used standalone; app.py just calls these and hands the returned
Figure to st.pyplot().
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

CELL_COLORS = {
    ".": "#f5f5f5",  # empty
    "#": "#333333",  # wall
    "~": "#a8d8ff",  # slippery
    "T": "#e05555",  # trap
    "S": "#ffe082",  # start
    "G": "#66bb6a",  # goal
}
CELL_LABELS = {"T": "T", "S": "S", "G": "G"}


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
    ax.plot(episodes, returns, color="#c8d6e5", linewidth=0.8, label="return")
    ax.plot(episodes, _rolling_mean(returns, window), color="#1f6feb", linewidth=2, label=f"rolling mean ({window})")
    ax.set_title("Episode return")
    ax.set_xlabel("episode")
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[0, 1]
    ax.plot(episodes, steps, color="#c8d6e5", linewidth=0.8, label="steps")
    ax.plot(episodes, _rolling_mean(steps, window), color="#1f6feb", linewidth=2, label=f"rolling mean ({window})")
    ax.set_title("Steps to terminal state")
    ax.set_xlabel("episode")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1, 0]
    ax.plot(episodes, epsilons, color="#e8590c")
    ax.set_title("Epsilon (exploration rate)")
    ax.set_xlabel("episode")

    ax = axes[1, 1]
    ax.plot(episodes, _rolling_mean(successes, window), color="#2f9e44")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Rolling success rate (window={window})")
    ax.set_xlabel("episode")

    fig.tight_layout()
    return fig


def trajectory_to_positions(trajectory: list[dict], n_cols: int = 10) -> list[tuple[int, int]]:
    """Convert a recorded trajectory's flat state indices to (row, col) coords."""
    return [divmod(step["state"], n_cols) for step in trajectory]


def render_grid(
    layout: tuple[str, ...],
    path: list[tuple[int, int]] | None = None,
    agent_pos: tuple[int, int] | None = None,
    title: str = "",
) -> Figure:
    """Static render of the grid, with an optional traversed path and agent marker."""
    n_rows = len(layout)
    n_cols = len(layout[0])

    fig, ax = plt.subplots(figsize=(5, 5))
    for r in range(n_rows):
        for c in range(n_cols):
            symbol = layout[r][c]
            color = CELL_COLORS.get(symbol, "#f5f5f5")
            ax.add_patch(plt.Rectangle((c, n_rows - 1 - r), 1, 1, facecolor=color, edgecolor="#cccccc"))
            label = CELL_LABELS.get(symbol)
            if label:
                ax.text(c + 0.5, n_rows - 1 - r + 0.5, label, ha="center", va="center", fontsize=9, fontweight="bold")

    if path:
        xs = [c + 0.5 for _, c in path]
        ys = [n_rows - 1 - r + 0.5 for r, _ in path]
        ax.plot(xs, ys, color="#1f6feb", linewidth=2, alpha=0.8, zorder=3)

    if agent_pos:
        r, c = agent_pos
        ax.plot(c + 0.5, n_rows - 1 - r + 0.5, marker="o", markersize=14, color="#d62828", zorder=4)

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title(title)
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
