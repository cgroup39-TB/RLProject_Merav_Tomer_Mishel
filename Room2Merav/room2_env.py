"""Room 2 -- SARSA, unknown model, slippery 10x10 grid.

Design (state / action / reward), per the assignment brief and course slides:

State space
    Discrete, 100 states: one per (row, col) cell on the 10x10 grid,
    flattened to an index 0-99 (`row * 10 + col`). The agent only ever
    observes this index -- never the transition probabilities -- which is
    what "model unknown" means here: SARSA must learn Q(s, a) purely from
    (state, action, reward, next_state, next_action) tuples collected by
    interacting with `step()`.

Action space
    Discrete(4): 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT. A move into a wall or the
    grid border keeps the agent in place (wasted step, still costs reward).

Reward function
    -1 per step (time cost).
    -20 and episode ends, on stepping into a trap cell ('T').
    On reaching the goal ('G'), the single terminal/exit state:
        reward = max(100 - 1 * steps_taken, 20)
    i.e. a large bonus that shrinks the longer the episode takes, on top of
    the per-step penalty -- this directly implements the brief's "the
    faster the agent escapes, the higher the reward" requirement, rather
    than leaving it implicit in the discount factor alone.
    Episodes are truncated (no bonus) after `max_steps` with no reward.

Slippery cells ('~')
    With probability `slip_prob` the actual move is one of the two
    directions perpendicular to the intended one (FrozenLake-style),
    instead of the intended direction. This is what makes the model
    "unknown": the agent cannot compute exact transition probabilities in
    advance and must learn from experience.

Layout
    A single start ('S') and single goal ('G'), a wall maze harder than
    Room 1's plain grid, a 3-cell vertical slip corridor, and one trap
    placed off the shortest path so it punishes careless exploration
    without being unavoidable.
"""
from __future__ import annotations

from typing import Optional

from grid_env import GridWorld, GridWorldConfig

ROOM2_LAYOUT = (
    "S...#.....",
    ".##.#.###.",
    "....#.....",
    ".###.#.##.",
    "....~.....",
    ".#.#~#.#..",
    ".#.#~#.#..",
    ".#...#.#..",
    ".#.###.#T.",
    ".......#.G",
)


def make_room2_env(
    slip_prob: float = 0.2,
    step_reward: float = -1.0,
    trap_reward: float = -20.0,
    goal_base_reward: float = 100.0,
    goal_decay_per_step: float = 1.0,
    goal_min_reward: float = 20.0,
    max_steps: int = 200,
    seed: Optional[int] = None,
) -> GridWorld:
    """Build Room 2's environment. All reward/slip knobs are exposed here
    so training code can sweep hyperparameters without touching the layout.
    """
    config = GridWorldConfig(
        layout=ROOM2_LAYOUT,
        slip_prob=slip_prob,
        step_reward=step_reward,
        trap_reward=trap_reward,
        goal_base_reward=goal_base_reward,
        goal_decay_per_step=goal_decay_per_step,
        goal_min_reward=goal_min_reward,
        max_steps=max_steps,
        seed=seed,
    )
    return GridWorld(config)
