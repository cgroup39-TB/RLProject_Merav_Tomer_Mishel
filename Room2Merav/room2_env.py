"""Room 2 -- SARSA, unknown model: "The Collapsing Bridge".

Theme: an abandoned factory. A chasm (the abyss) splits it in two; the only
way across is a single walkway (the bridge) that gives out the moment you
step off it. A key, tucked away from the direct route, unlocks the exit
door. Leaking pipes make some floor tiles slippery -- including right at
the edge of the abyss, which is the whole point of this room: an
on-policy agent that has to live with the consequences of its own
exploration should learn to keep its distance from a hazard a random slip
could drop it into, more so than an agent that always assumes it'll play
optimally from here on (Q-Learning, Room 3). See sarsa_vs_qlearning_demo.py
for a side-by-side check of that claim.

Design (state / action / reward), per the assignment brief, course slides,
and the team's spec for this room (bridge over a chasm, key that unlocks
the door):

State space
    Discrete, 400 states: (row, col, has_key, bridge_collapsed) flattened
    to a single index -- see grid_env.GridWorld.encode_state. Both flags
    have to be part of the *true* state (not just environment bookkeeping):
    the same cell behaves differently once the key is held (G becomes a
    terminal exit) and once the bridge has been used (stepping back onto
    it becomes fatal instead of safe). Leaving either out of the state
    would break the Markov property SARSA's update rule relies on. The
    agent only ever observes this index, never the transition
    probabilities.

Action space
    Discrete(4): 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT. A move into a wall or the
    grid border keeps the agent in place (wasted step, still costs reward).

Reward function
    -1 per step (time cost).
    -20 and episode ends, on falling into the abyss ('P') -- or on
        stepping back onto the bridge ('B') after it has already
        collapsed, which is exactly as fatal.
    +10 (one-time, on top of the step cost), the first time the agent
        reaches the key ('K') -- a subgoal bonus so a flat tabular agent
        has a learnable signal to bother detouring for it.
    On reaching the door ('G') *while holding the key*, the terminal exit:
    reward = max(100 - steps_taken, 20) -- a large bonus that shrinks the
    longer the episode takes, implementing "the faster the agent escapes,
    the higher the reward."
    Reaching 'G' *without* the key is just a normal floor cell (locked
    door, no bonus, no termination).
    Episodes are truncated (no bonus) after `max_steps`.

Slippery cells ('~') and the abyss edge
    With probability `slip_prob` the actual move is one of the two
    directions perpendicular to the intended one. Row 6, right along the
    abyss (row 7), is entirely slippery leaking-pipe floor: crossing to
    the bridge means walking the edge, where a slip can drop the agent
    straight into the abyss. This is deliberate -- it is what should make
    SARSA's on-policy caution visible in the learned path.

Bridge ('B')
    Row 7 is the abyss save for a single bridge cell (col 2). Stepping off
    it collapses it for the rest of the episode -- a second crossing
    attempt falls straight through. The key sits on the near (start) side
    of the bridge, so the intended solution never needs a second crossing;
    the collapse mainly punishes an agent that hasn't planned its route.

Layout
    Upper maze (rows 0-5) with the key in the top-right corner -- off the
    direct route, forcing a real detour -- scattered single-cell walls
    (rather than clustered blocks) standing in for factory machinery, a
    slippery band right at the abyss edge (row 6), the abyss/bridge row
    (7), and a small goal room (rows 8-9) with the locked door below it.
    "Repair area" (the assignment's added flavor note) is represented as
    the maze's walled sections generally -- no distinct game mechanic was
    specified for it, so it's treated as scenery rather than a new hazard
    type.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from grid_env import GridWorld, GridWorldConfig

BASE_ROOM2_LAYOUT = (
    "S..#..#..K",
    ".#.....#..",
    "...#..#..#",
    "#....#..#.",
    "..#.~..#..",
    ".#..#.#..#",
    "~~~~~~~~~~",
    "PPBPPPPPPP",
    "..........",
    ".........G",
)


def generate_random_layout(
    seed: Optional[int] = None,
    n_slippery: int = 12,
    n_pits: int = 10,
    n_walls: int = 12,
    base_layout: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Return a new layout with `~`, `P` and `#` scattered across floor tiles.

    Special cells `S`, `K`, `B` and `G` are preserved in place; only
    originally-empty floor cells are used for placing hazards and walls.
    Counts are capped to the number of available floor cells.
    """
    if base_layout is None:
        base_layout = BASE_ROOM2_LAYOUT

    rng = np.random.default_rng(seed)
    grid = [list(row) for row in base_layout]

    # collect special positions to protect them from being overwritten
    specials = {(r, c): grid[r][c] for r in range(10) for c in range(10) if grid[r][c] in ("S", "K", "B", "G")}

    # free floor positions (only '.')
    free = [(r, c) for r in range(10) for c in range(10) if grid[r][c] == "."]
    rng.shuffle(free)

    # cap counts
    max_free = len(free)
    n_walls = min(n_walls, max_free)
    remaining = max_free - n_walls
    n_slippery = min(n_slippery, remaining)
    remaining -= n_slippery
    n_pits = min(n_pits, remaining)

    idx = 0
    for _ in range(n_walls):
        r, c = free[idx]
        grid[r][c] = "#"
        idx += 1
    for _ in range(n_slippery):
        r, c = free[idx]
        grid[r][c] = "~"
        idx += 1
    for _ in range(n_pits):
        r, c = free[idx]
        grid[r][c] = "P"
        idx += 1

    # restore specials (shouldn't be necessary, but safe)
    for (r, c), ch in specials.items():
        grid[r][c] = ch

    return tuple("".join(row) for row in grid)


# keep the legacy name for code that imports ROOM2_LAYOUT directly
ROOM2_LAYOUT = BASE_ROOM2_LAYOUT


def make_room2_env(
    slip_prob: float = 0.2,
    step_reward: float = -1.0,
    pit_reward: float = -20.0,
    key_bonus: float = 10.0,
    goal_base_reward: float = 100.0,
    goal_decay_per_step: float = 1.0,
    goal_min_reward: float = 20.0,
    max_steps: int = 200,
    seed: Optional[int] = None,
    layout: tuple[str, ...] | None = None,
) -> GridWorld:
    """Build Room 2's environment. All reward/slip knobs are exposed here
    so training code can sweep hyperparameters without touching the layout.
    """
    config = GridWorldConfig(
        layout=ROOM2_LAYOUT if layout is None else layout,
        slip_prob=slip_prob,
        step_reward=step_reward,
        pit_reward=pit_reward,
        key_bonus=key_bonus,
        goal_base_reward=goal_base_reward,
        goal_decay_per_step=goal_decay_per_step,
        goal_min_reward=goal_min_reward,
        max_steps=max_steps,
        seed=seed,
    )
    return GridWorld(config)
