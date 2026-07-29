"""Room 2 -- SARSA, unknown model: "The Collapsing Bridge".

Theme: an abandoned factory. A chasm (the abyss) splits it in two; the only
way across is a single walkway (the bridge). A key, tucked away from the
direct route, unlocks the exit door. Slippery cells are scattered through
the factory -- including right at the edge of the abyss, which is the
whole point of this room: an on-policy agent has to live with the
consequences of its own exploration, so its value estimates should reflect
the real, ongoing risk of a slip near the abyss rather than assuming
flawless play from here on.

Design (state / action / reward), per the assignment brief, course slides,
and the team's spec for this room (bridge over a chasm, key that unlocks
the door):

State space
    Discrete, 200 states: (row, col, has_key) flattened to a single index
    -- see grid_env.GridWorld.encode_state. has_key has to be part of the
    *true* state (not just environment bookkeeping): the same cell behaves
    differently once the key is held (G becomes a terminal exit).
    Leaving it out of the state would break the Markov property SARSA's
    update rule relies on. The agent only ever observes this index, never
    the transition probabilities.

Action space
    Discrete(4): 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT. A move into a wall or the
    grid border keeps the agent in place (wasted step, still costs reward).

Reward function
    -1 per step (time cost).
    -20 and episode ends, on falling into the abyss ('P') -- including a
        failed bridge crossing (see below).
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

Slippery cells ('~') and the bridge ('B')
    "Slippery" here doesn't mean water -- it means a cell where, with
    probability `slip_prob`, the next state isn't the one the action
    intended: the actual move slips to one of the two directions
    perpendicular to it instead. The bridge is not a separate stateful
    mechanic -- it's simply another slippery cell, so "collapsing" isn't
    a one-time flag that gets set and remembered; every crossing attempt
    carries the same slip_prob risk, whether it's the agent's first
    attempt or its fifth. Scattered slippery cells sit right along the
    abyss edge (row 6) and through the upper maze, so a slip there can
    drop the agent straight in. This is deliberate -- it is what should
    make SARSA's on-policy caution visible in the learned path.

Layout
    Upper maze (rows 0-5) with the key in the top-right corner -- off the
    direct route, forcing a real detour -- scattered single-cell walls
    (rather than clustered blocks) standing in for factory machinery,
    scattered slippery cells along the abyss edge (row 6) rather than one
    solid band, the abyss/bridge row (7), and a small goal room (rows 8-9)
    with the locked door below it. "Repair area" (the assignment's added
    flavor note) is represented as the maze's walled sections generally --
    no distinct game mechanic was specified for it, so it's treated as
    scenery rather than a new hazard type.
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
    "~.~.~.~.~.",
    "PPBPPPPPPP",
    "..........",
    ".........G",
)


def _bfs_reachable(grid: list[list[str]], start: tuple[int, int]) -> set[tuple[int, int]]:
    n_rows, n_cols = len(grid), len(grid[0])
    seen = {start}
    frontier = [start]
    while frontier:
        r, c = frontier.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < n_rows
                and 0 <= nc < n_cols
                and grid[nr][nc] not in ("#", "P")
                and (nr, nc) not in seen
            ):
                seen.add((nr, nc))
                frontier.append((nr, nc))
    return seen


def generate_random_layout(
    seed: Optional[int] = None,
    n_slippery: int = 8,
    n_pits: int = 6,
    n_walls: int = 8,
    base_layout: tuple[str, ...] | None = None,
    max_attempts: int = 500,
) -> tuple[str, ...]:
    """Return a new layout with `~`, `P` and `#` scattered across floor tiles.

    Special cells `S`, `K`, `B` and `G` are preserved in place; only
    originally-empty floor cells are used for placing hazards and walls.
    Counts are capped to the number of available floor cells.

    A random scatter has no guarantee of leaving the maze solvable --
    corner cells like `G` have only two neighbors, and a single wall or
    pit landing on either one seals them off completely. So each attempt
    is checked with a BFS (treating '#' and 'P' as blocked, matching
    grid_env's actual failure semantics) and re-shuffled if `S` can't
    reach both `K` and `G`; raises after `max_attempts` failed tries
    rather than silently handing back an unsolvable maze.
    """
    if base_layout is None:
        base_layout = BASE_ROOM2_LAYOUT

    rng = np.random.default_rng(seed)
    specials = {
        (r, c): ch
        for r, row in enumerate(base_layout)
        for c, ch in enumerate(row)
        if ch in ("S", "K", "B", "G")
    }
    free = [
        (r, c)
        for r, row in enumerate(base_layout)
        for c, ch in enumerate(row)
        if ch == "."
    ]

    max_free = len(free)
    n_walls = min(n_walls, max_free)
    n_slippery = min(n_slippery, max_free - n_walls)
    n_pits = min(n_pits, max_free - n_walls - n_slippery)
    pos_of = {ch: (r, c) for (r, c), ch in specials.items()}

    for _ in range(max_attempts):
        shuffled = list(free)
        rng.shuffle(shuffled)
        grid = [list(row) for row in base_layout]

        idx = 0
        for _ in range(n_walls):
            r, c = shuffled[idx]
            grid[r][c] = "#"
            idx += 1
        for _ in range(n_slippery):
            r, c = shuffled[idx]
            grid[r][c] = "~"
            idx += 1
        for _ in range(n_pits):
            r, c = shuffled[idx]
            grid[r][c] = "P"
            idx += 1

        for (r, c), ch in specials.items():
            grid[r][c] = ch

        reachable = _bfs_reachable(grid, pos_of["S"])
        if pos_of["K"] in reachable and pos_of["G"] in reachable:
            return tuple("".join(row) for row in grid)

    raise ValueError(
        f"generate_random_layout: no solvable maze found in {max_attempts} attempts "
        f"with n_walls={n_walls}, n_slippery={n_slippery}, n_pits={n_pits} -- try lower counts"
    )


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
