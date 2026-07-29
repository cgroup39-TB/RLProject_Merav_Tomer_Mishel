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

from grid_env import GridWorld, GridWorldConfig

ROOM2_LAYOUT = (
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
) -> GridWorld:
    """Build Room 2's environment. All reward/slip knobs are exposed here
    so training code can sweep hyperparameters without touching the layout.
    """
    config = GridWorldConfig(
        layout=ROOM2_LAYOUT,
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
