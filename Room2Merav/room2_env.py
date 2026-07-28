"""Room 2 -- SARSA, unknown model, slippery 10x10 grid with a key-locked exit.

Design (state / action / reward), per the assignment brief, course slides,
and the team's added spec for this room: a bridge crossing a pit, and a key
that must be collected before the exit opens.

State space
    Discrete, 200 states: (row, col, has_key) flattened to a single index
    -- see grid_env.GridWorld.encode_state. The has_key bit has to be part
    of the state (not just environment bookkeeping) because the same cell
    behaves differently once the key is held -- G becomes a terminal exit
    only then. The agent only ever observes this index, never the
    transition probabilities: SARSA must learn Q(s, a) purely from
    (state, action, reward, next_state, next_action) tuples collected by
    interacting with `step()`.

Action space
    Discrete(4): 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT. A move into a wall or the
    grid border keeps the agent in place (wasted step, still costs reward).

Reward function
    -1 per step (time cost).
    -20 and episode ends, on stepping into a pit cell ('P').
    +10 (one-time, on top of the step cost) the first time the agent steps
        onto the key ('K') -- a subgoal bonus so a flat tabular agent has a
        learnable signal to go fetch the key, since it is otherwise
        temporally far from the goal payoff.
    On reaching the goal ('G') *while holding the key*, the terminal exit:
        reward = max(100 - 1 * steps_taken, 20)
    i.e. a large bonus that shrinks the longer the episode takes, on top of
    the per-step penalty -- implements the brief's "the faster the agent
    escapes, the higher the reward" requirement.
    Reaching 'G' *without* the key is just a normal floor cell (no bonus,
    no termination) -- the exit is locked until the key is collected.
    Episodes are truncated (no bonus) after `max_steps` with no reward.

Slippery cells ('~')
    With probability `slip_prob` the actual move is one of the two
    directions perpendicular to the intended one (FrozenLake-style),
    instead of the intended direction.

Pit and bridge ('P' / '.')
    Row 7 is a pit band spanning the full width of the grid except for a
    single "bridge" cell (col 2, left as plain floor) -- the only safe
    crossing from the upper maze (rows 0-6, where S and K live) down to the
    goal room (rows 8-9, where G lives). Stepping on any other cell in that
    row ends the episode, same as a trap.

Layout
    A wall maze in the upper section (rows 0-6) with a 3-cell vertical slip
    corridor, the key ('K') placed in the top-right corner -- off the
    direct route to the bridge, forcing a real detour -- the pit/bridge
    row (7), and a small goal room (rows 8-9) below it.
"""
from __future__ import annotations

from typing import Optional

from grid_env import GridWorld, GridWorldConfig

ROOM2_LAYOUT = (
    "S...#....K",
    ".##.#.###.",
    "....#.....",
    ".###.#.##.",
    "....~.....",
    ".#.#~#.#..",
    ".#.#~#.#..",
    "PP.PPPPPPP",
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
