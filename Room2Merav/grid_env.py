"""10x10 grid-world engine for Room 2 (SARSA).

Layout symbols:
    S   start cell (exactly one)
    G   goal cell (exactly one) -- the exit to the next room, but it only
        opens once the agent is holding the key (see below)
    K   key (exactly one) -- must be collected before G will end the episode
    #   wall (impassable)
    ~   slippery floor: the action taken may slip to a perpendicular direction
    P   pit (optional): terminates the episode with a penalty -- meant to be
        laid out as a barrier crossed by a single safe "bridge" cell (a
        normal '.' cell within the pit's row/column)
    .   free cell

room2_env.py configures the layout and reward shaping; this module only
implements the movement dynamics.

Key mechanic
    Because reaching G behaves differently depending on whether the key has
    been collected, "have I got the key" is part of the true state, not
    just an environment detail: the observed state index encodes
    (row, col, has_key), doubling the state space to n_rows*n_cols*2. This
    keeps the task Markovian -- without it, the same (row, col) would need
    two different values depending on unobserved history, breaking SARSA's
    update rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT")
_DELTA = {
    0: (-1, 0),  # UP
    1: (0, 1),   # RIGHT
    2: (1, 0),   # DOWN
    3: (0, -1),  # LEFT
}


@dataclass
class GridWorldConfig:
    layout: tuple[str, ...]
    slip_prob: float = 0.2
    step_reward: float = -1.0
    pit_reward: float = -20.0
    key_bonus: float = 10.0
    goal_base_reward: float = 100.0
    goal_decay_per_step: float = 1.0
    goal_min_reward: float = 20.0
    max_steps: int = 200
    seed: Optional[int] = None


class GridWorld:
    """Slippery grid-world with a single exit state gated behind a key.

    step() only exposes (next_state, reward, terminated, truncated, info) --
    the transition model is never returned, matching the "model unknown"
    premise of the TD rooms (SARSA/Q-Learning). A model-based room (DP) is
    free to derive its own P(s'|s,a) table separately since it is allowed
    to know the dynamics; that lookup does not belong on this class.
    """

    def __init__(self, config: GridWorldConfig):
        self.cfg = config
        self.grid = [list(row) for row in config.layout]
        self.n_rows = len(self.grid)
        self.n_cols = len(self.grid[0]) if self.grid else 0
        if self.n_rows != 10 or any(len(row) != 10 for row in self.grid):
            raise ValueError("Rooms 1-3 require a 10x10 layout")

        self.start = self._find_unique("S")
        self.goal = self._find_unique("G")
        self.key_pos = self._find_unique("K")
        self.rng = np.random.default_rng(config.seed)
        self.state = self.start
        self.has_key = False
        self.steps_taken = 0

    def _find_unique(self, symbol: str) -> tuple[int, int]:
        matches = [
            (r, c)
            for r, row in enumerate(self.grid)
            for c, val in enumerate(row)
            if val == symbol
        ]
        if len(matches) != 1:
            raise ValueError(f"layout must contain exactly one '{symbol}' cell, found {len(matches)}")
        return matches[0]

    @property
    def n_states(self) -> int:
        return self.n_rows * self.n_cols * 2  # x2 for the has_key flag

    @property
    def n_actions(self) -> int:
        return len(ACTIONS)

    def encode_state(self, pos: tuple[int, int], has_key: bool) -> int:
        r, c = pos
        return (r * self.n_cols + c) * 2 + int(has_key)

    def decode_state(self, index: int) -> tuple[tuple[int, int], bool]:
        pos_index, has_key = divmod(index, 2)
        return divmod(pos_index, self.n_cols), bool(has_key)

    def to_coords(self, index: int) -> tuple[int, int]:
        """(row, col) for a full state index, discarding the has_key bit -- for rendering."""
        pos, _ = self.decode_state(index)
        return pos

    def reset(self, seed: Optional[int] = None) -> tuple[int, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.start
        self.has_key = False
        self.steps_taken = 0
        return self.encode_state(self.state, self.has_key), {}

    def _is_wall(self, pos: tuple[int, int]) -> bool:
        r, c = pos
        if not (0 <= r < self.n_rows and 0 <= c < self.n_cols):
            return True
        return self.grid[r][c] == "#"

    def _attempt_move(self, pos: tuple[int, int], action: int) -> tuple[int, int]:
        dr, dc = _DELTA[action]
        new_pos = (pos[0] + dr, pos[1] + dc)
        return pos if self._is_wall(new_pos) else new_pos

    @staticmethod
    def _perpendicular_actions(action: int) -> tuple[int, int]:
        return (action - 1) % 4, (action + 1) % 4

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        if not (0 <= action < 4):
            raise ValueError(f"action must be in [0, 3], got {action}")

        current_cell = self.grid[self.state[0]][self.state[1]]
        slipped = False
        actual_action = action
        if current_cell == "~" and self.rng.random() < self.cfg.slip_prob:
            actual_action = self.rng.choice(self._perpendicular_actions(action))
            slipped = True

        self.state = self._attempt_move(self.state, actual_action)
        self.steps_taken += 1
        landed_cell = self.grid[self.state[0]][self.state[1]]

        terminated = False
        picked_key = False
        reward = self.cfg.step_reward
        if landed_cell == "P":
            reward = self.cfg.pit_reward
            terminated = True
        elif landed_cell == "K" and not self.has_key:
            self.has_key = True
            picked_key = True
            reward = self.cfg.step_reward + self.cfg.key_bonus
        elif landed_cell == "G" and self.has_key:
            bonus = self.cfg.goal_base_reward - self.cfg.goal_decay_per_step * self.steps_taken
            reward = max(bonus, self.cfg.goal_min_reward)
            terminated = True

        truncated = (not terminated) and self.steps_taken >= self.cfg.max_steps
        info = {
            "steps": self.steps_taken,
            "slipped": slipped,
            "intended_action": action,
            "has_key": self.has_key,
            "picked_key": picked_key,
        }
        return self.encode_state(self.state, self.has_key), reward, terminated, truncated, info

    def render(self) -> str:
        rows = []
        for r in range(self.n_rows):
            row_chars = [
                "A" if (r, c) == self.state else self.grid[r][c]
                for c in range(self.n_cols)
            ]
            rows.append("".join(row_chars))
        return "\n".join(rows) + f"\n(has_key={self.has_key})"
