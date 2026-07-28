"""10x10 grid-world engine for Room 2 (SARSA).

Layout symbols:
    S   start cell (exactly one)
    G   goal cell (exactly one) -- the single terminal state / exit to the next room
    #   wall (impassable)
    ~   slippery floor: the action taken may slip to a perpendicular direction
    T   trap (optional): terminates the episode with a penalty
    .   free cell

room2_env.py configures the layout and reward shaping; this module only
implements the movement dynamics.
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
    trap_reward: float = -20.0
    goal_base_reward: float = 100.0
    goal_decay_per_step: float = 1.0
    goal_min_reward: float = 20.0
    max_steps: int = 200
    seed: Optional[int] = None


class GridWorld:
    """Generic slippery grid-world with a single exit state.

    step() only exposes (next_state, reward, terminated, truncated, info) --
    the transition model is never returned, matching the "model unknown"
    premise of the TD rooms (SARSA/Q-Learning). A model-based room (DP) is
    free to derive its own P(s'|s,a) table separately since it is allowed
    to know the dynamics; that lookup does not belong on this shared class.
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
        self.rng = np.random.default_rng(config.seed)
        self.state = self.start
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
        return self.n_rows * self.n_cols

    @property
    def n_actions(self) -> int:
        return len(ACTIONS)

    def to_index(self, pos: tuple[int, int]) -> int:
        r, c = pos
        return r * self.n_cols + c

    def to_coords(self, index: int) -> tuple[int, int]:
        return divmod(index, self.n_cols)

    def reset(self, seed: Optional[int] = None) -> tuple[int, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.start
        self.steps_taken = 0
        return self.to_index(self.state), {}

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
        reward = self.cfg.step_reward
        if landed_cell == "T":
            reward = self.cfg.trap_reward
            terminated = True
        elif landed_cell == "G":
            bonus = self.cfg.goal_base_reward - self.cfg.goal_decay_per_step * self.steps_taken
            reward = max(bonus, self.cfg.goal_min_reward)
            terminated = True

        truncated = (not terminated) and self.steps_taken >= self.cfg.max_steps
        info = {"steps": self.steps_taken, "slipped": slipped, "intended_action": action}
        return self.to_index(self.state), reward, terminated, truncated, info

    def render(self) -> str:
        rows = []
        for r in range(self.n_rows):
            row_chars = [
                "A" if (r, c) == self.state else self.grid[r][c]
                for c in range(self.n_cols)
            ]
            rows.append("".join(row_chars))
        return "\n".join(rows)
