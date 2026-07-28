"""
Generalized, editable Grid World environment shared by the DP / SARSA /
Q-Learning rooms. The grid layout (walls, slippery cells, traps, start, goal)
and reward parameters are fully configurable at runtime from the UI.
"""

from dataclasses import dataclass, field


class Cell:
    EMPTY = "empty"
    WALL = "wall"
    SLIPPERY = "slippery"
    GOAL = "goal"
    TRAP = "trap"
    START = "start"


class Action:
    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


ACTIONS = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
ACTION_DELTAS = {Action.UP: (-1, 0), Action.DOWN: (1, 0), Action.LEFT: (0, -1), Action.RIGHT: (0, 1)}
ACTION_ARROWS = {Action.UP: "↑", Action.DOWN: "↓", Action.LEFT: "←", Action.RIGHT: "→"}

_CLOCKWISE = [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]


def rotate_clockwise(a):
    return _CLOCKWISE[(_CLOCKWISE.index(a) + 1) % 4]


@dataclass
class GridConfig:
    rows: int = 10
    cols: int = 10
    cells: dict = field(default_factory=dict)          # (r,c) -> Cell.* type, default EMPTY
    cell_rewards: dict = field(default_factory=dict)    # (r,c) -> reward override (paint-time)
    start: tuple = (0, 0)
    goal: tuple = (9, 9)

    def cell_type(self, r, c):
        return self.cells.get((r, c), Cell.EMPTY)


class GridWorldEnv:
    """
    A configurable Grid World whose full transition model is known,
    so it can be solved with Dynamic Programming (Value / Policy Iteration).
    The same class is reused as the base for the model-free rooms
    (SARSA / Q-Learning just won't call get_transition_model()).
    """

    def __init__(
        self,
        config: GridConfig,
        slip_prob=0.3,
        step_reward=0.0,
        goal_reward=100.0,
        trap_reward=-100.0,
        gamma=0.95,
        potential_shaping=False,
        max_steps=500,
    ):
        self.cfg = config
        self.slip_prob = slip_prob
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.trap_reward = trap_reward
        self.gamma = gamma
        self.potential_shaping = potential_shaping
        self.max_steps = max_steps

        self.rows, self.cols = config.rows, config.cols
        self.n_states = self.rows * self.cols
        self.n_actions = len(ACTIONS)

        self._state = None
        self._steps = 0

    # ---------- indexing ----------
    def s2i(self, s):
        r, c = s
        return r * self.cols + c

    def i2s(self, i):
        return divmod(i, self.cols)

    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_terminal(self, s):
        t = self.cfg.cell_type(*s)
        return t in (Cell.GOAL, Cell.TRAP)

    # ---------- reward shaping ----------
    def _potential(self, s):
        r, c = s
        gr, gc = self.cfg.goal
        return -(abs(r - gr) + abs(c - gc))

    def _base_reward(self, s):
        if s in self.cfg.cell_rewards:
            return self.cfg.cell_rewards[s]
        t = self.cfg.cell_type(*s)
        if t == Cell.GOAL:
            return self.goal_reward
        if t == Cell.TRAP:
            return self.trap_reward
        return self.step_reward

    def _transition_reward(self, s, s_next):
        reward = self._base_reward(s_next) if self.is_terminal(s_next) else self._base_reward(s)
        if self.potential_shaping and not self.is_terminal(s):
            reward += self.gamma * self._potential(s_next) - self._potential(s)
        return reward

    def _move(self, s, a):
        """Deterministic single move: walls and grid edges block movement (agent stays)."""
        r, c = s
        dr, dc = ACTION_DELTAS[a]
        nr, nc = r + dr, c + dc
        if not self.in_bounds(nr, nc) or self.cfg.cell_type(nr, nc) == Cell.WALL:
            return s
        return (nr, nc)

    # ---------- known model, for DP ----------
    def get_transition_model(self):
        P = {i: {a: [] for a in ACTIONS} for i in range(self.n_states)}
        for i in range(self.n_states):
            s = self.i2s(i)
            if self.cfg.cell_type(*s) == Cell.WALL:
                for a in ACTIONS:
                    P[i][a] = [(1.0, i, 0.0, True)]
                continue
            if self.is_terminal(s):
                for a in ACTIONS:
                    P[i][a] = [(1.0, i, 0.0, True)]
                continue
            for a in ACTIONS:
                if self.cfg.cell_type(*s) == Cell.SLIPPERY:
                    branches = [(1.0 - self.slip_prob, a), (self.slip_prob, rotate_clockwise(a))]
                else:
                    branches = [(1.0, a)]
                merged = {}
                for prob, real_a in branches:
                    s_next = self._move(s, real_a)
                    reward = self._transition_reward(s, s_next)
                    done = self.is_terminal(s_next)
                    key = (self.s2i(s_next), reward, done)
                    merged[key] = merged.get(key, 0.0) + prob
                P[i][a] = [(p, ni, r, d) for (ni, r, d), p in merged.items()]
        return P

    # ---------- simulation interface (model-free rooms + episode replay) ----------
    def reset(self):
        self._state = self.cfg.start
        self._steps = 0
        return self.s2i(self._state)

    def step(self, a):
        self._steps += 1
        s = self._state
        real_a = a
        if self.cfg.cell_type(*s) == Cell.SLIPPERY:
            import random
            if random.random() < self.slip_prob:
                real_a = rotate_clockwise(a)
        s_next = self._move(s, real_a)
        reward = self._transition_reward(s, s_next)
        done = self.is_terminal(s_next)
        self._state = s_next
        truncated = self._steps >= self.max_steps
        return self.s2i(s_next), reward, done, truncated, {}


def default_config(rows=10, cols=10):
    """Room 1 default layout, matching the earlier fixed-grid version."""
    cfg = GridConfig(rows=rows, cols=cols, start=(0, 0), goal=(rows - 1, cols - 1))
    slippery = {(2, 3), (2, 4), (2, 5), (5, 1), (5, 2), (6, 6), (6, 7), (7, 7), (3, 8), (4, 8)}
    for r, c in slippery:
        if r < rows and c < cols:
            cfg.cells[(r, c)] = Cell.SLIPPERY
    cfg.cells[cfg.goal] = Cell.GOAL
    return cfg
