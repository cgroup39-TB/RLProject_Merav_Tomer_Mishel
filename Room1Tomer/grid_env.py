"""
Generalized, editable Grid World environment shared by the DP / SARSA /
Q-Learning rooms. The grid layout (walls, slippery cells, traps, start, goal)
and reward parameters are fully configurable at runtime from the UI.
"""

from dataclasses import dataclass, field

import numpy as np


class Cell:
    EMPTY = "empty"
    WALL = "wall"
    SLIPPERY = "slippery"
    GOAL = "goal"
    TRAP = "trap"
    START = "start"
    LASER = "laser"


class Action:
    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


ACTIONS = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
ACTION_DELTAS = {Action.UP: (-1, 0), Action.DOWN: (1, 0), Action.LEFT: (0, -1), Action.RIGHT: (0, 1)}
ACTION_ARROWS = {Action.UP: "↑", Action.DOWN: "↓", Action.LEFT: "←", Action.RIGHT: "→"}

_CLOCKWISE = [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]

# Per-cell slip-direction modes: "cw"/"ccw" rotate the intended action;
# the four cardinal strings always slip toward that fixed direction.
_FIXED_SLIP_DIRS = {"up": Action.UP, "down": Action.DOWN, "left": Action.LEFT, "right": Action.RIGHT}
SLIP_DIR_MODES = ["cw", "ccw", "up", "down", "left", "right"]
SLIP_DIR_LABELS = {
    "cw": "↻ Rotate 90° clockwise (default)",
    "ccw": "↺ Rotate 90° counter-clockwise",
    "up": "↑ Always up",
    "down": "↓ Always down",
    "left": "← Always left",
    "right": "→ Always right",
}


def rotate_clockwise(a):
    return _CLOCKWISE[(_CLOCKWISE.index(a) + 1) % 4]


def rotate_counterclockwise(a):
    return _CLOCKWISE[(_CLOCKWISE.index(a) - 1) % 4]


@dataclass
class GridConfig:
    rows: int = 10
    cols: int = 10
    cells: dict = field(default_factory=dict)          # (r,c) -> Cell.* type, default EMPTY
    cell_rewards: dict = field(default_factory=dict)    # (r,c) -> reward override (paint-time)
    cell_slip_prob: dict = field(default_factory=dict)  # (r,c) -> slip probability override (SLIPPERY cells)
    cell_slip_dir: dict = field(default_factory=dict)   # (r,c) -> slip direction mode (see SLIP_DIR_MODES)
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
        laser_reward=-20.0,
        gamma=0.95,
        potential_shaping=False,
        max_steps=500,
        seed=None,
    ):
        self.cfg = config
        self.slip_prob = slip_prob
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.trap_reward = trap_reward
        self.laser_reward = laser_reward
        self.gamma = gamma
        self.potential_shaping = potential_shaping
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

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
        if t == Cell.LASER:
            return self.laser_reward
        return self.step_reward

    def _transition_reward(self, s, s_next):
        # Laser cells are non-terminal but still charge their penalty on entry,
        # same as terminal cells charge their reward on entry.
        if self.is_terminal(s_next) or self.cfg.cell_type(*s_next) == Cell.LASER:
            reward = self._base_reward(s_next)
        else:
            reward = self._base_reward(s)
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

    # ---------- per-cell slip lookups ----------
    def _slip_prob(self, s):
        return self.cfg.cell_slip_prob.get(s, self.slip_prob)

    def _slip_outcome_action(self, s, a):
        mode = self.cfg.cell_slip_dir.get(s, "cw")
        if mode == "cw":
            return rotate_clockwise(a)
        if mode == "ccw":
            return rotate_counterclockwise(a)
        return _FIXED_SLIP_DIRS.get(mode, rotate_clockwise(a))

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
                    sp = self._slip_prob(s)
                    branches = [(1.0 - sp, a), (sp, self._slip_outcome_action(s, a))]
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
            if self.rng.random() < self._slip_prob(s):
                real_a = self._slip_outcome_action(s, a)
        s_next = self._move(s, real_a)
        reward = self._transition_reward(s, s_next)
        done = self.is_terminal(s_next)
        self._state = s_next
        truncated = self._steps >= self.max_steps
        return self.s2i(s_next), reward, done, truncated, {}


def default_config(rows=10, cols=10):
    """
    Room 1 "Laser Room" default layout: a metal-wall maze with two routes
    from start to goal — a direct staircase lined with lasers (short,
    dangerous) and a perimeter corridor (top row + right column) that stays
    clear except for two slippery ice patches with distinct probability and
    slip direction (long, safe). Degrades gracefully at any grid size.
    """
    cfg = GridConfig(rows=rows, cols=cols, start=(0, 0), goal=(rows - 1, cols - 1))

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in (cfg.start, cfg.goal):
                cfg.cells[(r, c)] = Cell.WALL

    safe_route = {(0, c) for c in range(cols)} | {(r, cols - 1) for r in range(rows)}
    for cell in safe_route:
        if cell not in (cfg.start, cfg.goal):
            cfg.cells[cell] = Cell.EMPTY

    r, c = cfg.start
    gr, gc = cfg.goal
    go_right = True
    staircase = [(r, c)]
    while (r, c) != (gr, gc):
        if go_right and c < gc:
            c += 1
        elif r < gr:
            r += 1
        elif c < gc:
            c += 1
        go_right = not go_right
        staircase.append((r, c))
    for cell in staircase:
        if cell in (cfg.start, cfg.goal) or cell in safe_route:
            continue
        cfg.cells[cell] = Cell.LASER

    if cols > 3:
        ice1 = (0, cols // 3)
        if ice1 not in (cfg.start, cfg.goal):
            cfg.cells[ice1] = Cell.SLIPPERY
            cfg.cell_slip_prob[ice1] = 0.15
            cfg.cell_slip_dir[ice1] = "cw"
    if rows > 3:
        ice2 = (2 * rows // 3, cols - 1)
        if ice2 not in (cfg.start, cfg.goal):
            cfg.cells[ice2] = Cell.SLIPPERY
            cfg.cell_slip_prob[ice2] = 0.5
            cfg.cell_slip_dir[ice2] = "down"

    cfg.cells[cfg.goal] = Cell.GOAL
    return cfg
