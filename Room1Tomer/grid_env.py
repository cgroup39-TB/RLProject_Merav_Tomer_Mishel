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
    # Room 3 (Energy Room) additions.
    BATTERY = "battery"
    SWITCH_BLUE = "switch_blue"
    SWITCH_RED = "switch_red"
    ELECTRIC_TRAP = "electric_trap"
    DOOR = "door"
    CHARGER = "charger"
    SHORTCUT = "shortcut"


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
    guard_path: list = field(default_factory=list)      # Room 3: ordered patrol waypoints, [] = no guard

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


class EnergyRoomEnv(GridWorldEnv):
    """
    Room 3 -- The Energy Room. Extends GridWorldEnv with the flag-based
    state Q-Learning needs for a job that isn't just "reach the goal":
    collect the battery, then throw the blue switch, then the red switch --
    each one gates the next, and the exit itself is locked until all three
    are done -- plus a patrolling guard robot and an electric trap that's
    only live on even steps.

    State = (row, col, has_battery, blue_switch_on, red_switch_on), encoded
    as pos_index * 8 + flag_bits so the tabular Q-Learning solver (which
    only ever calls reset()/step(), same contract as SARSA in Room 2) can
    index a plain (n_states, n_actions) table. No known transition model is
    provided for this room -- Q-Learning doesn't need one.

    Every bonus/penalty below is added on top of the room's step_reward,
    the same convention Room 2's key bonus used: "-1 per step" is a
    baseline that always applies, and events like collecting the battery
    or hitting the guard add to it rather than replacing it.
    """

    def __init__(
        self,
        config,
        guard_path=None,
        battery_reward=15.0,
        switch_blue_reward=20.0,
        switch_red_reward=25.0,
        guard_penalty=-30.0,
        wall_penalty=-2.0,
        electric_trap_reward=-20.0,
        charger_reward=5.0,
        incomplete_exit_reward=-10.0,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.battery_reward = battery_reward
        self.switch_blue_reward = switch_blue_reward
        self.switch_red_reward = switch_red_reward
        self.guard_penalty = guard_penalty
        self.wall_penalty = wall_penalty
        self.electric_trap_reward = electric_trap_reward
        self.charger_reward = charger_reward
        self.incomplete_exit_reward = incomplete_exit_reward
        self.n_states = self.rows * self.cols * 8

        path = list(guard_path or [])
        # Ping-pong the patrol back and forth along its waypoints instead of
        # teleporting from the last one back to the first.
        self._guard_cycle = path + path[-2:0:-1] if len(path) > 1 else path

        self._has_battery = False
        self._blue_on = False
        self._red_on = False
        self._charger_visited = set()

    # ---------- flag-state encoding ----------
    def _flag_bits(self):
        return (int(self._has_battery) << 2) | (int(self._blue_on) << 1) | int(self._red_on)

    def _encode(self, pos):
        return self.s2i(pos) * 8 + self._flag_bits()

    def decode_flags(self, full_state):
        """full_state -> ((row, col), has_battery, blue_switch_on, red_switch_on)."""
        pos_i, bits = divmod(full_state, 8)
        return self.i2s(pos_i), bool(bits >> 2 & 1), bool(bits >> 1 & 1), bool(bits & 1)

    # ---------- guard patrol ----------
    def guard_pos(self):
        if not self._guard_cycle:
            return None
        return self._guard_cycle[self._steps % len(self._guard_cycle)]

    # ---------- passability, incl. the conditional door/shortcut ----------
    def _passable(self, r, c):
        t = self.cfg.cell_type(r, c)
        if t == Cell.WALL:
            return False
        if t == Cell.DOOR and not self._blue_on:
            return False
        if t == Cell.SHORTCUT and not self._has_battery:
            return False
        return True

    # ---------- simulation interface ----------
    def reset(self):
        self._state = self.cfg.start
        self._steps = 0
        self._has_battery = False
        self._blue_on = False
        self._red_on = False
        self._charger_visited = set()
        return self._encode(self._state)

    def step(self, a):
        self._steps += 1
        s = self._state
        real_a = a
        if self.cfg.cell_type(*s) == Cell.SLIPPERY:
            if self.rng.random() < self._slip_prob(s):
                real_a = self._slip_outcome_action(s, a)
        dr, dc = ACTION_DELTAS[real_a]
        nr, nc = s[0] + dr, s[1] + dc
        blocked = not self.in_bounds(nr, nc) or not self._passable(nr, nc)
        s_next = s if blocked else (nr, nc)

        guard_here = self.guard_pos()
        collided = guard_here is not None and s_next == guard_here
        if collided:
            s_next = s

        reward = self.step_reward
        if blocked:
            reward += self.wall_penalty
        if collided:
            reward += self.guard_penalty

        done = False
        if not collided:
            t = self.cfg.cell_type(*s_next)
            if t == Cell.BATTERY and not self._has_battery:
                self._has_battery = True
                reward += self.battery_reward
            elif t == Cell.SWITCH_BLUE and self._has_battery and not self._blue_on:
                self._blue_on = True
                reward += self.switch_blue_reward
            elif t == Cell.SWITCH_RED and self._blue_on and not self._red_on:
                self._red_on = True
                reward += self.switch_red_reward
            elif t == Cell.CHARGER and s_next not in self._charger_visited:
                # One-time per episode -- otherwise a repeatable reward
                # bigger than the round-trip step cost turns into a farming
                # loop instead of an incentive to actually finish the room.
                # Like the trap's step-parity, "already visited" isn't part
                # of the learned state, only of the reward this step.
                self._charger_visited.add(s_next)
                reward += self.charger_reward
            elif t == Cell.ELECTRIC_TRAP and self._steps % 2 == 0:
                # Only live on even steps -- the state deliberately doesn't
                # carry that parity (see the room's state tuple above), so
                # the agent can't learn to time it exactly; it has to learn
                # to be cautious near it instead.
                reward += self.electric_trap_reward
                done = True
            elif t == Cell.TRAP:
                reward += self.trap_reward
                done = True
            elif t == Cell.GOAL:
                if self._has_battery and self._blue_on and self._red_on:
                    reward += self.goal_reward
                    done = True
                else:
                    reward += self.incomplete_exit_reward

        self._state = s_next
        truncated = self._steps >= self.max_steps
        return self._encode(s_next), reward, done, truncated, {}


def default_room3_config(rows=10, cols=10):
    """
    Room 3 "Energy Room" default layout: a machine-room floor split by one
    control wall (row 6) into a front half -- start, battery, blue switch --
    and a back half -- red switch, exit -- reachable only through the DOOR
    gap (open once the blue switch is thrown) or the SHORTCUT gap (open
    once the battery is held). A guard robot patrols the corridor just past
    the wall, and two electric traps flicker along the same stretch, live
    only on even steps. Hand-designed for exactly 10x10, like Room 2's
    layout -- it doesn't attempt to degrade to other grid sizes.
    """
    layout = [
        "S........K",
        "..........",
        "..........",
        "....C.....",
        "..........",
        "U.........",
        "#####D#H##",
        ".E....E...",
        "..........",
        ".....Z...G",
    ]
    symbol_to_cell = {
        "#": Cell.WALL,
        "K": Cell.BATTERY,
        "U": Cell.SWITCH_BLUE,
        "Z": Cell.SWITCH_RED,
        "D": Cell.DOOR,
        "H": Cell.SHORTCUT,
        "C": Cell.CHARGER,
        "E": Cell.ELECTRIC_TRAP,
    }

    cfg = GridConfig(rows=10, cols=10, start=(0, 0), goal=(9, 9))
    for r, row in enumerate(layout):
        for c, ch in enumerate(row):
            if ch in symbol_to_cell:
                cfg.cells[(r, c)] = symbol_to_cell[ch]
    cfg.cells[cfg.goal] = Cell.GOAL
    cfg.guard_path = [(8, c) for c in range(1, 9)]
    return cfg


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
