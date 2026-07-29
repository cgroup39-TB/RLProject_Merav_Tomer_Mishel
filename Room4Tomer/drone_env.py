"""
Continuous-state drone environment for Room 4 (Function Approximation / DQN).

State: [x, y, vx, vy], a 10x10 meter room. Actions are 9 discrete per-axis
commanded-velocity directions {-1, 0, 1}^2. The engine rate-limits actual
velocity toward the commanded value each step, and wind/accel/decel zones
can push the drone off that commanded value — which is why velocity is part
of the observable state, not just position.
"""

from dataclasses import dataclass, field

import numpy as np


ACTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
N_ACTIONS = len(ACTIONS)
HOVER_ACTION = ACTIONS.index((0, 0))


@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float

    def contains(self, px, py):
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def circle_overlaps(self, cx, cy, radius):
        nx = min(max(cx, self.x), self.x + self.w)
        ny = min(max(cy, self.y), self.y + self.h)
        return (cx - nx) ** 2 + (cy - ny) ** 2 <= radius ** 2


@dataclass
class WindZone(Rect):
    fx: float = 0.0
    fy: float = 0.0
    gust_std: float = 0.0


@dataclass
class AccelZone(Rect):
    fx: float = 0.0
    fy: float = 0.0


@dataclass
class DecelZone(Rect):
    drag_coeff: float = 1.0


# Gates are static obstacles, same collision behavior as walls -- kept as a
# separate DroneConfig field (rather than folded into `walls`) purely so the
# canvas can theme them differently (a "danger" red, matching the Gate
# Gauntlet preset's narrative) from ordinary factory walls.


@dataclass
class Pad:
    x: float
    y: float
    radius: float


@dataclass
class DroneConfig:
    room_w: float = 10.0
    room_h: float = 10.0
    start: tuple = (1.0, 1.0)
    # Kept well clear of the walls (>= ~2m even at the UI's max pad-radius
    # slider value) -- a pad flush against a corner means any tiny overshoot
    # while landing trips the out-of-bounds check, and a DQN agent reliably
    # learns to stay cautiously short of it rather than risk that crash.
    pad: Pad = field(default_factory=lambda: Pad(7.5, 7.5, 0.9))
    walls: list = field(default_factory=list)
    wind_zones: list = field(default_factory=list)
    accel_zones: list = field(default_factory=list)
    decel_zones: list = field(default_factory=list)
    gates: list = field(default_factory=list)


class DroneEnv:
    """Continuous drone-landing environment. reset()/step() mirror the
    (next_state, reward, done, truncated, info) shape used by Room 1's
    GridWorldEnv, though the internals are unrelated."""

    def __init__(
        self,
        config: DroneConfig,
        dt=0.02,
        max_cmd_speed=2.0,
        max_accel=10.0,
        max_abs_speed=3.0,
        drone_radius=0.15,
        landing_speed_threshold=1.0,
        step_reward=-0.01,
        shaping_weight=3.0,
        crash_penalty=-30.0,
        landing_bonus_base=100.0,
        landing_bonus_gentleness=50.0,
        max_steps=500,
        seed=None,
    ):
        self.cfg = config
        self.dt = dt
        self.max_cmd_speed = max_cmd_speed
        self.max_accel = max_accel
        self.max_abs_speed = max_abs_speed
        self.drone_radius = drone_radius
        self.landing_speed_threshold = landing_speed_threshold
        self.step_reward = step_reward
        self.shaping_weight = shaping_weight
        self.crash_penalty = crash_penalty
        self.landing_bonus_base = landing_bonus_base
        self.landing_bonus_gentleness = landing_bonus_gentleness
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.n_actions = N_ACTIONS
        self.state_dim = 4

        self._state = None
        self._steps = 0
        self._t = 0.0

    def reset(self):
        x, y = self.cfg.start
        self._state = np.array([x, y, 0.0, 0.0], dtype=np.float64)
        self._steps = 0
        self._t = 0.0
        return self._state.copy()

    def _potential(self, x, y):
        return -float(np.hypot(x - self.cfg.pad.x, y - self.cfg.pad.y))


    def step(self, action):
        x, y, vx, vy = self._state
        ax_cmd, ay_cmd = ACTIONS[action]

        # 1) engine rate-limited response toward commanded velocity
        max_dv = self.max_accel * self.dt
        vx += np.clip(ax_cmd * self.max_cmd_speed - vx, -max_dv, max_dv)
        vy += np.clip(ay_cmd * self.max_cmd_speed - vy, -max_dv, max_dv)

        # 2) external zone forces (applied on top of engine response)
        wind_vec = (0.0, 0.0)
        for z in self.cfg.wind_zones:
            if z.contains(x, y):
                gx = self.rng.normal(0.0, z.gust_std) if z.gust_std > 0 else 0.0
                gy = self.rng.normal(0.0, z.gust_std) if z.gust_std > 0 else 0.0
                vx += (z.fx + gx) * self.dt
                vy += (z.fy + gy) * self.dt
                wind_vec = (z.fx, z.fy)
        for z in self.cfg.accel_zones:
            if z.contains(x, y):
                vx += z.fx * self.dt
                vy += z.fy * self.dt
        for z in self.cfg.decel_zones:
            if z.contains(x, y):
                vx -= vx * z.drag_coeff * self.dt
                vy -= vy * z.drag_coeff * self.dt

        # 3) clip resultant speed to the safety ceiling
        speed = float(np.hypot(vx, vy))
        if speed > self.max_abs_speed:
            scale = self.max_abs_speed / speed
            vx *= scale
            vy *= scale

        # 4) integrate position (semi-implicit Euler)
        x_new = x + vx * self.dt
        y_new = y + vy * self.dt

        self._steps += 1
        self._t += self.dt

        crashed = False
        landed = False

        if not (0.0 <= x_new <= self.cfg.room_w and 0.0 <= y_new <= self.cfg.room_h):
            crashed = True
            x_new = float(np.clip(x_new, 0.0, self.cfg.room_w))
            y_new = float(np.clip(y_new, 0.0, self.cfg.room_h))

        if not crashed:
            for w in self.cfg.walls:
                if w.circle_overlaps(x_new, y_new, self.drone_radius):
                    crashed = True
                    break

        if not crashed:
            for g in self.cfg.gates:
                if g.circle_overlaps(x_new, y_new, self.drone_radius):
                    crashed = True
                    break

        speed_new = float(np.hypot(vx, vy))

        if not crashed:
            dist_to_pad = np.hypot(x_new - self.cfg.pad.x, y_new - self.cfg.pad.y)
            if dist_to_pad <= self.cfg.pad.radius:
                if speed_new <= self.landing_speed_threshold:
                    landed = True
                else:
                    crashed = True  # reached the pad too fast: hard landing

        self._state = np.array([x_new, y_new, vx, vy], dtype=np.float64)

        done = crashed or landed
        truncated = (not done) and (self._steps >= self.max_steps)

        reward = self.step_reward
        reward += self.shaping_weight * (self._potential(x_new, y_new) - self._potential(x, y))
        if crashed:
            reward += self.crash_penalty
        if landed:
            gentleness = float(np.clip(1.0 - speed_new / max(self.landing_speed_threshold, 1e-6), 0.0, 1.0))
            reward += self.landing_bonus_base + self.landing_bonus_gentleness * gentleness

        info = dict(
            crashed=crashed,
            landed=landed,
            speed=speed_new,
            wind=wind_vec,
            gate_positions=[(r.x + r.w / 2, r.y + r.h / 2) for r in self.cfg.gates],
            t=self._t,
        )
        return self._state.copy(), float(reward), bool(done), bool(truncated), info


