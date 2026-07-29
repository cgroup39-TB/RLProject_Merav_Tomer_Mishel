"""
Room 5 -- The Shifting Warehouse: a continuous 10x10 metre room, solved with
function approximation instead of a table.

There is no grid here. The agent (a drone) can stand anywhere inside a
10x10 metre square. Its state is x, y (position in metres) and vx, vy
(current speed, one of -1/0/+1 m/s each) -- 4 numbers, not a (row, col)
pair, so there is no way to keep one Q-value per state anymore.

On top of that, crate-shaped obstacles are scattered at random, in new
positions every single episode, and the agent doesn't see the whole room --
only a short-range radar (`n_rays` distance readings around it, out to
`lookahead_m`). Nothing about this room can be memorised; the only thing
worth learning is a rule about what the radar says ("the reading dead
ahead is short -> that direction is bad"), which is exactly the kind of
rule that keeps working in a layout the agent has never seen. That's the
generalisation test the room is built around.

Physics runs in fixed 0.02s ticks (DT), but the agent only picks a new
direction once every `decision_interval` ticks -- that's what an "action"
means here, one held direction, not one tick.
"""

import math

import numpy as np

DT = 0.02
ARENA_SIZE = 10.0
AGENT_RADIUS = 0.2

SPEED_CHOICES = (-1, 0, 1)
ACTIONS = [(vx, vy) for vx in SPEED_CHOICES for vy in SPEED_CHOICES]
ACTION_NAMES = [f"vx={vx:+d}, vy={vy:+d}" for vx, vy in ACTIONS]
N_ACTIONS = len(ACTIONS)


class WarehouseRoom:
    """The 10x10 metre room for Room 5, with randomly-placed obstacles."""

    def __init__(
        self,
        n_obstacles=6,
        obstacle_size=0.5,
        lookahead_m=2.0,
        n_rays=16,
        goal_radius=0.7,
        goal_clearance=2.0,
        decision_interval=25,
        max_seconds=25.0,
        time_cost=-1.0,
        goal_reward=100.0,
        wall_reward=-1.0,
        crash_reward=-50.0,
        progress_reward=4.0,
        reverse_cost=-2.0,
        seed=None,
    ):
        self.n_obstacles = n_obstacles
        self.obstacle_size = obstacle_size
        self.lookahead_m = lookahead_m
        self.n_rays = n_rays
        self.goal_radius = goal_radius
        self.goal_clearance = goal_clearance
        self.decision_interval = decision_interval
        self.max_seconds = max_seconds
        self.time_cost = time_cost
        self.goal_reward = goal_reward
        self.wall_reward = wall_reward
        self.crash_reward = crash_reward
        self.progress_reward = progress_reward
        self.reverse_cost = reverse_cost

        self.start = (1.0, 1.0)
        self.goal = (9.0, 9.0)
        self.rng = np.random.default_rng(seed)

        angles = [2 * math.pi * i / n_rays for i in range(n_rays)]
        self.ray_directions = np.array([[math.cos(a), math.sin(a)] for a in angles])
        self.ray_samples = np.arange(0.25, lookahead_m + 0.001, 0.25)

        self.obstacles = np.zeros((0, 2))
        self.x = self.y = self.vx = self.vy = 0.0
        self.time = 0.0

    @property
    def max_decisions(self):
        seconds_per_decision = DT * self.decision_interval
        return int(self.max_seconds / seconds_per_decision)

    @property
    def observation_size(self):
        return 4 + self.n_rays

    def random_obstacles(self):
        """
        Scatter obstacles at random, keeping a clear ring around the start
        and a wider one around the goal (a corner, so a block dropped right
        in front of it could wall it off completely -- that would make the
        room impossible rather than merely hard).
        """
        centres = []
        attempts = 0
        while len(centres) < self.n_obstacles and attempts < 10000:
            attempts += 1
            x = self.rng.uniform(1.5, ARENA_SIZE - 1.5)
            y = self.rng.uniform(1.5, ARENA_SIZE - 1.5)
            if math.dist((x, y), self.start) < 1.2 or math.dist((x, y), self.goal) < self.goal_clearance:
                continue
            centres.append((x, y))
        return np.array(centres).reshape(len(centres), 2)

    def reset(self):
        self.x, self.y = self.start
        self.vx = 0
        self.vy = 0
        self.time = 0.0
        self.obstacles = self.random_obstacles()
        return self.observation()

    def radar(self):
        """
        Distance to the nearest obstacle along each ray, capped at
        `lookahead_m` ("the way is clear" reading beyond that range).
        """
        if len(self.obstacles) == 0:
            return np.full(self.n_rays, self.lookahead_m)

        here = np.array([self.x, self.y])
        points = here + self.ray_directions[:, None, :] * self.ray_samples[None, :, None]

        half = self.obstacle_size / 2 + AGENT_RADIUS
        distance_to_centres = np.abs(points[:, :, None, :] - self.obstacles[None, None, :, :])
        inside_square = (distance_to_centres <= half).all(axis=3)
        hit = inside_square.any(axis=2)

        found_something = hit.any(axis=1)
        first_hop = hit.argmax(axis=1)
        return np.where(found_something, self.ray_samples[first_hop], self.lookahead_m)

    def observation(self):
        return np.concatenate([[self.x, self.y, self.vx, self.vy], self.radar()])

    def hits_obstacle(self):
        if len(self.obstacles) == 0:
            return False
        half = self.obstacle_size / 2 + AGENT_RADIUS
        here = np.array([self.x, self.y])
        distance_to_centres = np.abs(self.obstacles - here)
        return bool((distance_to_centres <= half).all(axis=1).any())

    def at_goal(self):
        return math.dist((self.x, self.y), self.goal) <= self.goal_radius

    def step(self, action):
        """
        Hold the chosen direction for `decision_interval` ticks. Reward is
        mostly "-1 point per second" plus a small bonus per metre closer to
        the goal (without it the agent almost never bumps into the exit by
        accident and has nothing to learn from) and a small cost for
        reversing direction outright (otherwise a greedy agent can get
        stuck flipping between two opposite headings forever).
        """
        new_vx, new_vy = ACTIONS[action]
        reward = 0.0
        done = False
        bumped_a_wall = False

        standing_still = self.vx == 0 and self.vy == 0
        turning_back = new_vx == -self.vx and new_vy == -self.vy
        if turning_back and not standing_still:
            reward += self.reverse_cost

        self.vx, self.vy = new_vx, new_vy

        for _ in range(self.decision_interval):
            distance_before = math.dist((self.x, self.y), self.goal)
            self.x += self.vx * DT
            self.y += self.vy * DT
            self.time += DT

            clamped_x = min(max(self.x, AGENT_RADIUS), ARENA_SIZE - AGENT_RADIUS)
            clamped_y = min(max(self.y, AGENT_RADIUS), ARENA_SIZE - AGENT_RADIUS)
            if clamped_x != self.x or clamped_y != self.y:
                self.x, self.y = clamped_x, clamped_y
                bumped_a_wall = True

            distance_after = math.dist((self.x, self.y), self.goal)
            reward += self.time_cost * DT
            reward += self.progress_reward * (distance_before - distance_after)

            if self.hits_obstacle():
                reward += self.crash_reward
                done = True
                break
            if self.at_goal():
                reward += self.goal_reward
                done = True
                break

        if bumped_a_wall:
            reward += self.wall_reward

        return self.observation(), reward, done, {"crashed": self.hits_obstacle()}

    def escaped(self):
        return self.at_goal()

    def frame(self):
        """A snapshot for drawing/replay: position, speed, time, obstacle layout."""
        return {
            "x": self.x, "y": self.y, "vx": self.vx, "vy": self.vy,
            "time": self.time, "obstacles": self.obstacles.copy(),
        }
