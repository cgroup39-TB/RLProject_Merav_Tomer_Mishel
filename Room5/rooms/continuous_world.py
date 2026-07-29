"""
The continuous room used by rooms 4 and 5.

There is no grid here. The room is a square of 10 x 10 METRES and the agent can
stand anywhere inside it. Its state is four numbers:

    x, y    -> where it is, in metres
    vx, vy  -> how it is currently moving

Time moves in ticks of dt = 0.02 seconds (this number is fixed by the
assignment). Every tick the agent moves x += vx * dt and y += vy * dt.

The speed is discrete: vx and vy can only be -1, 0 or 1 metres per second.
That gives 3 x 3 = 9 possible actions (including "stand still").

Because 0.02 seconds is very short, crossing the room would take 500 ticks.
To keep training fast the agent chooses a new direction once every
`decision_interval` ticks (5 by default) and keeps that direction in between.
The physics still runs at the required 0.02 seconds per tick.

Room 5 adds obstacles: squares 0.5 m wide, whose number and positions are drawn
again at the start of every episode. The agent does not see the whole room - it
only sees obstacles up to `lookahead_m` metres away, through a small "radar" of
`n_rays` distance readings around it.
"""

import math
import random

import numpy as np

from rooms.base_room import BaseRoom

DT = 0.02            # seconds per physics tick (fixed by the assignment)
ARENA_SIZE = 10.0    # the room is 10 x 10 metres
AGENT_RADIUS = 0.2   # the agent is a small circle, measured from its centre

# The 9 actions: every combination of vx and vy from {-1, 0, 1}.
SPEED_CHOICES = [-1, 0, 1]
ACTIONS = [(vx, vy) for vx in SPEED_CHOICES for vy in SPEED_CHOICES]
ACTION_NAMES = [f"vx={vx:+d}, vy={vy:+d}" for vx, vy in ACTIONS]


class ContinuousRoom(BaseRoom):
    """The 10x10 metre room for rooms 4 and 5. Obstacles are optional."""

    def __init__(self,
                 n_obstacles=0,
                 obstacle_size=0.5,
                 lookahead_m=3.0,
                 n_rays=8,
                 goal_radius=0.7,
                 goal_clearance=2.0,
                 decision_interval=5,
                 max_seconds=40.0,
                 time_cost=-1.0,
                 goal_reward=100.0,
                 wall_reward=-1.0,
                 crash_reward=-50.0,
                 progress_reward=1.0,
                 reverse_cost=-2.0):
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

        # Room 4 has no obstacles, so it has no radar either.
        self.uses_radar = n_obstacles > 0

        # The radar directions never change, so we build them once here:
        # n_rays arrows spread evenly all around the agent.
        angles = [2 * math.pi * i / n_rays for i in range(n_rays)]
        self.ray_directions = np.array([[math.cos(a), math.sin(a)] for a in angles])

        # How far along each arrow we look for an obstacle (every 25 cm).
        self.ray_samples = np.arange(0.25, lookahead_m + 0.001, 0.25)

        self.obstacles = np.zeros((0, 2))
        self.reset()

    # ------------------------------------------------------------------
    # Sizes that the learning algorithm needs to know
    # ------------------------------------------------------------------

    @property
    def max_decisions(self):
        """How many decisions fit into one episode before we give up."""
        seconds_per_decision = DT * self.decision_interval
        return int(self.max_seconds / seconds_per_decision)

    @property
    def observation_size(self):
        """x, y, vx, vy - plus one distance per radar ray in room 5."""
        if self.uses_radar:
            return 4 + self.n_rays
        return 4

    # ------------------------------------------------------------------
    # Building a room
    # ------------------------------------------------------------------

    def random_obstacles(self):
        """
        Scatter the obstacles at random. This runs at the start of every episode,
        which is what makes room 5 "dynamic": the agent can never memorise one
        fixed layout, it has to learn to react to what its radar sees.
        """
        centres = []
        while len(centres) < self.n_obstacles:
            x = random.uniform(1.5, ARENA_SIZE - 1.5)
            y = random.uniform(1.5, ARENA_SIZE - 1.5)
            distance_to_start = math.dist((x, y), self.start)
            distance_to_goal = math.dist((x, y), self.goal)
            # Keep a clear ring around the start and a wider one around the exit.
            # The exit sits in a corner, so blocks dropped right in front of it can
            # wall it off completely and make the room impossible - which would be
            # an unfair test rather than a hard one.
            if distance_to_start < 1.2 or distance_to_goal < self.goal_clearance:
                continue
            centres.append((x, y))
        return np.array(centres).reshape(len(centres), 2)

    def reset(self):
        """Start a new episode: agent back at the start, new obstacle layout."""
        self.x, self.y = self.start
        self.vx = 0
        self.vy = 0
        self.time = 0.0
        self.obstacles = self.random_obstacles()
        return self.observation()

    # ------------------------------------------------------------------
    # What the agent can see
    # ------------------------------------------------------------------

    def radar(self):
        """
        For every radar direction: how far away is the nearest obstacle?
        If nothing is found within `lookahead_m`, the answer is `lookahead_m`
        ("the way is clear").

        We simply walk along each arrow in small hops and check whether that
        point is inside an obstacle square. numpy does all the arrows and all
        the hops at once, which keeps it fast.
        """
        if len(self.obstacles) == 0:
            return np.full(self.n_rays, self.lookahead_m)

        here = np.array([self.x, self.y])
        # points[ray, hop] = the position of that hop along that arrow
        points = here + self.ray_directions[:, None, :] * self.ray_samples[None, :, None]

        half = self.obstacle_size / 2 + AGENT_RADIUS
        distance_to_centres = np.abs(points[:, :, None, :] - self.obstacles[None, None, :, :])
        inside_square = (distance_to_centres <= half).all(axis=3)   # [ray, hop, obstacle]
        hit = inside_square.any(axis=2)                             # [ray, hop]

        found_something = hit.any(axis=1)
        first_hop = hit.argmax(axis=1)
        return np.where(found_something, self.ray_samples[first_hop], self.lookahead_m)

    def observation(self):
        """Everything the agent knows right now, as a list of numbers."""
        basic = [self.x, self.y, self.vx, self.vy]
        if not self.uses_radar:
            return np.array(basic)
        return np.concatenate([basic, self.radar()])

    # ------------------------------------------------------------------
    # The rules of the room
    # ------------------------------------------------------------------

    def hits_obstacle(self):
        """Is the agent currently touching an obstacle?"""
        if len(self.obstacles) == 0:
            return False
        half = self.obstacle_size / 2 + AGENT_RADIUS
        here = np.array([self.x, self.y])
        distance_to_centres = np.abs(self.obstacles - here)
        inside_square = (distance_to_centres <= half).all(axis=1)
        return bool(inside_square.any())

    def at_goal(self):
        """Has the agent reached the exit?"""
        distance = math.dist((self.x, self.y), self.goal)
        return distance <= self.goal_radius

    def step(self, action):
        """
        Keep the chosen direction for `decision_interval` physics ticks.
        Returns (observation, reward, done, info).

        The reward is mostly "minus one point per second", so the faster the
        agent escapes, the more reward it keeps. On top of that there is a small
        reward for every metre it gets CLOSER to the exit (see progress_reward
        in __init__) - without it the agent would almost never bump into the
        exit by accident and would have nothing to learn from.
        """
        new_vx, new_vy = ACTIONS[action]
        reward = 0.0
        done = False
        bumped_a_wall = False

        # Turning a full 180 degrees wastes time, so it costs a little.
        # This is also the reason the speed belongs in the state: how good a
        # situation is depends on which way the agent is already going. Without
        # this cost a greedy agent can get stuck flipping between two opposite
        # directions forever instead of walking to the exit.
        standing_still = self.vx == 0 and self.vy == 0
        turning_back = new_vx == -self.vx and new_vy == -self.vy
        if turning_back and not standing_still:
            reward += self.reverse_cost

        self.vx = new_vx
        self.vy = new_vy

        for _ in range(self.decision_interval):
            distance_before = math.dist((self.x, self.y), self.goal)
            self.x += self.vx * DT
            self.y += self.vy * DT
            self.time += DT

            # The room has walls: the agent cannot leave it.
            clamped_x = min(max(self.x, AGENT_RADIUS), ARENA_SIZE - AGENT_RADIUS)
            clamped_y = min(max(self.y, AGENT_RADIUS), ARENA_SIZE - AGENT_RADIUS)
            if clamped_x != self.x or clamped_y != self.y:
                self.x = clamped_x
                self.y = clamped_y
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

        info = {"time": self.time,
                "crashed": self.hits_obstacle(),
                "bumped_a_wall": bumped_a_wall}
        return self.observation(), reward, done, info

    def escaped(self):
        """Did the agent finish through the exit?"""
        return self.at_goal()

    def get_state(self):
        """What the agent can see: its position, its speed, and the radar."""
        return self.observation()

    def is_terminal(self, state=None):
        """The episode ends at the exit or in a crash."""
        return self.at_goal() or self.hits_obstacle()

    def render(self):
        """Where everything is right now. draw_arena() turns this into a picture."""
        return self.frame()

    def frame(self):
        """A snapshot of the room, used for drawing and for replays."""
        return {"x": self.x,
                "y": self.y,
                "vx": self.vx,
                "vy": self.vy,
                "time": self.time,
                "obstacles": self.obstacles.copy()}
