"""
Rooms 4 and 5: learning with an APPROXIMATE function.

In rooms 1-3 we could keep one number for every state, because there were only
100 or 200 states. In the continuous room the agent can stand anywhere, so there
are infinitely many states and no table has a row for them.

(You CAN invent rows, by rounding the position to the nearest square. That is
what discrete_agent.py does, and the experiment on the room 4 page measures how
far it gets - honestly, in room 4 it gets surprisingly far. Where it stops being
possible at all is room 5, once the radar is part of what the agent sees.)

The trick: describe the situation with a list of simple yes/no features, and
estimate the value as a weighted sum of those features:

    q(state, action) = weights[action] . features(state)

Learning then means adjusting the weights instead of filling a table. Nearby
positions share features, so what the agent learns at one spot automatically
helps at the spots around it. That is the whole point of "generalisation".

HOW WE BUILD THE FEATURES
-------------------------
All our features are either 0 or 1, and only a handful are 1 at any moment. So
instead of storing a long vector full of zeros we just store the LIST OF
POSITIONS that are 1 (see Features.active). Summing a few weights is then very
fast.

Part 1 - tile coding of the position (x, y):
    We lay a coarse grid of squares over the room and mark the square the agent
    is in. Then we do it again with a few more grids that are slightly shifted.
    Several overlapping coarse grids give smooth, cheap generalisation - this is
    the classic "tile coding" from the textbook.

Part 2 - the speed:
    vx and vy can only be -1, 0 or 1, so we simply mark which one it is.

Part 3 - the radar (room 5 only):
    Each radar distance is put into one of a few "bins" (very close / close /
    far / clear) and we mark that bin. Every action has its own weights, so the
    agent can learn things like "when the radar to my right is very short, going
    right is a bad idea" - and that rule keeps working in obstacle layouts it has
    never seen before.
"""

import random

import numpy as np

from rooms.continuous_world import ACTIONS, ARENA_SIZE
from training.metrics import EpisodeLog


class Features:
    """Turns an observation into the list of feature positions that are 1."""

    def __init__(self, room, n_tilings=4, tiles=8, ray_bins=4):
        self.room = room
        self.n_tilings = n_tilings
        self.tiles = tiles
        self.ray_bins = ray_bins

        self.tile_width = ARENA_SIZE / tiles

        # Where each part of the feature list starts.
        self.position_part = n_tilings * tiles * tiles
        self.speed_part = 3 + 3
        self.radar_part = room.n_rays * ray_bins if room.uses_radar else 0

        self.speed_start = self.position_part
        self.radar_start = self.speed_start + self.speed_part
        self.size = self.radar_start + self.radar_part

    def active(self, observation):
        """The positions in the feature list that are 1 for this observation."""
        x = observation[0]
        y = observation[1]
        vx = int(observation[2])
        vy = int(observation[3])
        indices = []

        # Part 1: which square am I in, on each of the shifted grids?
        for tiling in range(self.n_tilings):
            offset = self.tile_width * tiling / self.n_tilings
            column = int((x + offset) / self.tile_width)
            row = int((y + offset) / self.tile_width)
            column = min(column, self.tiles - 1)
            row = min(row, self.tiles - 1)
            indices.append(tiling * self.tiles * self.tiles + row * self.tiles + column)

        # Part 2: my current speed.
        indices.append(self.speed_start + (vx + 1))
        indices.append(self.speed_start + 3 + (vy + 1))

        # Part 3: what the radar sees (room 5 only).
        if self.radar_part > 0:
            for ray in range(self.room.n_rays):
                distance = observation[4 + ray]
                share = distance / self.room.lookahead_m
                bin_index = int(share * self.ray_bins)
                bin_index = min(bin_index, self.ray_bins - 1)
                indices.append(self.radar_start + ray * self.ray_bins + bin_index)

        return np.array(indices, dtype=int)


def new_weights(features):
    """One weight per feature, for each of the 9 actions. We start from zero."""
    return np.zeros((len(ACTIONS), features.size))


def q_all_actions(weights, active):
    """The estimated value of every action in this situation."""
    return weights[:, active].sum(axis=1)


def q_value(weights, active, action):
    """The estimated value of one action."""
    return weights[action][active].sum()


def greedy_action(weights, active):
    """The action the agent currently believes is best."""
    values = q_all_actions(weights, active)
    return int(np.argmax(values))


def epsilon_greedy(weights, active, epsilon):
    """Mostly the best action, sometimes a random one."""
    if random.random() < epsilon:
        return random.randrange(len(ACTIONS))
    return greedy_action(weights, active)


def update(weights, active, action, target, alpha):
    """
    Move the weights of the active features a little towards the target.

    We divide alpha by the number of active features, so that the total change
    to q(state, action) is about the size we asked for - not many times bigger.
    """
    prediction = q_value(weights, active, action)
    error = target - prediction
    weights[action][active] += (alpha / len(active)) * error
    return error


def train_sarsa(room,
                features,
                episodes=400,
                alpha=0.2,
                gamma=0.99,
                epsilon_start=1.0,
                epsilon_end=0.05,
                epsilon_decay=0.99,
                snapshot_count=10,
                progress=None):
    """
    Room 4: semi-gradient SARSA with the linear model.
    Same idea as the SARSA of room 2, but a table update becomes a weight update.
    """
    weights = new_weights(features)
    log = EpisodeLog()
    snapshots = [(0, weights.copy())]
    snapshot_every = max(1, episodes // snapshot_count)
    epsilon = epsilon_start
    max_steps = room.max_decisions

    for episode in range(1, episodes + 1):
        observation = room.reset()
        active = features.active(observation)
        action = epsilon_greedy(weights, active, epsilon)
        log.start()
        steps = 0
        done = False

        while not done and steps < max_steps:
            frame = room.frame()
            observation, reward, done, _ = room.step(action)
            next_active = features.active(observation)
            next_action = epsilon_greedy(weights, next_active, epsilon)

            # ---- SARSA: use the action we are actually going to take ----
            future = 0.0 if done else q_value(weights, next_active, next_action)
            target = reward + gamma * future
            update(weights, active, action, target, alpha)

            log.add(frame, action, reward)
            active = next_active
            action = next_action
            steps += 1

        log.finish(episode, epsilon, room.escaped())

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if episode % snapshot_every == 0 or episode == episodes:
            snapshots.append((episode, weights.copy()))

        if progress is not None:
            progress(episode, episodes)

    return {"weights": weights, "log": log.as_dict(), "kept": log.kept,
            "snapshots": snapshots}


def train_qlearning(room,
                    features,
                    episodes=600,
                    alpha=0.2,
                    gamma=0.99,
                    epsilon_start=1.0,
                    epsilon_end=0.05,
                    epsilon_decay=0.995,
                    snapshot_count=10,
                    progress=None):
    """
    Room 5: semi-gradient Q-Learning with the linear model.
    The only difference from train_sarsa is the marked line below.
    """
    weights = new_weights(features)
    log = EpisodeLog()
    snapshots = [(0, weights.copy())]
    snapshot_every = max(1, episodes // snapshot_count)
    epsilon = epsilon_start
    max_steps = room.max_decisions

    for episode in range(1, episodes + 1):
        observation = room.reset()
        active = features.active(observation)
        log.start()
        steps = 0
        done = False

        while not done and steps < max_steps:
            action = epsilon_greedy(weights, active, epsilon)
            frame = room.frame()
            observation, reward, done, _ = room.step(action)
            next_active = features.active(observation)

            # ---- Q-Learning: use the BEST next action, whatever we do next ----
            best_next = q_all_actions(weights, next_active).max()
            future = 0.0 if done else best_next
            target = reward + gamma * future
            update(weights, active, action, target, alpha)

            log.add(frame, action, reward)
            active = next_active
            steps += 1

        log.finish(episode, epsilon, room.escaped())

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if episode % snapshot_every == 0 or episode == episodes:
            snapshots.append((episode, weights.copy()))

        if progress is not None:
            progress(episode, episodes)

    return {"weights": weights, "log": log.as_dict(), "kept": log.kept,
            "snapshots": snapshots}
