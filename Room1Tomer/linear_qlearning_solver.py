"""
Room 5 -- semi-gradient Q-Learning with a linear model (function approximation).

Rooms 1-3 could keep one number per state because there were only a few
hundred of them. Room 5's state is continuous (x, y, vx, vy, plus a radar
reading per ray), so there is no table to fill -- instead the value of an
action is a weighted sum of features:

    q(state, action) = weights[action] . features(state)

and learning means nudging the weights, not filling in table rows. All
features here are 0/1 indicators, and only a handful are ever 1 at once, so
rather than storing a long mostly-zero vector, `Features.active()` returns
just the list of indices that ARE 1 -- summing a few weights is fast.

    Part 1 -- tile coding of position (x, y): lay a coarse grid over the
    room, mark the tile the agent is standing in, then repeat with a few
    more grids shifted slightly. Overlapping tilings give smooth, cheap
    generalisation between nearby positions -- the textbook technique.

    Part 2 -- speed: vx and vy only take 3 values each, so just mark which.

    Part 3 -- radar: each ray's distance goes into one of a few bins (very
    close / close / far / clear), one bin marked per ray. Every action has
    its own weights, so the agent can learn rules like "radar-right very
    close -> going right is bad" -- a rule that means the same thing in
    every obstacle layout, which is exactly what lets it transfer to one
    it's never seen.

The update rule is semi-gradient Q-Learning -- same "bootstrap off the best
next action" idea as qlearning_solver.py's tabular version, just applied to
weights instead of a Q-table:

    target = reward + gamma * max_a' q(next_state, a')
    weights[action][active_features] += (alpha / n_active) * (target - q(state, action))
"""

import numpy as np


class Features:
    """Turns an observation (x, y, vx, vy, radar...) into active feature indices."""

    def __init__(self, arena_size, n_rays, lookahead_m, n_tilings=4, tiles=10, ray_bins=4):
        self.n_tilings = n_tilings
        self.tiles = tiles
        self.n_rays = n_rays
        self.lookahead_m = lookahead_m
        self.ray_bins = ray_bins
        self.tile_width = arena_size / tiles

        self.position_part = n_tilings * tiles * tiles
        self.speed_part = 3 + 3
        self.radar_part = n_rays * ray_bins

        self.speed_start = self.position_part
        self.radar_start = self.speed_start + self.speed_part
        self.size = self.radar_start + self.radar_part

    def active(self, observation):
        x, y, vx, vy = observation[0], observation[1], int(observation[2]), int(observation[3])
        indices = []

        for tiling in range(self.n_tilings):
            offset = self.tile_width * tiling / self.n_tilings
            column = min(int((x + offset) / self.tile_width), self.tiles - 1)
            row = min(int((y + offset) / self.tile_width), self.tiles - 1)
            indices.append(tiling * self.tiles * self.tiles + row * self.tiles + column)

        indices.append(self.speed_start + (vx + 1))
        indices.append(self.speed_start + 3 + (vy + 1))

        for ray in range(self.n_rays):
            distance = observation[4 + ray]
            share = distance / self.lookahead_m
            bin_index = min(int(share * self.ray_bins), self.ray_bins - 1)
            indices.append(self.radar_start + ray * self.ray_bins + bin_index)

        return np.array(indices, dtype=int)


def new_weights(n_actions, features):
    return np.zeros((n_actions, features.size))


def q_all_actions(weights, active):
    return weights[:, active].sum(axis=1)


def q_value(weights, active, action):
    return weights[action][active].sum()


def greedy_action(weights, active):
    return int(np.argmax(q_all_actions(weights, active)))


def epsilon_greedy(weights, active, n_actions, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(n_actions))
    return greedy_action(weights, active)


def update(weights, active, action, target, alpha):
    prediction = q_value(weights, active, action)
    error = target - prediction
    weights[action][active] += (alpha / len(active)) * error


def train_linear_qlearning(
    env,
    features,
    n_actions,
    episodes=700,
    alpha=0.3,
    gamma=0.995,
    epsilon=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.995,
    seed=0,
):
    rng = np.random.default_rng(seed)
    weights = new_weights(n_actions, features)
    reward_history = []
    steps_history = []
    eps = epsilon
    max_steps = env.max_decisions

    for _ in range(episodes):
        observation = env.reset()
        active = features.active(observation)
        total_r = 0.0
        steps = 0
        done = False

        while not done and steps < max_steps:
            action = epsilon_greedy(weights, active, n_actions, eps, rng)
            observation, reward, done, _ = env.step(action)
            next_active = features.active(observation)

            best_next = q_all_actions(weights, next_active).max()
            target = reward + (0.0 if done else gamma * best_next)
            update(weights, active, action, target, alpha)

            active = next_active
            total_r += reward
            steps += 1

        reward_history.append(total_r)
        steps_history.append(steps)
        eps = max(epsilon_min, eps * epsilon_decay)

    info = {
        "algorithm": "Semi-gradient Q-Learning (linear)",
        "episodes": episodes,
        "final_epsilon": eps,
        "reward_history": reward_history,
        "steps_history": steps_history,
    }
    return weights, info
