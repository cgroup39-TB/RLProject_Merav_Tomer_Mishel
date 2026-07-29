"""On-policy TD control (SARSA) for the tabular grid rooms.

Update rule (Room 2 uses this):
    Q(s, a) += alpha * (r + gamma * Q(s', a') - Q(s, a))
where a' is the action the agent will *actually* take next under its
epsilon-greedy behavior policy -- not the greedy max -- which is what
makes this on-policy: it commits to the consequences of its own
exploration when updating its value estimates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SARSAConfig:
    n_states: int
    n_actions: int
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    seed: Optional[int] = None


class SARSAAgent:
    def __init__(self, config: SARSAConfig):
        self.cfg = config
        self.q = np.zeros((config.n_states, config.n_actions))
        self.epsilon = config.epsilon_start
        self.rng = np.random.default_rng(config.seed)

    def select_action(self, state: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.cfg.n_actions))
        return self.greedy_action(state)

    def greedy_action(self, state: int) -> int:
        return int(np.argmax(self.q[state]))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: int,
        terminated: bool,
    ) -> None:
        target = reward
        if not terminated:
            target += self.cfg.gamma * self.q[next_state, next_action]
        self.q[state, action] += self.cfg.alpha * (target - self.q[state, action])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)
