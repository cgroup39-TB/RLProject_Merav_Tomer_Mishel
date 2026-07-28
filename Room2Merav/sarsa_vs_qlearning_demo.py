"""Side-by-side SARSA vs. Q-Learning demo on Room 2's environment.

This room's "core idea" (per the assignment spec) is to show that SARSA
picks a safer route than an off-policy learner would. This script actually
checks that claim empirically rather than asserting it, since Room 2's
risk source is different from the textbook Cliff Walking setup: here, the
danger (slipping into the abyss) is a property of the *environment's*
transition function ('~' tiles), not something that only exists because
of the agent's own epsilon-greedy exploration. That distinction matters --
see the README's "SARSA vs Q-Learning: does it actually pick a safer
route?" section for what this script actually found, since it is not
assumed in advance.

Not a Room 3 submission -- Q-Learning here is a minimal, self-contained
comparison agent for this room's own analysis, not the room's official
implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from room2_env import make_room2_env


@dataclass
class QLearningConfig:
    n_states: int
    n_actions: int
    alpha: float = 0.3
    gamma: float = 0.9
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.99
    seed: Optional[int] = None


class QLearningAgent:
    """Off-policy TD control: bootstraps off the greedy next action,
    regardless of what the behavior policy actually takes next."""

    def __init__(self, config: QLearningConfig):
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

    def update(self, state: int, action: int, reward: float, next_state: int, terminated: bool) -> None:
        target = reward
        if not terminated:
            target += self.cfg.gamma * self.q[next_state].max()
        self.q[state, action] += self.cfg.alpha * (target - self.q[state, action])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)


def train_qlearning(episodes: int, slip_prob: float, max_steps: int, seed: int, **agent_kwargs):
    env = make_room2_env(slip_prob=slip_prob, max_steps=max_steps, seed=seed)
    agent = QLearningAgent(QLearningConfig(n_states=env.n_states, n_actions=env.n_actions, seed=seed, **agent_kwargs))
    history = []
    for ep in range(episodes):
        state, _ = env.reset()
        terminated = truncated = False
        total_reward = 0.0
        steps = 0
        last_reward = 0.0
        while not (terminated or truncated):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.update(state, action, reward, next_state, terminated)
            state = next_state
            total_reward += reward
            last_reward = reward
            steps += 1
        agent.decay_epsilon()
        history.append({"episode": ep, "return": total_reward, "steps": steps, "success": bool(terminated and last_reward > 0)})
    return agent, history


def rollout_greedy(agent, slip_prob: float, max_steps: int, seed: int):
    """One greedy rollout; returns (success, steps, row6_lateral_moves, positions)."""
    env = make_room2_env(slip_prob=slip_prob, max_steps=max_steps, seed=seed)
    state, _ = env.reset()
    positions = [env.to_coords(state)]
    terminated = truncated = False
    lateral_in_row6 = 0
    last_reward = 0.0
    while not (terminated or truncated):
        action = agent.greedy_action(state)
        # action 1=RIGHT, 3=LEFT are horizontal
        if env.state[0] == 6 and action in (1, 3):
            lateral_in_row6 += 1
        state, last_reward, terminated, truncated, _ = env.step(action)
        positions.append(env.to_coords(state))
    success = bool(terminated and last_reward > 0)
    return success, len(positions) - 1, lateral_in_row6, positions


def train_sarsa(episodes: int, slip_prob: float, max_steps: int, seed: int, **agent_kwargs):
    from sarsa_agent import SARSAAgent, SARSAConfig

    env = make_room2_env(slip_prob=slip_prob, max_steps=max_steps, seed=seed)
    agent = SARSAAgent(SARSAConfig(n_states=env.n_states, n_actions=env.n_actions, seed=seed, **agent_kwargs))
    history = []
    for ep in range(episodes):
        state, _ = env.reset()
        action = agent.select_action(state)
        terminated = truncated = False
        total_reward = 0.0
        steps = 0
        last_reward = 0.0
        while not (terminated or truncated):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = agent.select_action(next_state)
            agent.update(state, action, reward, next_state, next_action, terminated)
            state, action = next_state, next_action
            total_reward += reward
            last_reward = reward
            steps += 1
        agent.decay_epsilon()
        history.append({"episode": ep, "return": total_reward, "steps": steps, "success": bool(terminated and last_reward > 0)})
    return agent, history


def _rolling_mean(values, window):
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes, slip_prob, max_steps, seed = 3000, 0.2, 200, 0

    sarsa, sarsa_history = train_sarsa(episodes, slip_prob, max_steps, seed)
    qlearn, q_history = train_qlearning(episodes, slip_prob, max_steps, seed)

    n_eval = 200
    print(f"{'Agent':<12}{'success%':>10}{'avg steps':>12}{'avg row6 lateral moves':>26}")
    for name, agent in [("SARSA", sarsa), ("Q-Learning", qlearn)]:
        successes, steps_list, lateral_list = [], [], []
        for i in range(n_eval):
            success, steps, lateral, _ = rollout_greedy(agent, slip_prob, max_steps, seed=10_000 + i)
            successes.append(success)
            if success:
                steps_list.append(steps)
                lateral_list.append(lateral)
        succ_rate = sum(successes) / n_eval
        avg_steps = sum(steps_list) / len(steps_list) if steps_list else float("nan")
        avg_lateral = sum(lateral_list) / len(lateral_list) if lateral_list else float("nan")
        print(f"{name:<12}{succ_rate:>9.1%}{avg_steps:>12.1f}{avg_lateral:>26.2f}")

    sarsa_avg = sum(h["return"] for h in sarsa_history[-500:]) / 500
    q_avg = sum(h["return"] for h in q_history[-500:]) / 500
    print(f"\nAvg training-time return (last 500 episodes): SARSA={sarsa_avg:.2f}  Q-Learning={q_avg:.2f}")

    from viz import BG_COLOR, PANEL_COLOR, GRID_LINE_COLOR, TEXT_COLOR, _style_dark_axes

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG_COLOR)
    ax.plot(_rolling_mean([h["return"] for h in sarsa_history], 50), label="SARSA", color="#5fd0ff")
    ax.plot(_rolling_mean([h["return"] for h in q_history], 50), label="Q-Learning", color="#ff7a1a")
    ax.set_xlabel("episode")
    ax.set_ylabel("return (rolling mean, window=50)")
    ax.set_title("Training-time return: SARSA vs Q-Learning")
    ax.legend(facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR, edgecolor=GRID_LINE_COLOR)
    _style_dark_axes(ax)
    ax.grid(True, color=GRID_LINE_COLOR, linewidth=0.5)
    fig.tight_layout()
    fig.savefig("sarsa_vs_qlearning.png", dpi=110, facecolor=fig.get_facecolor())
    print("saved sarsa_vs_qlearning.png")


if __name__ == "__main__":
    main()
