"""Training loop for Room 2 (SARSA).

Runs the agent against Room 2's environment, logs per-episode metrics
(return, steps, epsilon, success) for the learning-curve plots, and saves
full trajectories of a few representative episodes (first / middle / last)
so the app can replay them afterward.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from agents.sarsa_agent import SARSAAgent, SARSAConfig
from environments.room2_sarsa_env import make_room2_env


@dataclass
class TrainRoom2Config:
    episodes: int = 3000
    max_steps: int = 200
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    slip_prob: float = 0.2
    seed: Optional[int] = 0
    results_dir: str = "results/room2"


def _episodes_to_record(n_episodes: int) -> set[int]:
    if n_episodes <= 0:
        return set()
    picks = {0, n_episodes // 2, n_episodes - 1}
    return {p for p in picks if 0 <= p < n_episodes}


def train(cfg: TrainRoom2Config):
    env = make_room2_env(slip_prob=cfg.slip_prob, max_steps=cfg.max_steps, seed=cfg.seed)
    agent = SARSAAgent(
        SARSAConfig(
            n_states=env.n_states,
            n_actions=env.n_actions,
            alpha=cfg.alpha,
            gamma=cfg.gamma,
            epsilon_start=cfg.epsilon_start,
            epsilon_min=cfg.epsilon_min,
            epsilon_decay=cfg.epsilon_decay,
            seed=cfg.seed,
        )
    )

    record_episodes = _episodes_to_record(cfg.episodes)
    history = []
    trajectories = {}

    for ep in range(cfg.episodes):
        state, _ = env.reset()
        action = agent.select_action(state)
        record = ep in record_episodes
        traj = [{"state": state, "action": None, "reward": None}] if record else None

        total_reward = 0.0
        steps = 0
        terminated = truncated = False
        last_reward = 0.0

        while not (terminated or truncated):
            next_state, reward, terminated, truncated, info = env.step(action)
            next_action = agent.select_action(next_state)
            agent.update(state, action, reward, next_state, next_action, terminated)

            if record:
                traj[-1]["action"] = action
                traj[-1]["reward"] = reward
                traj.append({"state": next_state, "action": None, "reward": None})

            state, action = next_state, next_action
            total_reward += reward
            last_reward = reward
            steps += 1

        if record:
            trajectories[ep] = traj

        agent.decay_epsilon()
        history.append(
            {
                "episode": ep,
                "return": total_reward,
                "steps": steps,
                "epsilon": agent.epsilon,
                "success": bool(terminated and last_reward > 0),
            }
        )

    _save_results(cfg, agent, history, trajectories)
    return agent, history, trajectories


def _save_results(cfg: TrainRoom2Config, agent: SARSAAgent, history: list, trajectories: dict) -> None:
    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "q_table.npy", agent.q)
    (out_dir / "history.json").write_text(json.dumps(history))
    (out_dir / "trajectories.json").write_text(json.dumps(trajectories))
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg)))


if __name__ == "__main__":
    train(TrainRoom2Config())
