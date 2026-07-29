"""Hyperparameter sweep for Room 2's SARSA agent.

Sweeps alpha / gamma / epsilon_decay -- the agent's tunable hyperparameters.
slip_prob and the reward shape are the room's fixed design (not swept here;
changing them would change the room, not tune the agent).

For each combination, trains agents over a couple of seeds, then evaluates
the learned policy *greedily* (epsilon=0, on fresh episodes) so final policy
quality is judged independent of training-time exploration noise. Since the
bridge is itself a slippery cell -- not a one-time-safe crossing -- every
episode's one mandatory bridge-exit move carries an irreducible slip_prob
chance of failure, capping the achievable success rate at roughly
1 - slip_prob (~80% at the default slip_prob=0.2), not 100%. So we measure,
from the training history, the first episode at which a 50-episode rolling
success rate reaches CONVERGENCE_THRESHOLD -- set safely below that ceiling
(accounting for epsilon_min residual exploration noise during training),
rather than the unreachable 0.9 that would make sense for a room without an
irreducible per-episode risk.

Results are written to sweep_results.csv, best config first (ranked by
greedy success rate, then convergence speed, then steps-to-goal).
"""
from __future__ import annotations

import csv
import itertools
import math
import statistics
import tempfile

from room2_env import make_room2_env
from train_room2 import TrainRoom2Config, train

ALPHAS = (0.05, 0.1, 0.2, 0.3)
GAMMAS = (0.9, 0.95, 0.99)
EPSILON_DECAYS = (0.99, 0.995, 0.999)
TRAIN_SEEDS = (0, 1)
TRAIN_EPISODES = 2000
EVAL_EPISODES = 200
CONVERGENCE_WINDOW = 50
CONVERGENCE_THRESHOLD = 0.65


def episodes_to_convergence(history: list[dict]) -> float:
    """First episode index where a rolling window of successes reaches the
    threshold; TRAIN_EPISODES (i.e. "never converged in time") if it doesn't.
    """
    successes = [h["success"] for h in history]
    n = len(successes)
    if n < CONVERGENCE_WINDOW:
        return float(TRAIN_EPISODES)
    running = sum(successes[:CONVERGENCE_WINDOW])
    if running / CONVERGENCE_WINDOW >= CONVERGENCE_THRESHOLD:
        return float(CONVERGENCE_WINDOW - 1)
    for i in range(CONVERGENCE_WINDOW, n):
        running += successes[i] - successes[i - CONVERGENCE_WINDOW]
        if running / CONVERGENCE_WINDOW >= CONVERGENCE_THRESHOLD:
            return float(i)
    return float(TRAIN_EPISODES)


def evaluate_greedy(agent, slip_prob: float, max_steps: int, seed: int) -> tuple[float, float]:
    successes = 0
    steps_of_successes = []
    for i in range(EVAL_EPISODES):
        env = make_room2_env(slip_prob=slip_prob, max_steps=max_steps, seed=seed + i)
        state, _ = env.reset()
        terminated = truncated = False
        steps = 0
        last_reward = 0.0
        while not (terminated or truncated):
            action = agent.greedy_action(state)
            state, last_reward, terminated, truncated, _ = env.step(action)
            steps += 1
        if terminated and last_reward > 0:
            successes += 1
            steps_of_successes.append(steps)
    success_rate = successes / EVAL_EPISODES
    avg_steps = statistics.fmean(steps_of_successes) if steps_of_successes else math.nan
    return success_rate, avg_steps


def run_sweep():
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for alpha, gamma, epsilon_decay in itertools.product(ALPHAS, GAMMAS, EPSILON_DECAYS):
            success_rates = []
            avg_steps_list = []
            convergence_list = []
            for seed in TRAIN_SEEDS:
                cfg = TrainRoom2Config(
                    episodes=TRAIN_EPISODES,
                    alpha=alpha,
                    gamma=gamma,
                    epsilon_decay=epsilon_decay,
                    seed=seed,
                    results_dir=tmp,
                )
                agent, history, _ = train(cfg)
                sr, steps = evaluate_greedy(
                    agent, slip_prob=cfg.slip_prob, max_steps=cfg.max_steps, seed=10_000 + seed
                )
                success_rates.append(sr)
                convergence_list.append(episodes_to_convergence(history))
                if not math.isnan(steps):
                    avg_steps_list.append(steps)

            mean_sr = statistics.fmean(success_rates)
            mean_steps = statistics.fmean(avg_steps_list) if avg_steps_list else math.nan
            mean_convergence = statistics.fmean(convergence_list)
            row = {
                "alpha": alpha,
                "gamma": gamma,
                "epsilon_decay": epsilon_decay,
                "success_rate": mean_sr,
                "avg_steps": mean_steps,
                "episodes_to_converge": mean_convergence,
            }
            results.append(row)
            print(
                f"alpha={alpha:<4} gamma={gamma:<4} eps_decay={epsilon_decay:<5} "
                f"-> success={mean_sr:6.1%}  avg_steps={mean_steps:6.1f}  "
                f"converge_at={mean_convergence:7.1f}"
            )

    results.sort(
        key=lambda r: (
            -r["success_rate"],
            r["episodes_to_converge"],
            r["avg_steps"] if not math.isnan(r["avg_steps"]) else 1e9,
        )
    )

    with open("sweep_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["alpha", "gamma", "epsilon_decay", "success_rate", "avg_steps", "episodes_to_converge"],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nTop 5 configs (by success rate, then fewer steps):")
    for row in results[:5]:
        print(row)

    return results


if __name__ == "__main__":
    run_sweep()
