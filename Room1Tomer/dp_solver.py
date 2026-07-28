"""
Dynamic Programming solvers for Room 1: Value Iteration and Policy Iteration.

Both operate directly on the known model P[s][a] = [(prob, next_s, reward, done), ...]
returned by Room1Env.get_transition_model().
"""

import time
import numpy as np


def q_value(P, V, s, a, gamma):
    return sum(
        prob * (reward + gamma * V[next_s] * (0.0 if done else 1.0))
        for prob, next_s, reward, done in P[s][a]
    )


def extract_policy(P, V, n_states, n_actions, gamma):
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        q_values = [q_value(P, V, s, a, gamma) for a in range(n_actions)]
        policy[s] = int(np.argmax(q_values))
    return policy


def value_iteration(P, n_states, n_actions, gamma=0.95, theta=1e-6, max_iterations=10000):
    V = np.zeros(n_states)
    start = time.time()
    it = 0
    for it in range(1, max_iterations + 1):
        delta = 0.0
        for s in range(n_states):
            q_values = [q_value(P, V, s, a, gamma) for a in range(n_actions)]
            best = max(q_values)
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        if delta < theta:
            break
    elapsed = time.time() - start

    policy = extract_policy(P, V, n_states, n_actions, gamma)
    return V, policy, {"algorithm": "Value Iteration", "iterations": it, "time_sec": elapsed}


def policy_evaluation(P, policy, n_states, n_actions, gamma, theta=1e-6, max_iterations=10000):
    V = np.zeros(n_states)
    for _ in range(max_iterations):
        delta = 0.0
        for s in range(n_states):
            a = int(policy[s])
            v = q_value(P, V, s, a, gamma)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V


def policy_iteration(P, n_states, n_actions, gamma=0.95, theta=1e-6, max_iterations=1000, seed=0):
    rng = np.random.default_rng(seed)
    policy = rng.integers(0, n_actions, size=n_states)
    start = time.time()
    it = 0
    V = np.zeros(n_states)
    for it in range(1, max_iterations + 1):
        V = policy_evaluation(P, policy, n_states, n_actions, gamma, theta)
        new_policy = extract_policy(P, V, n_states, n_actions, gamma)
        if np.array_equal(new_policy, policy):
            policy = new_policy
            break
        policy = new_policy
    elapsed = time.time() - start

    return V, policy, {"algorithm": "Policy Iteration", "iterations": it, "time_sec": elapsed}
