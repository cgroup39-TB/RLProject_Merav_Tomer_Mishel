"""
Tabular Q-Learning (off-policy TD control) for Room 3.

Like sarsa_solver.py, the environment's model is not assumed known -- the
agent only interacts with env.reset()/env.step(). The difference from SARSA
is one line: the update bootstraps off the BEST next action
(max_a Q(s', a)), not the action the behavior policy actually takes next,
which is what makes Q-Learning off-policy -- it learns the greedy path even
while still exploring randomly.
"""

import numpy as np


def epsilon_greedy(Q, s, n_actions, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(n_actions))
    return int(np.argmax(Q[s]))


def q_learning(
    env,
    n_states,
    n_actions,
    episodes=2000,
    alpha=0.1,
    gamma=0.95,
    epsilon=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.995,
    max_steps=200,
    seed=0,
):
    rng = np.random.default_rng(seed)
    Q = np.zeros((n_states, n_actions))
    reward_history = []
    steps_history = []
    eps = epsilon

    for ep in range(episodes):
        s = env.reset()
        total_r = 0.0
        steps = 0
        for steps in range(1, max_steps + 1):
            a = epsilon_greedy(Q, s, n_actions, eps, rng)
            s_next, r, done, truncated, _ = env.step(a)

            best_next = np.max(Q[s_next])
            td_target = r + (0.0 if done else gamma * best_next)
            Q[s, a] += alpha * (td_target - Q[s, a])

            total_r += r
            s = s_next
            if done or truncated:
                break

        reward_history.append(total_r)
        steps_history.append(steps)
        eps = max(epsilon_min, eps * epsilon_decay)

    policy = np.argmax(Q, axis=1)
    info = {
        "algorithm": "Q-Learning",
        "episodes": episodes,
        "final_epsilon": eps,
        "reward_history": reward_history,
        "steps_history": steps_history,
    }
    return Q, policy, info
