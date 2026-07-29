"""
Room 3: Q-Learning (the model of the room is NOT known).

This file is deliberately almost identical to sarsa_agent.py, because the two
algorithms really are almost identical. Read them side by side - the difference
is ONE line:

    SARSA       target = reward + gamma * q[next_state][next_action]
                                          the action we will actually take

    Q-Learning  target = reward + gamma * max(q[next_state])
                                          the BEST action, even if we do not take it

Because Q-Learning always assumes the best possible continuation, it learns the
shortest route ("off-policy" learning) even while it is still exploring
randomly. In a room with traps that makes it braver - and less careful - than
SARSA.

Room 3 is also harder than rooms 1 and 2 for a second reason: the door stays dead
until three jobs have been done IN ORDER - fetch the power cell, throw breaker A,
throw breaker B - so a state is (row, column, stage) with stage counting 0 to 3.
That is a 10 x 10 x 4 table instead of 10 x 10 x 1: the same table, one dimension
bigger, holding four different plans at once.
"""

import random

import numpy as np

from rooms.grid_world import ACTIONS
from training.metrics import EpisodeLog


def epsilon_greedy(q, state, epsilon):
    """Same helper as in sarsa_agent.py: mostly greedy, sometimes random."""
    if random.random() < epsilon:
        return random.randrange(len(ACTIONS))
    return int(np.argmax(q[state]))


def train(env,
          episodes=1200,
          alpha=0.1,
          gamma=0.95,
          epsilon_start=1.0,
          epsilon_end=0.05,
          epsilon_decay=0.997,
          max_steps=400,
          snapshot_count=10,
          progress=None):
    """Play the room `episodes` times and learn a q table from experience."""
    q = np.zeros(env.q_table_shape())

    # How often each tile was stood on, for the exploration heat map.
    visits = np.zeros((env.n_rows, env.n_cols))
    log = EpisodeLog()

    snapshots = [(0, q.copy())]
    snapshot_every = max(1, episodes // snapshot_count)

    epsilon = epsilon_start

    for episode in range(1, episodes + 1):
        state = env.reset()
        log.start()
        steps = 0
        done = False

        while not done and steps < max_steps:
            action = epsilon_greedy(q, state, epsilon)
            next_state, reward, done, _ = env.step(action)

            # ---- the Q-Learning update ----
            best_next = np.max(q[next_state])
            future = 0.0 if done else best_next
            target = reward + gamma * future
            old_value = q[state][action]
            q[state][action] = old_value + alpha * (target - old_value)

            log.add(state, action, reward)
            visits[state[0]][state[1]] += 1
            state = next_state
            steps += 1

        log.finish(episode, epsilon, env.escaped(state))

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if episode % snapshot_every == 0 or episode == episodes:
            snapshots.append((episode, q.copy()))

        if progress is not None:
            progress(episode, episodes)

    return {"q": q, "log": log.as_dict(), "kept": log.kept,
            "snapshots": snapshots, "visits": visits}
