"""
Hand-written NumPy DQN (no PyTorch/TensorFlow) for Room 4.

A small 2-hidden-layer MLP approximates Q(s,a) over the 4-dimensional
continuous drone state. Forward pass, backprop, and the Adam optimizer are
all implemented by hand (matching the pure-NumPy style of dp_solver.py /
sarsa_solver.py), plus the standard DQN stabilizers: experience replay and a
periodically-synced target network.
"""

import time

import numpy as np

from drone_env import N_ACTIONS, HOVER_ACTION


def _he_init(rng, fan_in, fan_out):
    return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))


def _xavier_init(rng, fan_in, fan_out):
    return rng.normal(0.0, np.sqrt(1.0 / fan_in), size=(fan_in, fan_out))


class QNetwork:
    """Input(4) -> Dense(h1) -> ReLU -> Dense(h2) -> ReLU -> Dense(n_actions, linear)."""

    def __init__(self, input_dim=4, hidden=(64, 64), n_actions=N_ACTIONS, lr=1e-3,
                 seed=None, beta1=0.9, beta2=0.999, eps=1e-8):
        rng = np.random.default_rng(seed)
        h1, h2 = hidden
        self.params = {
            "W1": _he_init(rng, input_dim, h1), "b1": np.zeros(h1),
            "W2": _he_init(rng, h1, h2), "b2": np.zeros(h2),
            "W3": _xavier_init(rng, h2, n_actions), "b3": np.zeros(n_actions),
        }
        self.lr = lr
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.adam_t = 0
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.input_dim = input_dim
        self.n_actions = n_actions

    def forward(self, X, cache=False):
        p = self.params
        z1 = X @ p["W1"] + p["b1"]
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ p["W2"] + p["b2"]
        a2 = np.maximum(z2, 0.0)
        q = a2 @ p["W3"] + p["b3"]
        if cache:
            return q, (X, z1, a1, z2, a2)
        return q

    def loss_and_grads(self, X, actions, targets):
        """Masked-MSE TD loss (gradient flows only through each sample's
        taken action). Returns (loss, grads) without mutating params --
        kept separate from the optimizer step so it can be checked directly
        against finite differences."""
        p = self.params
        B = X.shape[0]
        q, (X, z1, a1, z2, a2) = self.forward(X, cache=True)
        rows = np.arange(B)

        dQ = np.zeros_like(q)
        dQ[rows, actions] = 2.0 * (q[rows, actions] - targets) / B

        grads = {}
        grads["W3"] = a2.T @ dQ
        grads["b3"] = dQ.sum(axis=0)
        da2 = dQ @ p["W3"].T
        dz2 = da2 * (z2 > 0)
        grads["W2"] = a1.T @ dz2
        grads["b2"] = dz2.sum(axis=0)
        da1 = dz2 @ p["W2"].T
        dz1 = da1 * (z1 > 0)
        grads["W1"] = X.T @ dz1
        grads["b1"] = dz1.sum(axis=0)

        loss = float(np.mean((q[rows, actions] - targets) ** 2))
        return loss, grads

    def apply_grads(self, grads):
        self.adam_t += 1
        p = self.params
        for k in p:
            g = grads[k]
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[k] / (1 - self.beta1 ** self.adam_t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.adam_t)
            p[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def train_step(self, X, actions, targets):
        """One Adam update from a masked-MSE TD loss. Returns the scalar loss."""
        loss, grads = self.loss_and_grads(X, actions, targets)
        self.apply_grads(grads)
        return loss

    def get_weights(self):
        return {k: v.copy() for k, v in self.params.items()}

    def set_weights(self, weights):
        for k, v in weights.items():
            self.params[k] = v.copy()


class ReplayBuffer:
    def __init__(self, capacity, state_dim=4, seed=None):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim))
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity)
        self.next_states = np.zeros((capacity, state_dim))
        self.dones = np.zeros(capacity)
        self.idx = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def push(self, s, a, r, s_next, done):
        i = self.idx
        self.states[i] = s
        self.actions[i] = a
        self.rewards[i] = r
        self.next_states[i] = s_next
        self.dones[i] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self):
        return self.size

    def sample(self, batch_size):
        idxs = self.rng.integers(0, self.size, size=batch_size)
        return (self.states[idxs], self.actions[idxs], self.rewards[idxs],
                self.next_states[idxs], self.dones[idxs])


def normalize_state(states, room_w, room_h, max_abs_speed):
    """states: (B, 4) array -> normalized (B, 4) array in roughly [-1, 1]."""
    x = states[:, 0] / room_w * 2 - 1
    y = states[:, 1] / room_h * 2 - 1
    vx = states[:, 2] / max_abs_speed
    vy = states[:, 3] / max_abs_speed
    return np.stack([x, y, vx, vy], axis=1)


def epsilon_greedy(network, state, room_w, room_h, max_abs_speed, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))
    x = normalize_state(state[None, :], room_w, room_h, max_abs_speed)
    q = network.forward(x)
    return int(np.argmax(q[0]))


def train_dqn(
    env,
    episodes=1200,
    max_steps=350,
    lr=1e-3,
    hidden=(32, 32),
    buffer_capacity=30000,
    batch_size=64,
    train_freq=2,
    target_sync_freq=200,
    gamma=0.95,
    epsilon=1.0,
    epsilon_min=0.15,
    epsilon_decay=0.997,
    seed=0,
    progress_callback=None,
):
    rng = np.random.default_rng(seed)
    online = QNetwork(input_dim=4, hidden=hidden, n_actions=env.n_actions, lr=lr, seed=seed)
    target = QNetwork(input_dim=4, hidden=hidden, n_actions=env.n_actions, lr=lr, seed=seed)
    target.set_weights(online.get_weights())
    buffer = ReplayBuffer(buffer_capacity, state_dim=4, seed=seed)

    room_w, room_h, max_abs_speed = env.cfg.room_w, env.cfg.room_h, env.max_abs_speed

    reward_history, steps_history, loss_history = [], [], []
    total_steps = 0
    eps = epsilon
    start = time.time()

    for ep in range(episodes):
        s = env.reset()
        ep_reward = 0.0
        steps = 0
        for steps in range(1, max_steps + 1):
            a = epsilon_greedy(online, s, room_w, room_h, max_abs_speed, eps, rng)
            s_next, r, done, truncated, info = env.step(a)
            # push true terminals only (`done`) -- truncation must still bootstrap
            buffer.push(s, a, r, s_next, done)
            total_steps += 1

            if len(buffer) >= batch_size and total_steps % train_freq == 0:
                S, A, R, S2, D = buffer.sample(batch_size)
                Xn = normalize_state(S, room_w, room_h, max_abs_speed)
                X2n = normalize_state(S2, room_w, room_h, max_abs_speed)
                # Double DQN: online network picks the best next action, target
                # network evaluates it -- decouples selection from evaluation
                # and avoids vanilla DQN's max-operator overestimation bias,
                # which otherwise tends to inflate every action's Q-value
                # roughly uniformly and blur the differences between them.
                best_next_actions = np.argmax(online.forward(X2n), axis=1)
                q_next = target.forward(X2n)[np.arange(len(S2)), best_next_actions]
                y = R + gamma * q_next * (1.0 - D)
                loss_history.append(online.train_step(Xn, A, y))

            if total_steps % target_sync_freq == 0:
                target.set_weights(online.get_weights())

            ep_reward += r
            s = s_next
            if done or truncated:
                break

        reward_history.append(ep_reward)
        steps_history.append(steps)
        eps = max(epsilon_min, eps * epsilon_decay)

        if progress_callback is not None:
            progress_callback(ep + 1, episodes, ep_reward)

    info = {
        "algorithm": "DQN",
        "episodes": episodes,
        "final_epsilon": eps,
        "reward_history": reward_history,
        "steps_history": steps_history,
        "loss_history": loss_history,
        "time_sec": time.time() - start,
    }
    return online, info


def run_episode(env, network=None, epsilon=0.0, seed=None, max_steps=None):
    """Record a full frame-by-frame trajectory. network=None (or epsilon=1.0)
    means pure random actions; this single function produces both the
    'before' (fresh/random) and 'after' (trained) replay data."""
    rng = np.random.default_rng(seed)
    room_w, room_h, max_abs_speed = env.cfg.room_w, env.cfg.room_h, env.max_abs_speed
    max_steps = max_steps or env.max_steps

    s = env.reset()
    frames = [dict(
        t=0.0, x=float(s[0]), y=float(s[1]), vx=float(s[2]), vy=float(s[3]),
        speed=float(np.hypot(s[2], s[3])), action=HOVER_ACTION,
        wind=(0.0, 0.0), reward=0.0, crashed=False, landed=False,
    )]
    total_reward = 0.0
    landed = crashed = False
    steps = 0

    for steps in range(1, max_steps + 1):
        if network is None or rng.random() < epsilon:
            a = int(rng.integers(N_ACTIONS))
        else:
            x = normalize_state(s[None, :], room_w, room_h, max_abs_speed)
            a = int(np.argmax(network.forward(x)[0]))

        s, r, done, truncated, info = env.step(a)
        total_reward += r
        frames.append(dict(
            t=info["t"], x=float(s[0]), y=float(s[1]), vx=float(s[2]), vy=float(s[3]),
            speed=info["speed"], action=a, wind=info["wind"],
            reward=float(r), crashed=info["crashed"], landed=info["landed"],
        ))
        if done:
            landed, crashed = info["landed"], info["crashed"]
            break
        if truncated:
            break

    return dict(frames=frames, solved=landed, crashed=crashed, total_reward=total_reward, steps=steps)
