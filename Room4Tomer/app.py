"""
Escape Room RL — Room 4: The Drone Room (Function Approximation / DQN).

Standalone Streamlit app (one folder per room per person, matching the
team's convention). Continuous state space (X, Y, Vx, Vy), solved with a
hand-written NumPy DQN. Shows the drone's flight path before and after
training.

Run locally with:
    streamlit run app.py
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from drone_env import DroneEnv, DroneConfig, Pad, Rect, WindZone
from dqn_solver import QNetwork, train_dqn, run_episode
from flight_canvas import render_flight_canvas

if not st.session_state.get("_embedded"):
    st.set_page_config(page_title="Escape Room RL — Room 4", page_icon="🚁", layout="wide")


# ---------------------------------------------------------------------------
# session state
# ---------------------------------------------------------------------------
def init_state():
    if "initialized" in st.session_state:
        return
    st.session_state.initialized = True
    st.session_state.results = {}


init_state()


# ---------------------------------------------------------------------------
# config building
# ---------------------------------------------------------------------------
def build_drone_config(params):
    walls = [Rect(w["x"], w["y"], w["w"], w["h"]) for w in params["walls"]]
    wind_zones = []
    if params["wind_scale"] > 0:
        # one fixed zone covering the middle of the room; strength scales
        # with the slider, so 0 == no wind at all
        wind_zones.append(WindZone(3.0, 3.0, 4.0, 4.0, fx=0.0, fy=1.5 * params["wind_scale"],
                                    gust_std=0.4 * params["wind_scale"]))
    return DroneConfig(
        start=(1.0, 1.0),
        pad=Pad(7.5, 7.5, params["pad_radius"]),
        walls=walls,
        wind_zones=wind_zones,
    )


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.header("🚁 Flight Setup")

        st.subheader("Training run")
        episodes = st.number_input(
            "Episodes", min_value=20, max_value=3000, value=1200, step=20, key="episodes",
            help="How many full episodes (reset → fly until landed/crashed/timeout) to train the DQN for.",
        )
        max_steps = st.number_input(
            "Max steps / episode", min_value=50, max_value=1000, value=350, step=25, key="max_steps",
            help="Episode is cut off (truncated, no bonus) after this many physics steps (each dt=0.02s) "
                 "if the drone hasn't landed or crashed yet.",
        )
        seed = st.number_input(
            "Random seed", min_value=0, value=0, step=1, key="seed",
            help="Seeds the wind's random gusts, the network's weight initialization, and replay "
                 "buffer sampling — same seed + same settings reproduces the exact same run.",
        )

        st.divider()
        st.subheader("Room layout")
        st.caption("Room is 10×10m. (x,y) = a wall's bottom-left corner; (w,h) = its size. "
                   "Walls are drawn as red \"danger gates\" — colliding with one is a crash.")
        n_walls = st.number_input("Number of walls", min_value=0, max_value=6, value=0, step=1, key="n_walls")
        walls = []
        for i in range(int(n_walls)):
            with st.expander(f"🔴 Wall {i + 1}", expanded=True):
                c1, c2 = st.columns(2)
                x = c1.number_input("x", 0.0, 9.5, 4.0, step=0.5, key=f"wall_{i}_x")
                y = c2.number_input("y", 0.0, 9.5, 0.0, step=0.5, key=f"wall_{i}_y")
                c3, c4 = st.columns(2)
                w = c3.number_input("width", 0.2, 10.0, 1.0, step=0.2, key=f"wall_{i}_w")
                h = c4.number_input("height", 0.2, 10.0, 3.0, step=0.2, key=f"wall_{i}_h")
                walls.append({"x": x, "y": y, "w": w, "h": h})

        wind_scale = st.slider(
            "Wind strength ×", 0.0, 3.0, 0.0, key="wind_scale",
            help="0 = no wind. Above 0, applies one fixed wind zone covering the middle of the room.",
        )

        st.divider()
        st.subheader("Drone dynamics")
        max_cmd_speed = st.slider(
            "Max commanded speed (m/s)", 0.5, 4.0, 2.0, key="max_cmd_speed",
            help="The velocity the drone accelerates toward while an action is held — action (1,1) "
                 "targets (max_cmd_speed, max_cmd_speed) m/s. Actual speed can still differ from this "
                 "due to momentum (see 'Engine responsiveness') and wind.",
        )
        max_accel = st.slider(
            "Engine responsiveness (m/s²)", 0.5, 15.0, 10.0, key="max_accel",
            help="Rate limit on how fast actual velocity can change toward the commanded speed each "
                 "step: Δv is clamped to ±max_accel·dt. Higher = snappier drone, easier to brake in "
                 "time for a soft landing; lower = more momentum/'weight', harder to control precisely.",
        )
        drone_radius = st.slider(
            "Drone radius (m)", 0.05, 0.5, 0.15, key="drone_radius",
            help="Collision radius used for wall/room-boundary/pad checks — the drone is modeled "
                 "as a circle, not a single point.",
        )

        st.divider()
        st.subheader("Landing rules")
        landing_speed_threshold = st.slider(
            "Max safe landing speed (m/s)", 0.1, 2.0, 1.0, key="landing_speed_threshold",
            help="Reaching the pad at or below this speed = a successful soft landing. Above it = a "
                 "crash ('hard landing'), same as hitting a wall.",
        )
        pad_radius = st.slider(
            "Landing pad radius (m)", 0.2, 1.5, 0.9, key="pad_radius",
            help="How close (in meters, from the drone's center to the pad's center) counts as "
                 "'reached the pad'.",
        )

        st.divider()
        st.subheader("Reward shaping")
        step_reward = st.number_input(
            "Step penalty", value=-0.01, step=0.01, format="%.3f", key="step_reward",
            help="Fixed reward added every single physics step, regardless of anything else — a small "
                 "constant time cost that nudges the agent to finish faster rather than loiter.",
        )
        shaping_weight = st.slider(
            "Distance-shaping weight", 0.0, 6.0, 3.0, key="shaping_weight",
            help="Potential-based reward shaping: extra reward = weight × (how much closer to the pad "
                 "this step got you). Rewards *progress toward the goal on every step*, not just the "
                 "final landing — this is the 'compass' that gives DQN a learnable signal before it's "
                 "ever actually landed even once. 0 = disabled (sparse reward only).",
        )
        crash_penalty = st.number_input(
            "Crash penalty", value=-30.0, step=5.0, key="crash_penalty",
            help="Reward on any crash: hitting a wall, flying outside the 10×10m room, or reaching "
                 "the pad faster than the safe landing speed.",
        )
        landing_bonus_base = st.number_input(
            "Landing bonus (base)", value=100.0, step=10.0, key="landing_bonus_base",
            help="Reward on a successful soft landing: base + gentleness × (0 to 1 score for how far "
                 "under the speed limit you were). This is that 'base' — the guaranteed part.",
        )
        landing_bonus_gentleness = st.number_input(
            "Landing bonus (gentleness)", value=50.0, step=10.0, key="landing_bonus_gentleness",
            help="Extra reward on top of the base landing bonus, scaled by how much slower than the "
                 "speed limit the drone was landing (1.0 = came to a near-complete stop, 0.0 = landed "
                 "right at the limit). Rewards gentler landings, not just legal ones.",
        )

        st.divider()
        st.subheader("DQN hyperparameters")
        lr = st.select_slider(
            "Learning rate", options=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2], value=1e-3, key="lr",
            help="Adam optimizer step size for the Q-network's weight updates. Higher = learns faster "
                 "but can be unstable; lower = more stable but slower to converge.",
        )
        hidden = st.select_slider(
            "Hidden layer size (×2 layers)", options=[16, 32, 64, 128], value=32, key="hidden",
            help="Units in each of the network's 2 hidden layers: Input(4) → Dense(h) → ReLU → "
                 "Dense(h) → ReLU → Dense(9 Q-values). Bigger = more capacity to represent complex "
                 "behavior, but slower per training step.",
        )
        gamma = st.slider(
            "Discount γ", 0.90, 0.999, 0.95, key="gamma",
            help="Discount factor in the DQN target: y = r + γ · max_a Q(s', a) — the same γ from the "
                 "Bellman equation. Closer to 1 = the agent weighs a distant future reward (like the "
                 "eventual landing bonus, dozens of steps away) more heavily.",
        )
        buffer_capacity = st.number_input(
            "Replay buffer capacity", min_value=1000, max_value=200000, value=30000, step=1000, key="buffer_capacity",
            help="How many past (state, action, reward, next_state, done) transitions the replay "
                 "buffer holds before it starts overwriting the oldest ones.",
        )
        batch_size = st.select_slider(
            "Batch size", options=[16, 32, 64, 128, 256], value=64, key="batch_size",
            help="How many stored transitions are randomly sampled from the replay buffer for each "
                 "training update (gradient step).",
        )
        train_freq = st.slider(
            "Train every N steps", 1, 10, 2, key="train_freq",
            help="Run one training update every this many environment steps, instead of every single "
                 "one — cuts compute cost roughly N× with little effect on learning quality.",
        )
        target_sync_freq = st.number_input(
            "Target network sync (steps)", min_value=50, max_value=5000, value=200, step=50, key="target_sync_freq",
            help="DQN uses two networks: 'online' (being trained) and 'target' (used to compute the "
                 "y = r + γ·max_a Q(s',a) target). Every this many steps, the target network's weights "
                 "are copied from the online one. Keeping the target briefly frozen like this is what "
                 "stabilizes training — without it, the target would shift every single update.",
        )
        epsilon = st.slider(
            "Initial ε (exploration)", 0.0, 1.0, 1.0, key="epsilon",
            help="ε-greedy policy: with probability ε, take a random action instead of the network's "
                 "current best guess. This is ε at the very start of training (episode 0).",
        )
        epsilon_min = st.slider(
            "Min ε", 0.0, 0.5, 0.15, key="epsilon_min",
            help="The floor ε decays to and never drops below — keeps a little randomness alive even "
                 "late in training.",
        )
        epsilon_decay = st.slider(
            "ε decay per episode", 0.90, 0.9999, 0.997, key="epsilon_decay", format="%.4f",
            help="After every episode: ε ← max(ε_min, ε · decay). Closer to 1 = exploration phase "
                 "lasts longer; further from 1 = shifts to exploiting the learned policy sooner.",
        )

        st.divider()
        train_clicked = st.button("▶ Train Agent", type="primary", use_container_width=True)

        params = dict(
            episodes=int(episodes), max_steps=int(max_steps), seed=int(seed),
            walls=walls, wind_scale=wind_scale,
            max_cmd_speed=max_cmd_speed, max_accel=max_accel, drone_radius=drone_radius,
            landing_speed_threshold=landing_speed_threshold, pad_radius=pad_radius,
            step_reward=step_reward, shaping_weight=shaping_weight, crash_penalty=crash_penalty,
            landing_bonus_base=landing_bonus_base, landing_bonus_gentleness=landing_bonus_gentleness,
            lr=lr, hidden=int(hidden), gamma=gamma, buffer_capacity=int(buffer_capacity),
            batch_size=int(batch_size), train_freq=int(train_freq), target_sync_freq=int(target_sync_freq),
            epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay,
        )
        return params, train_clicked


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train_room4(params):
    cfg = build_drone_config(params)
    env = DroneEnv(
        cfg,
        max_cmd_speed=params["max_cmd_speed"], max_accel=params["max_accel"],
        drone_radius=params["drone_radius"], landing_speed_threshold=params["landing_speed_threshold"],
        step_reward=params["step_reward"], shaping_weight=params["shaping_weight"],
        crash_penalty=params["crash_penalty"], landing_bonus_base=params["landing_bonus_base"],
        landing_bonus_gentleness=params["landing_bonus_gentleness"],
        max_steps=params["max_steps"], seed=params["seed"],
    )

    fresh_net = QNetwork(input_dim=4, hidden=(params["hidden"], params["hidden"]),
                          lr=params["lr"], seed=params["seed"])
    before = run_episode(env, fresh_net, epsilon=0.0, seed=params["seed"])

    total_episodes = params["episodes"]
    progress_bar = st.progress(0.0, text=f"Training... episode 0 / {total_episodes}")
    update_every = max(1, total_episodes // 100)  # cap ~100 UI updates regardless of episode count

    def on_progress(ep, total, ep_reward):
        if ep % update_every == 0 or ep == total:
            progress_bar.progress(ep / total, text=f"Training... episode {ep} / {total}  (last reward: {ep_reward:.1f})")

    trained_net, info = train_dqn(
        env, episodes=params["episodes"], max_steps=params["max_steps"], lr=params["lr"],
        hidden=(params["hidden"], params["hidden"]), buffer_capacity=params["buffer_capacity"],
        batch_size=params["batch_size"], train_freq=params["train_freq"],
        target_sync_freq=params["target_sync_freq"], gamma=params["gamma"],
        epsilon=params["epsilon"], epsilon_min=params["epsilon_min"],
        epsilon_decay=params["epsilon_decay"], seed=params["seed"],
        progress_callback=on_progress,
    )
    progress_bar.empty()

    after = run_episode(env, trained_net, epsilon=0.0, seed=params["seed"])

    st.session_state.results[4] = dict(before=before, after=after, info=info, cfg=cfg, env=env)


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
def plot_learning_curve(reward_history, window=20):
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(reward_history, color="gray", alpha=0.3, linewidth=0.7, label="episode reward")
    if len(reward_history) >= window:
        smoothed = np.convolve(reward_history, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(reward_history)), smoothed, color="orange", linewidth=2, label=f"moving avg ({window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    return fig


def render_room_preview(cfg):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.set_facecolor("#0d141c")
    fig.patch.set_facecolor("#0d141c")
    ax.add_patch(plt.Rectangle((0, 0), cfg.room_w, cfg.room_h, fill=False, edgecolor="#4a535c", linewidth=1.5))
    for z in cfg.wind_zones:
        ax.add_patch(plt.Rectangle((z.x, z.y), z.w, z.h, color="#3a8fd6", alpha=0.25))
    for w in cfg.walls:
        ax.add_patch(plt.Rectangle((w.x, w.y), w.w, w.h, color="#e05050", alpha=0.85))
    ax.add_patch(plt.Circle((cfg.pad.x, cfg.pad.y), cfg.pad.radius, color="#35ff8a", alpha=0.35))
    ax.plot(*cfg.start, "o", color="#ffb84d", markersize=10, markeredgecolor="black")
    ax.text(cfg.start[0], cfg.start[1] - 0.6, "start", ha="center", color="#ffb84d", fontsize=8)
    ax.text(cfg.pad.x, cfg.pad.y, "pad", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
    ax.set_xlim(-0.3, cfg.room_w + 0.3)
    ax.set_ylim(-0.3, cfg.room_h + 0.3)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, int(cfg.room_w) + 1, 2))
    ax.set_yticks(range(0, int(cfg.room_h) + 1, 2))
    ax.tick_params(colors="#7f97ab", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#4a535c")
    ax.set_title("Room preview (top-down, meters)", color="white", fontsize=10)
    plt.tight_layout()
    return fig


def render_results(params):
    st.subheader("📊 Results")
    res = st.session_state.results.get(4)
    if res is None:
        st.info("Configure the room in the sidebar, then press **Train Agent**. Preview below updates live.")
        st.pyplot(render_room_preview(build_drone_config(params)))
        return

    info = res["info"]
    st.pyplot(plot_learning_curve(info["reward_history"]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Landed (after)?", "✅ Yes" if res["after"]["solved"] else "❌ No")
    c2.metric("Episodes trained", info["episodes"])
    c3.metric("Final ε", f"{info['final_epsilon']:.3f}")
    c4.metric("Train time", f"{info['time_sec']:.1f}s")

    canvas_px = 420
    c1, c2 = st.columns(2)
    with c1:
        html = render_flight_canvas(res["before"], res["cfg"], "Before training (fresh network)",
                                     element_id="before", canvas_px=canvas_px)
        st.iframe(html, height=canvas_px + 110)
    with c2:
        html = render_flight_canvas(res["after"], res["cfg"], "After training",
                                     element_id="after", canvas_px=canvas_px)
        st.iframe(html, height=canvas_px + 110)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    st.title("🚁 Room 4 — The Drone Room")
    st.caption(
        "Fly a drone through wind and past danger gates to a soft landing. "
        "Function Approximation (DQN) — continuous state (X, Y, Vx, Vy)."
    )
    st.divider()

    is_training = st.session_state.get("is_training", False)
    if is_training:
        with st.sidebar:
            st.header("🚁 Flight Setup")
            st.warning("🔒 Training in progress — parameters are locked until it finishes.")
        params = st.session_state.get("_locked_params", {})
    else:
        params, train_clicked = render_sidebar()
        if train_clicked:
            st.session_state["is_training"] = True
            st.session_state["_locked_params"] = params
            st.rerun()

    if is_training:
        train_room4(params)  # shows its own progress bar
        st.session_state["is_training"] = False
        st.rerun()
    render_results(params)


if __name__ == "__main__":
    main()
