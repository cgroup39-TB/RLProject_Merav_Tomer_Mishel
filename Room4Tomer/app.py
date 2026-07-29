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

import drone_env
from drone_env import DroneEnv, Pad
from dqn_solver import QNetwork, train_dqn, run_episode
from flight_canvas import render_flight_canvas

if not st.session_state.get("_embedded"):
    st.set_page_config(page_title="Escape Room RL — Room 4", page_icon="🚁", layout="wide")

PRESET_NAMES = list(drone_env.PRESETS.keys())
PRESET_DESCRIPTIONS = {
    "Open Room": "No obstacles — empty room, just the pad. Good for testing dynamics/reward tuning alone.",
    "Wind Corridor": "2 static walls forming a corridor, plus 1 wind zone pushing across it.",
    "Gate Gauntlet": "2 moving gates (oscillating obstacles), no walls or wind.",
}


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
    cfg = drone_env.PRESETS[params["preset"]]()
    for z in cfg.wind_zones:
        z.fx *= params["wind_scale"]
        z.fy *= params["wind_scale"]
        z.gust_std *= params["wind_scale"]
    for z in cfg.accel_zones:
        z.fx *= params["accel_decel_scale"]
        z.fy *= params["accel_decel_scale"]
    for z in cfg.decel_zones:
        z.drag_coeff *= params["accel_decel_scale"]
    cfg.pad = Pad(cfg.pad.x, cfg.pad.y, params["pad_radius"])
    return cfg


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.header("🚁 Flight Setup")

        st.subheader("Training run")
        episodes = st.number_input("Episodes", min_value=20, max_value=3000, value=800, step=20, key="episodes")
        max_steps = st.number_input("Max steps / episode", min_value=50, max_value=1000, value=300, step=25, key="max_steps")
        seed = st.number_input("Random seed", min_value=0, value=0, step=1, key="seed")

        st.divider()
        st.subheader("Room layout")
        preset = st.selectbox("Preset", options=PRESET_NAMES, key="preset")
        st.caption(PRESET_DESCRIPTIONS.get(preset, ""))
        wind_scale = st.slider("Wind strength ×", 0.0, 3.0, 1.0, key="wind_scale")
        accel_decel_scale = st.slider("Accel/decel zone strength ×", 0.0, 3.0, 1.0, key="accel_decel_scale")

        st.divider()
        st.subheader("Drone dynamics")
        max_cmd_speed = st.slider("Max commanded speed (m/s)", 0.5, 4.0, 2.0, key="max_cmd_speed")
        max_accel = st.slider("Engine responsiveness (m/s²)", 0.5, 15.0, 8.0, key="max_accel")
        drone_radius = st.slider("Drone radius (m)", 0.05, 0.5, 0.15, key="drone_radius")

        st.divider()
        st.subheader("Landing rules")
        landing_speed_threshold = st.slider("Max safe landing speed (m/s)", 0.1, 2.0, 0.8, key="landing_speed_threshold")
        pad_radius = st.slider("Landing pad radius (m)", 0.2, 1.5, 0.7, key="pad_radius")

        st.divider()
        st.subheader("Reward shaping")
        step_reward = st.number_input("Step penalty", value=-0.01, step=0.01, format="%.3f", key="step_reward")
        shaping_weight = st.slider("Distance-shaping weight", 0.0, 6.0, 4.0, key="shaping_weight")
        crash_penalty = st.number_input("Crash penalty", value=-50.0, step=5.0, key="crash_penalty")
        landing_bonus_base = st.number_input("Landing bonus (base)", value=200.0, step=10.0, key="landing_bonus_base")
        landing_bonus_gentleness = st.number_input("Landing bonus (gentleness)", value=50.0, step=10.0, key="landing_bonus_gentleness")

        st.divider()
        st.subheader("DQN hyperparameters")
        lr = st.select_slider("Learning rate", options=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2], value=1e-3, key="lr")
        hidden = st.select_slider("Hidden layer size (×2 layers)", options=[16, 32, 64, 128], value=32, key="hidden")
        gamma = st.slider("Discount γ", 0.90, 0.999, 0.99, key="gamma")
        buffer_capacity = st.number_input("Replay buffer capacity", min_value=1000, max_value=200000, value=30000, step=1000, key="buffer_capacity")
        batch_size = st.select_slider("Batch size", options=[16, 32, 64, 128, 256], value=64, key="batch_size")
        train_freq = st.slider("Train every N steps", 1, 10, 4, key="train_freq")
        target_sync_freq = st.number_input("Target network sync (steps)", min_value=50, max_value=5000, value=300, step=50, key="target_sync_freq")
        epsilon = st.slider("Initial ε (exploration)", 0.0, 1.0, 1.0, key="epsilon")
        epsilon_min = st.slider("Min ε", 0.0, 0.5, 0.1, key="epsilon_min")
        epsilon_decay = st.slider("ε decay per episode", 0.90, 0.9999, 0.996, key="epsilon_decay", format="%.4f")

        st.divider()
        train_clicked = st.button("▶ Train Agent", type="primary", use_container_width=True)

        params = dict(
            episodes=int(episodes), max_steps=int(max_steps), seed=int(seed),
            preset=preset, wind_scale=wind_scale, accel_decel_scale=accel_decel_scale,
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


def render_results():
    st.subheader("📊 Results")
    res = st.session_state.results.get(4)
    if res is None:
        st.info("Configure the room and press **Train Agent**.")
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
        "Fly a drone through wind, moving gates, and speed zones to a soft landing. "
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
    render_results()


if __name__ == "__main__":
    main()
