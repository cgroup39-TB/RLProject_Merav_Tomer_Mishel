"""
Room 2 — SARSA, unknown model, slippery grid.

Standalone Streamlit app for Room 2: configure hyperparameters, train the
SARSA agent, inspect learning curves, and replay individual episodes.

Run locally with:
    streamlit run app.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import streamlit as st

from room2_env import ROOM2_LAYOUT
from train_room2 import TrainRoom2Config, train
from viz import plot_learning_curves, render_grid, render_episode_step

st.set_page_config(page_title="Room 2 — Collapsing Bridge (SARSA)", page_icon="🏭", layout="wide")

DEFAULTS = TrainRoom2Config()


def render_page_header() -> None:
    st.markdown(
        """
        <div style='padding:24px; background:linear-gradient(180deg, rgba(18,22,29,0.95), rgba(8,10,14,0.95)); border:1px solid #2a2e34; border-radius:22px; margin-bottom:20px;'>
            <h1 style='margin:0; color:#ff8a1a; font-size:2.4rem;'>🌉 Room 2 — The Collapsing Bridge</h1>
            <p style='margin:10px 0 0 0; color:#cbd5e1; font-size:1.05rem; line-height:1.65;'>SARSA explores an abandoned factory, secures the access card, and chooses the safest path across a collapsing bridge while avoiding slippery pipes and a bottomless abyss.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def init_state():
    if "initialized" in st.session_state:
        return
    st.session_state.initialized = True
    st.session_state.history = None
    st.session_state.trajectories = None
    st.session_state.trained_cfg = None


init_state()


def render_sidebar() -> TrainRoom2Config:
    with st.sidebar:
        st.markdown("### Training configuration")
        alpha = st.slider("Learning rate α", 0.01, 1.0, DEFAULTS.alpha, step=0.01)
        gamma = st.slider("Discount γ", 0.0, 0.999, DEFAULTS.gamma, step=0.01)
        epsilon_start = st.slider("Initial ε", 0.0, 1.0, DEFAULTS.epsilon_start, step=0.05)
        epsilon_min = st.slider("Min ε", 0.0, 0.5, DEFAULTS.epsilon_min, step=0.01)
        epsilon_decay = st.slider(
            "ε decay per episode", 0.90, 0.9999, DEFAULTS.epsilon_decay, step=0.0005, format="%.4f"
        )

        st.divider()
        st.markdown("### Environment hazards")
        slip_prob = st.slider("Slip probability", 0.0, 1.0, DEFAULTS.slip_prob, step=0.05)
        max_steps = st.number_input("Max steps / episode", 20, 1000, DEFAULTS.max_steps, step=10)

        st.divider()
        st.markdown("### Run settings")
        episodes = st.number_input("Episodes", 100, 20000, DEFAULTS.episodes, step=100)
        seed = st.number_input("Seed", 0, 10_000, DEFAULTS.seed, step=1)

        train_clicked = st.button("▶ Train Agent", type="primary", use_container_width=True)

        cfg = TrainRoom2Config(
            episodes=int(episodes),
            max_steps=int(max_steps),
            alpha=alpha,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            slip_prob=slip_prob,
            seed=int(seed),
            results_dir="results/room2",
        )
        return cfg, train_clicked


def render_metrics(history: list[dict], window: int = 50):
    last_window = history[-window:] if len(history) >= window else history
    success_rate = sum(h["success"] for h in last_window) / len(last_window)
    successful_steps = [h["steps"] for h in last_window if h["success"]]
    avg_steps = sum(successful_steps) / len(successful_steps) if successful_steps else float("nan")

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Success rate (last {len(last_window)} ep)", f"{success_rate:.0%}")
    c2.metric("Avg steps to exit (successes)", f"{avg_steps:.1f}" if successful_steps else "—")
    c3.metric("Final ε", f"{history[-1]['epsilon']:.3f}")


def render_room_briefing() -> None:
    st.markdown("### Mission briefing")
    st.markdown("- **משימה:** קח את הכרטיס (`K`) ואז פתח את הדלת (`G`).")
    st.markdown("- **סכנות:** תהום (`P`), רצפה חלקלקה (`~`), וגשר שקורס (`B`).")
    st.markdown("- **עיקרון:** SARSA לומד להימנע מסיכונים בסביבה חלקלקה.")
    st.markdown("### Legend")
    st.markdown("- 🏮 S: start")
    st.markdown("- 🔑 K: access card")
    st.markdown("- 🚪 G: locked exit")
    st.markdown("- 💧 ~: slippery floor")
    st.markdown("- 🕳 P: abyss")
    st.markdown("- 🌉 B: collapsing bridge")


def render_replay(trajectories: dict):
    st.subheader("Episode replay")
    if not trajectories:
        st.info("No recorded episodes.")
        return
    ep_keys = sorted(trajectories.keys())
    labels = {ep: f"Episode {ep}" for ep in ep_keys}
    chosen = st.selectbox("Pick a recorded episode", ep_keys, format_func=lambda e: labels[e])
    trajectory = trajectories[chosen]
    step = st.slider("Step", 0, len(trajectory) - 1, len(trajectory) - 1)
    st.pyplot(render_episode_step(ROOM2_LAYOUT, trajectory, step))


def main():
    render_page_header()

    cfg, train_clicked = render_sidebar()

    col_grid, col_info = st.columns([2, 1])
    with col_grid:
        st.subheader("Room layout")
        st.pyplot(render_grid(ROOM2_LAYOUT, title="Room 2: The Collapsing Bridge"))
        st.caption(
            "S=start · K=access card · G=door/exit · ~ slippery floor · P=abyss · B=bridge"
        )

    if train_clicked:
        with st.spinner(f"Training SARSA for {cfg.episodes} episodes..."):
            _, history, trajectories = train(cfg)
        st.session_state.history = history
        st.session_state.trajectories = trajectories
        st.session_state.trained_cfg = cfg

    with col_info:
        if st.session_state.history is None:
            st.info("Set your hyperparameters and press **Train Agent**.")
        else:
            render_metrics(st.session_state.history)

        st.divider()
        render_room_briefing()

    if st.session_state.history is not None:
        st.divider()
        st.subheader("Learning curves")
        st.pyplot(plot_learning_curves(st.session_state.history))

        st.divider()
        render_replay(st.session_state.trajectories)


if __name__ == "__main__":
    main()
