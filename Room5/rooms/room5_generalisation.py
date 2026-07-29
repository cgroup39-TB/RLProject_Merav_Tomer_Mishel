"""
Sector 5 - The Shifting Warehouse, solved by semi-gradient Q-Learning.

The crates stand somewhere new every episode, so nothing about this room can
be memorised. All the agent gets is a short-range radar.

The dictionary below is picked up by definitions.py, which collects the five
rooms into one ROOMS table. Everything about this sector - its map, its story,
its tuned defaults - lives here and nowhere else.
"""

ROOM = {
    "title": "Room 5 - The Shifting Warehouse",
    "short": "Shifting Warehouse",
    "kind": "continuous",
    "algorithm": "Semi-gradient Q-Learning with a linear model (function approximation)",
    "lesson": "generalising to a room you have never seen",
    "prize": "Exit Key",
    "prize_icon": "🗝️",
    "story": "The last sector is the loading warehouse, and its stacking robots "
             "never stopped working: 0.5 metre crates stand somewhere new at the "
             "start of every single run. R-5 cannot memorise this room, because "
             "this room is never the same room twice. It cannot even see it - all "
             "it gets is a short-range radar reporting how far the nearest crate "
             "is in each direction. Learning a route is useless here; the only "
             "thing worth learning is a rule about what the radar says. Solve it "
             "and the Exit Key is yours.",
    "state": "(x, y, vx, vy) plus one radar distance per ray. The radar is the "
             "interesting part: it is what lets one set of weights work in a "
             "layout the agent has never seen.",
    "actions": "9 actions: every combination of vx and vy from {-1, 0, +1} m/s.",
    "why": "Position features alone cannot help when the obstacles move. Radar "
           "features can: 'the block on my right is very close' means the same "
           "thing in every layout, so the learned rule transfers.",
    "defaults": {"episodes": 700,
                 "alpha": 0.3,
                 "gamma": 0.995,
                 "epsilon_start": 1.0,
                 "epsilon_end": 0.05,
                 "epsilon_decay": 0.995,
                 "decision_interval": 25,
                 "max_seconds": 25.0,
                 "n_tilings": 4,
                 "tiles": 10,
                 "progress_reward": 4.0,
                 "reverse_cost": -2.0,
                 "n_obstacles": 6,
                 "lookahead_m": 2.0,
                 "n_rays": 16,
                 "ray_bins": 4},
}
