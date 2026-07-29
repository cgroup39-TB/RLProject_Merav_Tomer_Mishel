"""
Sector 3 - The Reactor Bay, solved by Q-Learning.

Three jobs in order before the door will open, so the state needs a stage
counter and the greedy move points the wrong way for most of the episode.

The dictionary below is picked up by definitions.py, which collects the five
rooms into one ROOMS table. Everything about this sector - its map, its story,
its tuned defaults - lives here and nowhere else.
"""

# Three jobs, in order, before the door will open: fetch the power cell K in the
# bottom-left corner, throw breaker A in the top-right one, then breaker B on the
# way down. The exit is passed twice before it is any use, which is exactly the
# point - the greedy move and the correct move disagree for most of the episode.
ROOM3_MAP = """
S...#....A
....#.~~..
.~~.#.~~..
.~~.......
....#.....
###.#.###.
K...#.....
....#....B
..........
....#....D
"""

ROOM = {
    "title": "Room 3 - The Reactor Bay",
    "short": "Reactor Bay",
    "kind": "grid",
    "map": ROOM3_MAP,
    "tasks": "KAB",    # power cell, then breaker A, then breaker B
    "task_names": ["the power cell", "breaker A", "breaker B"],
    "algorithm": "Q-Learning (model unknown)",
    "lesson": "a three-part job, where the exit is useless until it is done",
    "prize": "Power Cell",
    "prize_icon": "⚡",
    "story": "The Security Keycard opens the reactor bay, but the blast door on "
             "the far side is dead without power. The spare Power Cell is racked in "
             "the opposite corner, so for the first time R-5 has to walk AWAY from "
             "the exit to be able to leave at all - the greedy move and the correct "
             "move point in opposite directions. Fetching the cell also charges the "
             "drone in the next bay, which is the only reason sector 4 can be "
             "entered.",
    "state": "(row, column, stage) - a 10 x 10 x 4 table, of which 344 states can "
             "actually be stood on. `stage` counts how many of the "
             "three jobs are done (0, 1, 2 or 3). That extra number is what makes "
             "the task solvable at all: without it the agent could not tell "
             "'I am at the door' from 'I am at the door with the job finished', "
             "and those two situations need completely different actions.",
    "actions": "4 actions: up, down, left, right.",
    "why": "Q-Learning always learns from the best possible continuation, so it "
           "finds the shortest cell-breaker-breaker-door route even while it is "
           "still exploring. Four stages of one table are learned at once.",
    "defaults": {"slip_prob": 0.1,
                 "ice_prob": 0.25,
                 "episodes": 1500,
                 "alpha": 0.15,
                 "gamma": 0.97,
                 "epsilon_start": 1.0,
                 "epsilon_end": 0.05,
                 "epsilon_decay": 0.998,
                 "max_steps": 400},
}
