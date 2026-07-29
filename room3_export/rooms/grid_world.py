"""
The 10x10 grid room used by rooms 1, 2 and 3.

A room is written as a picture made of letters (see rooms.py), for example:

    S....#....
    .....#....
    ..~~.#....
    ..~~.....D

    .  an empty floor tile
    #  a wall (you cannot walk into it)
    ~  a wet tile (you might slide in a random direction)
    %  an icy tile (the same idea, but much more slippery)
    S  where the agent starts
    D  the door = the exit of the room (the only way to finish)
    X  a trap (the episode ends immediately, big negative reward)
    T  a teleport: step on one and you come out at the other one
    K  the first job of the room, A the second, B the third (see `tasks` below)

JOBS TO DO BEFORE THE DOOR OPENS
--------------------------------
A room can ask for a list of tiles to be visited IN ORDER before its door
unlocks. Room 3 asks for "KAB": fetch the power cell, then throw breaker A,
then breaker B. The state carries how far down that list the agent has got,
so the state of a grid room is always the triple

    (row, column, stage)

with stage 0 meaning "nothing done yet" and stage len(tasks) meaning "the door
is open". Rooms 1 and 2 ask for nothing, so their stage is always 0 and they
have only 100 states.

THE MOST IMPORTANT IDEA IN THIS FILE
------------------------------------
This class offers two different ways to interact with the room:

1. transitions(state, action) -> tells you EVERYTHING that could happen,
   together with the probability of each outcome. This is "the model of the
   environment". Room 1 (Dynamic Programming) is allowed to read it, because
   in room 1 we pretend the model is known.

2. step(action) -> actually performs the action and returns only ONE result,
   picked at random using those same probabilities. Rooms 2 and 3 (SARSA and
   Q-Learning) may only use this one, because there we pretend we do NOT know
   the model and can only learn by trying things out.

Both use the same rules, so the two kinds of algorithms really do solve the
same room - they just get different amounts of information.
"""

import random

from rooms.base_room import BaseRoom

# The letters used in the room pictures.
EMPTY = "."
WALL = "#"
SLIPPERY = "~"
ICE = "%"
START = "S"
DOOR = "D"
TRAP = "X"
TELEPORT = "T"
KEY = "K"
SWITCH_A = "A"
SWITCH_B = "B"

# The tiles a room can ask the agent to visit, in order, before the door opens.
TASK_LETTERS = [KEY, SWITCH_A, SWITCH_B]

# The four things the agent can do. Each action is a (row change, column change).
# Row 0 is the top row, so "up" means row - 1.
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["up", "down", "left", "right"]

# Arrows used when we draw the learned policy on the map.
ACTION_ARROWS = ["^", "v", "<", ">"]


class GridWorld(BaseRoom):
    """One grid room. A state is the tuple (row, column, stage)."""

    def __init__(self,
                 text_map,
                 slip_prob=0.2,
                 ice_prob=0.45,
                 step_reward=-1.0,
                 door_reward=100.0,
                 trap_reward=-100.0,
                 wall_reward=-1.0,
                 task_reward=10.0,
                 tasks=""):
        self.grid = [list(line) for line in text_map.strip().splitlines()]
        self.n_rows = len(self.grid)
        self.n_cols = len(self.grid[0])
        self.slip_prob = slip_prob
        self.ice_prob = ice_prob
        self.step_reward = step_reward
        self.door_reward = door_reward
        self.trap_reward = trap_reward
        self.wall_reward = wall_reward
        self.task_reward = task_reward
        self.tasks = tasks

        # One stage per job, plus the stage where everything is done. A room with
        # no jobs has a single stage, which keeps it at 100 states instead of 400.
        self.n_stages = len(tasks) + 1

        self.start = self.find_letter(START)
        self.door = self.find_letter(DOOR)
        self.task_positions = [self.find_letter(letter) for letter in tasks]
        self.teleports = self.find_all(TELEPORT)
        self.state = self.reset()

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def find_letter(self, letter):
        """Where is this letter on the map? Returns (row, column) or None."""
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                if self.grid[row][col] == letter:
                    return (row, col)
        return None

    def find_all(self, letter):
        """Every position holding this letter. Used for the pair of teleports."""
        found = []
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                if self.grid[row][col] == letter:
                    found.append((row, col))
        return found

    def cell(self, row, col):
        """The letter at this position."""
        return self.grid[row][col]

    def teleport_target(self, row, col):
        """Step on one teleport and you come out at the other one."""
        first, second = self.teleports
        if (row, col) == first:
            return second
        return first

    def slip_chance(self, row, col):
        """
        How likely is this tile to send the agent somewhere it did not choose?
        Dry floor is not slippery at all, a wet tile a little, ice a lot.
        """
        letter = self.cell(row, col)
        if letter == SLIPPERY:
            return self.slip_prob
        if letter == ICE:
            return self.ice_prob
        return 0.0

    def is_inside(self, row, col):
        """Is this position still on the map?"""
        inside_rows = 0 <= row < self.n_rows
        inside_cols = 0 <= col < self.n_cols
        return inside_rows and inside_cols

    def all_states(self):
        """Every state of this room. Dynamic Programming (room 1) walks over this list."""
        states = []
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                if self.cell(row, col) == WALL:
                    continue
                for stage in range(self.n_stages):
                    states.append((row, col, stage))
        return states

    def value_table_shape(self):
        """Shape of a table that stores one number per state."""
        return (self.n_rows, self.n_cols, self.n_stages)

    def q_table_shape(self):
        """Shape of a table that stores one number per state AND action."""
        return (self.n_rows, self.n_cols, self.n_stages, len(ACTIONS))

    def door_is_open(self, stage):
        """The door unlocks only once every job on the list has been done."""
        return stage == len(self.tasks)

    def is_terminal(self, state=None):
        """
        Is this state an end of the episode? (the open door, or a trap)
        With no argument it asks about where the agent is standing now.
        """
        if state is None:
            state = self.state
        row, col, stage = state
        letter = self.cell(row, col)
        if letter == TRAP:
            return True
        if letter == DOOR:
            return self.door_is_open(stage)
        return False

    def escaped(self, state=None):
        """Did the agent finish the room through the door (and not through a trap)?"""
        if state is None:
            state = self.state
        row, col, stage = state
        if self.cell(row, col) != DOOR:
            return False
        return self.door_is_open(stage)

    # ------------------------------------------------------------------
    # The rules of the room
    # ------------------------------------------------------------------

    def move(self, state, action):
        """
        What happens if the agent really moves one tile in this direction?
        Returns (next_state, reward, done). No randomness here - the randomness
        of the slippery tiles is handled in transitions().
        """
        row, col, stage = state
        change_row, change_col = ACTIONS[action]
        new_row = row + change_row
        new_col = col + change_col
        reward = self.step_reward

        # Walking into a wall (or off the map) means staying where you are.
        outside = not self.is_inside(new_row, new_col)
        if outside or self.cell(new_row, new_col) == WALL:
            new_row = row
            new_col = col
            reward += self.wall_reward

        letter = self.cell(new_row, new_col)

        # A teleport moves the agent straight to its twin. Coming OUT of a
        # teleport does not set it off again, because we jump once and then look
        # at what we landed on.
        if letter == TELEPORT:
            new_row, new_col = self.teleport_target(new_row, new_col)
            letter = self.cell(new_row, new_col)

        # Is this the next job on the room's list? Jobs must be done in order,
        # so walking over a later one early simply does nothing.
        next_job = self.tasks[stage] if stage < len(self.tasks) else None
        if letter == next_job:
            stage += 1
            reward += self.task_reward

        done = False
        if letter == TRAP:
            reward += self.trap_reward
            done = True
        elif letter == DOOR and self.door_is_open(stage):
            reward += self.door_reward
            done = True

        return (new_row, new_col, stage), reward, done

    def transitions(self, state, action):
        """
        THE MODEL. Every possible outcome of doing `action` in `state`, as a list
        of (probability, next_state, reward, done).

        Only room 1 (Dynamic Programming) is allowed to look at this.
        """
        if self.is_terminal(state):
            return [(1.0, state, 0.0, True)]

        row, col, _ = state
        chance = self.slip_chance(row, col)

        # Dry floor: the action always does exactly what you asked.
        if chance == 0.0:
            next_state, reward, done = self.move(state, action)
            return [(1.0, next_state, reward, done)]

        # A wet or icy tile: with probability `chance` the agent slides in a
        # random direction instead. "Random" includes the direction it wanted,
        # so the wanted direction is a little more likely than the others.
        outcomes = []
        random_share = chance / len(ACTIONS)
        for other_action in range(len(ACTIONS)):
            probability = random_share
            if other_action == action:
                probability += 1.0 - chance
            next_state, reward, done = self.move(state, other_action)
            outcomes.append((probability, next_state, reward, done))
        return outcomes

    # ------------------------------------------------------------------
    # Playing the room step by step (all that rooms 2 and 3 may use)
    # ------------------------------------------------------------------

    def reset(self):
        """Put the agent back at the S tile and start a new episode."""
        self.state = (self.start[0], self.start[1], 0)
        return self.state

    def step(self, action):
        """
        Do one action and see what happens.
        Returns (next_state, reward, done, info).

        The result is drawn at random from the outcomes of transitions(), which
        is why a learning agent can use this without knowing the model.
        `info` is for watching, not for learning.
        """
        outcomes = self.transitions(self.state, action)
        probabilities = [outcome[0] for outcome in outcomes]
        chosen = random.choices(outcomes, weights=probabilities)[0]
        _, next_state, reward, done = chosen
        self.state = next_state
        info = {"stage": next_state[2],
                "tile": self.cell(next_state[0], next_state[1])}
        return next_state, reward, done, info

    def get_state(self):
        """What the agent can see: its square and how many jobs it has done."""
        return self.state

    def render(self):
        """The map as rows of letters, with the agent drawn on top as @."""
        picture = [row.copy() for row in self.grid]
        row, col, _ = self.state
        picture[row][col] = "@"
        return ["".join(line) for line in picture]
