"""
The shape every room in this game has.

Both kinds of room in this project - the 10x10 grid of rooms 1 to 3, and the
continuous 10x10 metre field of rooms 4 and 5 - answer the same five questions.
Writing that down in one place is what lets the training loops, the replay panel
and the self test treat any room the same way, without caring which one it is.

    reset()        start a new episode, and hand back the first state
    step(action)   do one action -> next_state, reward, done, info
    get_state()    what the agent can see right now
    is_terminal()  is the episode over?
    render()       a plain description of the room, for the drawing code

`step` returns FOUR things. The first three are the ones the learning
algorithms use. The fourth, `info`, is a small dictionary of extra facts that
are useful for watching and debugging but that the agent is NOT allowed to
learn from - how many jobs are done, how many seconds have passed, whether the
last move was a crash. Keeping them out of the state is the point: the agent
must work with `next_state` alone.
"""


class BaseRoom:
    """The interface. Every room below fills these in."""

    def reset(self):
        """Begin a new episode and return the first state."""
        raise NotImplementedError

    def step(self, action):
        """Do one action. Returns (next_state, reward, done, info)."""
        raise NotImplementedError

    def get_state(self):
        """What the agent can see at this moment."""
        raise NotImplementedError

    def is_terminal(self, state=None):
        """
        Is the episode over? With no argument this asks about where the agent is
        standing now; Dynamic Programming asks about states it is only thinking
        about, so it passes one in.
        """
        raise NotImplementedError

    def render(self):
        """
        A plain description of the room right now - letters for a grid room, a
        dictionary of positions for the continuous one. The drawing code turns
        this into a picture; nothing here knows about matplotlib.
        """
        raise NotImplementedError

    def escaped(self):
        """Did the episode end at the exit, rather than badly or not at all?"""
        raise NotImplementedError
