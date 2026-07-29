"""
The numbers collected while an agent trains.

Every training loop in agents/ measures the same five things after each episode -
which episode it was, the total reward, how many steps it took, how much
exploration was still switched on, and whether the agent got out. Before this
file existed those six lines were copied into all five trainers; now each one
says three short sentences instead:

    log = EpisodeLog()          before the loop
    log.start()                 at the top of each episode
    log.add(state, action, reward)   once per step
    log.finish(episode, epsilon, escaped)   at the end of each episode

Note what is NOT in here: the update rule. `sarsa_agent.py` and
`qlearning_agent.py` still spell their own training loop out in full, because
the single line where they differ is the whole point of having both. Only the
bookkeeping moved.

KEEPING WHOLE EPISODES
----------------------
As well as one summary row per episode, the log holds on to three episodes in
full - every state, action and reward of them:

    first   what the agent did before it had learned anything
    best    its highest-scoring episode of the whole run
    final   what it does at the end

Watching those three in order is the clearest picture of what changed. Only
three are kept, so this costs almost nothing even over thousands of episodes.
"""


class EpisodeLog:
    """One row per episode, plus three whole episodes worth keeping."""

    def __init__(self):
        self.episode = []
        self.reward = []
        self.steps = []
        self.epsilon = []
        self.escaped = []

        # first / best / final, each a full record of one episode
        self.kept = {}
        self.trace = None

    def start(self):
        """Begin recording a new episode."""
        self.trace = {"states": [], "actions": [], "rewards": []}

    def add(self, state, action, reward):
        """One step: where we were, what we did, what we got for it."""
        self.trace["states"].append(state)
        self.trace["actions"].append(action)
        self.trace["rewards"].append(reward)

    def finish(self, episode, epsilon, escaped):
        """End of an episode: write the summary row and update the kept three."""
        total_reward = sum(self.trace["rewards"])
        steps = len(self.trace["actions"])

        self.episode.append(episode)
        self.reward.append(total_reward)
        self.steps.append(steps)
        self.epsilon.append(epsilon)
        self.escaped.append(1 if escaped else 0)

        record = {"episode": episode,
                  "states": self.trace["states"],
                  "actions": self.trace["actions"],
                  "rewards": self.trace["rewards"],
                  "total_reward": total_reward,
                  "steps": steps,
                  "success": bool(escaped)}

        if "first" not in self.kept:
            self.kept["first"] = record
        best = self.kept.get("best")
        if best is None or total_reward > best["total_reward"]:
            self.kept["best"] = record
        self.kept["final"] = record

    def as_dict(self):
        """
        The log in the shape the graphs and the CSV expect: five lists of the
        same length, one entry per episode.
        """
        return {"episode": self.episode,
                "reward": self.reward,
                "steps": self.steps,
                "epsilon": self.epsilon,
                "escaped": self.escaped}
