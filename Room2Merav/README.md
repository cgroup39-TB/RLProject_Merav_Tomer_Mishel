# Room 2 — The Collapsing Bridge (SARSA)

Escape-room chapter 2: an abandoned factory. The environment's transition
model is **unknown** to the agent, so it learns a policy purely from
experience using **SARSA** (on-policy Temporal-Difference control), rather
than by planning over a known model like Room 1's Dynamic Programming.

Theme: a chasm (the abyss) splits the factory in two. The only way across
is a single walkway (the bridge). A key, tucked away from the direct route,
unlocks the exit door — **you cannot leave this room for the next one
without it**, no matter how you cross the bridge. Slippery cells are
scattered through the factory, including the bridge itself — see "Core
idea: the reward function" below for why that's the point of this room.

## Run it

```bash
cd Room2Merav
pip install -r requirements.txt
streamlit run app.py
```

## State space

Discrete, **200 states** — `(row, col, has_key)` on the 10x10 grid,
flattened to a single index (`grid_env.GridWorld.encode_state`). `has_key`
is part of the *true* state, not just bookkeeping: the same `(row, col)`
cell means something different depending on whether the key has already
been collected (`G` only ends the episode once `has_key` is true).
Leaving it out of the state would break the Markov property SARSA's
update rule relies on. The agent only ever observes this index, never the
transition probabilities: that's what "model unknown" means here — SARSA
learns `Q(s, a)` purely from `(state, action, reward, next_state,
next_action)` tuples collected by interacting with the environment.

The bridge needs no equivalent flag. It isn't a "used once, then broken"
mechanic — see below — so whether it's safe to stand on never depends on
history, and it doesn't have to be part of the state.

## Action space

Discrete(4): `0=UP, 1=RIGHT, 2=DOWN, 3=LEFT`. Moving into a wall or the
grid border leaves the agent in place — a wasted step that still costs the
step reward.

## Core idea: the reward function

This room's design centers on its reward function — everything else (the
maze, the key, the bridge) exists to give that reward function something
to shape a policy around:

| Event | Reward |
|---|---|
| Every step | `-1` (time cost) |
| Falling into the abyss (`P`) — including a failed bridge crossing | `-20`, episode ends |
| Reaching the key (`K`), first time | `-1 + 10` (step cost + one-time subgoal bonus) |
| Reaching the door (`G`) **while holding the key** | `max(100 - steps_taken, 20)`, episode ends |
| Reaching `G` **without** the key | `-1` — door is locked, just a normal floor cell, episode continues |
| Truncated at `max_steps` without exiting | `0` bonus |

Why each piece is there:

- **The `-1` step cost** is what makes speed matter at all — without it,
  wandering forever would be free.
- **The `+10` key bonus** is reward shaping for a subgoal: the key is
  mandatory (the door never opens without it — see the state space note
  above), but a flat tabular agent has no way to "look ahead" to the
  distant exit payoff early in training. The one-time bonus gives it an
  immediate, learnable signal to bother detouring for the key instead of
  wandering the maze without purpose.
- **The `max(100 - steps_taken, 20)` exit bonus** directly implements the
  assignment's "the faster the agent escapes, the higher the reward"
  requirement: it's not just a flat completion bonus, it actively prefers
  shorter paths, with a floor of 20 so a slow-but-successful episode is
  still clearly better than failing.
- **The `-20` abyss penalty** is large enough, relative to the `-1`
  step cost, to make caution near the abyss worth learning even though
  the shortest path runs right past it.
- **The door staying locked (not fatal) without the key** means a
  premature visit to `G` just costs a wasted step, not the episode —
  the key requirement is enforced by reward structure, not by treating
  an early visit as an error.

Because SARSA is on-policy, its `Q(s, a)` updates bootstrap off the action
the agent's own epsilon-greedy policy will *actually* take next (see
"SARSA update rule" below) — so this reward function isn't just shaping a
final greedy policy, it's shaping what the agent learns to value *while
it's still exploring*, which matters a lot next to a `-20` hazard.

(Internally, the code keeps generic parameter names — `pit_reward`,
`key_bonus` — since `grid_env.py` is a generic engine shared by any 10x10
layout; `room2_env.py` is what actually ties them to the abyss and the key.)

## Slippery cells (`~`) and the bridge (`B`)

"Slippery" here doesn't mean water — it means a cell where, with
probability `slip_prob` (default `0.2`), the next state isn't the one the
action intended: the actual move slips to one of the two directions
**perpendicular** to it instead (FrozenLake-style).

The bridge is not a separate "collapses after one use" mechanic — it
behaves exactly like a `~` cell. Every time the agent is on the bridge and
attempts to cross, there's the same `slip_prob` chance the crossing slips
sideways into the abyss on either side — the first attempt is exactly as
risky as any later one, since there's no persistent "already used" state
to track. Given the layout, this means every episode's one mandatory
bridge-exit move carries an **irreducible** `slip_prob` chance of failure:
even a perfect policy can't push the achievable success rate above roughly
`1 - slip_prob` (~80% at the default). That's a deliberate design choice —
see "Optimal hyperparameters" below for how it shows up in practice.

## Room layout

```
S..#..#..K
.#.....#..
...#..#..#
#....#..#.
..#.~..#..
.#..#.#..#
P~.~PP~P~~
.PBP..P.PP
..........
.........G
```

`S` start · `K` key · `G` door/exit (locked until key collected) ·
`#` wall · `~` slippery · `P` abyss · `B` bridge (slippery, see above)

![Room 2 layout](room_layout.png)

Rendered to match Room 1's DP visualization exactly (`viz.py`), for a
consistent look across the merged multi-room app: `S` in lime, `G` in
red, the abyss (`P`) as an orangered `X` (Room 1's trap letter/color),
black squares for walls, and — once trained — a viridis heatmap of
`V(s) = max_a Q(s,a)` with a white greedy-policy arrow on every ordinary
cell, exactly like Room 1's post-training plot. Room 2 adds two cell
types Room 1 doesn't have (the key and the bridge), rendered in the same
bold-colored-letter style: `K` in gold, `B` in deep sky blue. Because
Room 2's state includes `has_key`, the app has a "Before key / After key"
toggle to view either state's heatmap and policy. The Streamlit sidebar's
legend and grid-editor icons also match Room 1's exact choices where the
concept overlaps (🚦 start, 🏁 goal, 🧱 wall, ≈ slippery, ☠ trap/abyss),
with new icons in the same style for the key (🔑) and bridge (🌉).

The upper maze's walls are scattered single cells rather than clustered
blocks, standing in individually for factory machinery. Slippery cells are
likewise scattered rather than forming one solid band along the abyss
edge, and the abyss itself is staggered across two rows (6-7) in a jagged
line rather than one uniform row of pits — every column except the bridge
is blocked in *either* row 6 or row 7 (never both left open), so the
barrier stays complete no matter which row does the blocking for that
column. Verified by BFS two ways: the maze is solvable with the bridge in
place, and removing the bridge (treating it as a wall) makes the goal
completely unreachable — confirming it really is the chasm's only gap,
not just incidentally so. "Repair area" (an added flavor note from the
assignment) is represented as the maze's walled sections generally — no
distinct game mechanic was specified for it, so it's treated as scenery
rather than a new hazard type.

BFS distances (ignoring the key requirement, since that changes *whether*
`G` terminates, not raw reachability): `S → K` is 17 steps, `K → G` is 23
more — around 40 steps total, all funneled through the bridge.

## SARSA update rule

```
Q(s, a) += alpha * (r + gamma * Q(s', a') - Q(s, a))
```

where `a'` is the action the agent will **actually** take next under its
own epsilon-greedy behavior policy — this is what makes SARSA on-policy:
it commits to the consequences of its own exploration when updating its
value estimates, rather than bootstrapping off an idealized greedy action.

## Optimal hyperparameters

Found via `hyperparam_sweep.py`, which trains on a grid of
`(alpha, gamma, epsilon_decay)` — `slip_prob` and the reward shape are the
room's fixed design, not swept — and ranks configs by final greedy success
rate, then by convergence speed.

Because the bridge is itself slippery (see above), the achievable success
rate is capped at roughly `1 - slip_prob` (~80% at `slip_prob=0.2`), not
100% — a big shift from a plain maze, and it makes this a genuinely hard
credit-assignment problem: many `(alpha, gamma)` combinations get stuck at
a much lower success rate (some effectively never learn to attempt the
crossing at all), while only a few configs reliably approach that ~80%
ceiling. `hyperparam_sweep.py`'s convergence threshold is set at 0.65 (not
a naive 0.9) specifically because of this ceiling — see the module
docstring.

| Parameter | Value |
|---|---|
| `alpha` (learning rate) | **0.2** |
| `gamma` (discount) | **0.95** |
| `epsilon_start` | 1.0 |
| `epsilon_min` | 0.05 |
| `epsilon_decay` | **0.99** |

These are `TrainRoom2Config`'s defaults: 84% greedy success at ~41 steps,
converging by ~episode 336 — see `sweep_results.csv` for the full grid.
Full run (3000 episodes, these defaults):

![Learning curves](learning_curve.png)

- Success rate: 0% (first 50 episodes) → ~70-75% (last 50 episodes,
  training-time — still with `epsilon_min=0.05` exploration noise, so a
  bit below the 84% pure-greedy figure)
- Average steps on success: ~41-47, consistent with the ~40-step
  `S → K → bridge → G` shortest route

Tested across 10 training seeds with these hyperparameters: all 10
learned a working crossing policy, each reaching 84% greedy success —
consistent and reliable at this tuning, on the current (jagged, two-row)
chasm layout.

## Episode replay

`app.py`'s training run records the first, middle, and last episode's full
trajectory (state/action/reward per step) so you can step through them
afterward — compare the wandering, undertrained Episode 0 against the
converged final episode. Once trained, both the room layout and the
replay view show the viridis `V(s)` heatmap and greedy-policy arrows
(see above) under whichever "Before key / After key" toggle is selected,
so you can watch how the path and the underlying value estimates differ
once the key changes what `G` means.

## Grid editor: per-cell overrides

Mirroring Room 1's paintable `Cell`/`GridConfig` system, any individual
cell (except `S`, `K`, `G`, which are structurally tied to the state
machine) can be customized independent of its layout symbol, via
`grid_env.CellOverride`:

- **Slippery, and how** — `slip_prob: float` gives that specific cell its
  own slip probability, overriding the global `slip_prob` (and applying
  even to a plain `.` cell that wasn't slippery at all).
- **A reward** — `reward: float` grants that value on every visit
  (non-terminal, repeatable — no extra state needed, unlike the key).
- **A terminal state** — `terminal_reward: float` ends the episode with
  that reward the moment the cell is reached, positive or negative —
  generalizing both the abyss (fixed at -20) and the goal (fixed,
  key-gated) into an arbitrary custom ending.

All three are independent — a cell can be slippery *and* grant a reward,
for instance — and overrides always take priority over the cell's base
symbol behavior. `app.py`'s "🎨 Grid editor" expander provides a paint-tool
UI for this: pick a tool (slippery/reward/terminal/clear), set its value,
then click cells in the 10x10 grid to apply it — the room preview and the
episode replay both mark overridden cells with small corner badges (blue
for slippery, green for reward, red for terminal — Room 1 has no
equivalent convention to match here, since its own `cell_rewards` isn't
visualized on the grid either). Training passes the current overrides
straight into `make_room2_env(cell_overrides=...)`, so whatever's painted
when you hit **Train Agent** is what the agent learns against.

## Random layout variants

`room2_env.generate_random_layout()` scatters walls/slippery cells/pits
across the base layout's floor tiles (keeping `S`, `K`, `B`, `G` fixed) for
anyone who wants a different maze to train on. A random scatter has no
inherent guarantee of staying solvable — corner cells like `G` have only
two neighbors, and a single obstacle on either one seals them off
completely — so each candidate is checked with a BFS and re-shuffled if
`S` can't reach both `K` and `G`, raising only if no solvable layout turns
up within `max_attempts`. Not used by the app or the tuned defaults above
(which are for the hand-designed `BASE_ROOM2_LAYOUT`); pass the result to
`make_room2_env(layout=...)` to train on a generated maze instead.

## Files

| File | Contents |
|---|---|
| `grid_env.py` | 10x10 grid engine: movement, walls, slip mechanic, key state, terminal states |
| `room2_env.py` | Room 2's specific layout and reward shaping |
| `sarsa_agent.py` | On-policy SARSA agent (epsilon-greedy, tabular `Q(s,a)`) |
| `train_room2.py` | Training loop, per-episode logging, trajectory recording |
| `hyperparam_sweep.py` | Sweeps alpha/gamma/epsilon_decay, ranks by success rate + convergence speed |
| `viz.py` | Learning-curve plots, grid rendering, episode-replay rendering |
| `app.py` | Streamlit UI: hyperparameter controls, training, plots, replay |
