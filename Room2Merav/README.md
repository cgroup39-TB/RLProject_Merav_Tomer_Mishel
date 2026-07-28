# Room 2 — SARSA (unknown model)

Escape-room chapter 2: the environment's transition model is **unknown** to
the agent, so it learns a policy purely from experience using **SARSA**
(on-policy Temporal-Difference control), rather than by planning over a
known model like Room 1's Dynamic Programming.

## Run it

```bash
cd Room2Merav
pip install -r requirements.txt
streamlit run app.py
```

## State space

Discrete, 100 states — one per `(row, col)` cell on the 10x10 grid,
flattened to an index `0..99` (`row * 10 + col`). The agent only ever
observes this index, never the transition probabilities: that's what
"model unknown" means here — SARSA learns `Q(s, a)` purely from
`(state, action, reward, next_state, next_action)` tuples collected by
interacting with the environment, never from a `P(s'|s,a)` table.

## Action space

Discrete(4): `0=UP, 1=RIGHT, 2=DOWN, 3=LEFT`. Moving into a wall or the
grid border leaves the agent in place — a wasted step that still costs the
step reward.

## Reward function

| Event | Reward |
|---|---|
| Every step | `-1` (time cost) |
| Stepping onto a trap (`T`) | `-20`, episode ends |
| Reaching the goal (`G`, the single exit) | `max(100 - steps_taken, 20)` |
| Truncated at `max_steps` without reaching `G` | `0` bonus |

The goal bonus shrinks with the number of steps taken (down to a floor of
20), which directly implements the assignment's "the faster the agent
escapes, the higher the reward" requirement, instead of leaving that
implicit in the discount factor alone.

## Slippery cells (`~`)

With probability `slip_prob` (default `0.2`) the actual move is one of the
two directions **perpendicular** to the intended one (FrozenLake-style),
instead of the intended direction. This is what makes the model
"unknown" in practice: the agent can't compute exact transition
probabilities in advance and has to learn the effect of slipping from
experience.

## Room layout

```
S...#.....
.##.#.###.
....#.....
.###.#.##.
....~.....
.#.#~#.#..
.#.#~#.#..
.#...#.#..
.#.###.#T.
.......#.G
```

`S` start · `G` goal/exit · `#` wall · `~` slippery · `T` trap

![Room 2 layout](room_layout.png)

A wall maze harder than Room 1's open grid, a 3-cell vertical slip
corridor the agent must cross, and one trap placed off the shortest path
so it punishes careless exploration without being unavoidable. The
shortest path (BFS) from `S` to `G` is **18 steps**.

## SARSA update rule

```
Q(s, a) += alpha * (r + gamma * Q(s', a') - Q(s, a))
```

where `a'` is the action the agent will **actually** take next under its
own epsilon-greedy behavior policy — not the greedy max (that's
Q-Learning's off-policy update, Room 3). This is what makes SARSA
on-policy: it commits to the consequences of its own exploration when
updating its value estimates, which tends to make it learn more
conservative routes around traps/slippery hazards than Q-Learning would.

## Optimal hyperparameters

Found via `hyperparam_sweep.py`, which trains on a grid of
`(alpha, gamma, epsilon_decay)` — `slip_prob` and the reward shape are the
room's fixed design, not swept — and ranks configs first by final greedy
success rate, then by **how fast** they converge (first episode where a
50-episode rolling success rate reaches 90%). Most reasonable configs
eventually reach 100% success on this maze, so convergence speed is the
real differentiator, and it's the one that matters for a "faster escape"
framing.

| Parameter | Value |
|---|---|
| `alpha` (learning rate) | **0.2** |
| `gamma` (discount) | **0.9** |
| `epsilon_start` | 1.0 |
| `epsilon_min` | 0.05 |
| `epsilon_decay` | **0.99** |

These are `TrainRoom2Config`'s defaults. This config reaches 100% greedy
success in ~125 episodes and converges to the BFS-optimal ~18-19 step
path, versus 200-1000+ episodes for slower epsilon decays (`0.995`,
`0.999`) — see `sweep_results.csv` for the full grid. Full run (3000
episodes, these defaults):

![Learning curves](learning_curve.png)

- Success rate: ~15% (first 50 episodes) → ~98-100% (last 50 episodes)
- Average steps on success: ~19, matching the BFS-optimal 18-step path

## Episode replay

`app.py`'s training run records the first, middle, and last episode's full
trajectory (state/action/reward per step) so you can step through them
afterward — compare the wandering, undertrained Episode 0 against the
converged final episode.

## Files

| File | Contents |
|---|---|
| `grid_env.py` | 10x10 grid engine: movement, walls, slip mechanic, terminal states |
| `room2_env.py` | Room 2's specific layout and reward shaping |
| `sarsa_agent.py` | On-policy SARSA agent (epsilon-greedy, tabular `Q(s,a)`) |
| `train_room2.py` | Training loop, per-episode logging, trajectory recording |
| `hyperparam_sweep.py` | Sweeps alpha/gamma/epsilon_decay, ranks by success rate + convergence speed |
| `viz.py` | Learning-curve plots, grid rendering, episode-replay rendering |
| `app.py` | Streamlit UI: hyperparameter controls, training, plots, replay |
