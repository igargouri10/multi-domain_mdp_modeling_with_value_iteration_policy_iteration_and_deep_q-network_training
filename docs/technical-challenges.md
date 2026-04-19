# Technical challenges encountered during Mini-2

This file documents problems that slowed the implementation and the
resolution we took. It satisfies the V1 "technical challenges" deliverable
and is intended to help future iterations of this project (and a reader who
wants to reproduce the work).

## 1. Three-branch wind distribution in Problem 1

The assignment's wind law (`p`, `(1-p) p²`, `(1-p)(1-p²)`) is subtle: the
probabilities must sum to one for the construction to yield a stochastic
kernel. Verifying
`p + (1-p)p² + (1-p)(1-p²) = p + (1-p)(p² + 1 − p²) = p + (1-p) = 1`
was the first step before coding. The symmetric split into `±1` and `±2`
shifts with each branch halved is our design choice that matches the PDF's
centre-line description and extends consistently to `j ≠ centre`.

## 2. Terminal logic ordering

On a `forward` action, the deterministic move may already land on the goal
row `i = H-1`. In that case we must skip the wind altogether: once the
drone is in the goal row, the wind has no physical meaning. We handled this
by checking `ip == H-1` *before* applying the wind kernel in
`WindyChasmMDP.__init__`.

## 3. Lidar-cone goal placement

Initial implementation put the goal at `(6, 4)` (one cell from the tip),
which turned `(6, 5)` into a dead-end where every action crashed.
`V*(start) ≈ −20` was symptomatic. Moving the goal to `(6, 5)` (the tip
itself) resolved the issue — `V*(start) = 26.83` — and matches the PDF's
intent (traverse the corridor along its axis).

## 4. State-space explosion in Problem 3

For `H = W = 3`, an unrestricted MDP has `|S| ≤ 9 · 3^9 · 2 = 354 294`
states. Tabular value iteration is memory-feasible but slow. We restrict
the builder with two knobs:
  - `max_components` (≤ 3): bounds simultaneous non-empty cells.
  - `max_steps`: caps episode length so the state includes `steps`.

This yields a 391-state reachable subgraph for the default spec that VI
solves in 7 sweeps — enough to validate the framework. The bigger point
(we document in the report) is that tabular DP is not the right tool here
once `H·W ≥ 12`.

## 5. Torch unavailability in the sandbox

Anthropic's sandbox cannot `pip install torch`. We wrote a pure-NumPy
DQN (Adam, ReLU MLP, uniform replay, hard target-network sync) so the V2
deliverable reproduces without any DL framework. The torch path is still
present as `agents/dqn.py` for users with PyTorch installed locally; both
share the same `ReplayBuffer`.

## 6. IEEEtran.cls unavailability

The TeX Live installation lacks `IEEEtran.cls` and `IEEEtran.bst`.
We use the standard `article` class with a manually-typeset IEEE-style
numbered `thebibliography` environment — the formatting (numbered
references, initials before surnames, italic journal names) matches the
IEEE transactions style without requiring the cls.

## 7. Matplotlib's non-interactive backend

The grid has no display. `matplotlib.use("Agg")` is set at the top of
`run_all.py` before any `pyplot` import; this is the only change needed
to make the driver work in a headless environment.

## 8. BFS state-space discovery for Problem 3

Enumerating states for the circuit MDP by naively iterating over grid
configurations wastes memory on unreachable configurations. We do a BFS
from the start state instead, expanding children as we encounter them.
This keeps the state count to 391 for the default spec.
