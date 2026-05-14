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

---

## V3 additions: Bayesian hyperparameter optimization

### N+1. Proxy mismatch between `tail_mean` and `eval_return_start`

**Problem.** The first V3 BO sweep maximized the trailing-window mean of
training returns (`tail_mean`, last 50 episodes). The "best" config
returned `tail_mean = -44.69` but its greedy policy from the start state
gave `eval_return_start = -63.28`, worse than the V2 hand-tuned baseline.
Meanwhile a *different* config in the same sweep gave
`eval_return_start = -10.20` while only ranking 5th by tail_mean.

**Cause.** The trailing-window mean averages over noisy ε-greedy
rollouts: a config with a high residual ε keeps exploring late in
training, which boosts the worst-case return floor but does not reflect
the *greedy* policy quality the user actually cares about.

**Fix.** Added an `objective_signal` knob to `DQNWindyChasmObjective`
(`"tail_mean"` vs `"eval_return_start"`) and re-ran the sweep with the
direct greedy-eval signal. The second run found a config with
`eval_return_start = -12.20`, within one wind-stochastic rollout's
variance of value iteration's `V* = -14.998`. Both runs are kept in the
report so the proxy mismatch stays visible.

### N+2. EI did not separate from LHS at this trial budget

**Problem.** In both BO sweeps the best-by-objective trial came from
the four initial Latin-Hypercube points, not from the twelve EI
iterations. The running-best curve is flat after the LHS phase.

**Cause.** DQN training-seed noise at 250 episodes per trial swamps EI's
signal at 16 trials in a 4-D search space — exactly the regime where
random search competes with BO (Bergstra & Bengio 2012).

**Fix planned, not executed.** Either (a) lift trials per config to
multiple seeds and use the seed-mean as the BO objective, or (b) adopt
multi-fidelity BO (Klein et al. 2017). Both are documented in
`DECISIONS.md` §6 as next-week work. Cost-wise (a) is the cheapest
improvement and is the recommended next step.

### N+3. GP Cholesky breakdown at small training sets

**Problem.** With ~4 LHS points the kernel matrix `K + σ_n² I` is small
and its conditioning depends on the heuristic length-scale. A near-
duplicate LHS sample once produced `LinAlgError` in `numpy.linalg.cholesky`.

**Fix.** `hpo/gp.py` retries the factorisation with progressively larger
jitter (R&W 2006, §5.4.2): start at the configured `noise_var`, double
the jitter each retry, cap at six tries. The second attempt always
succeeds in practice.

### N+4. No `scipy.stats.norm` → hand-rolled standard-normal Φ / φ

**Problem.** Expected Improvement needs `Φ(z)` and `φ(z)`. Adding
`scipy.stats` for one function re-breaks the no-extra-deps story
established in V2 (the NumPy DQN explicitly avoided scipy and torch).

**Fix.** `hpo/optimizer.py` uses `math.erf` (stdlib) for the CDF and
computes `φ(z) = e^{-z²/2} / √(2π)` directly. A unit test asserts
`EI = 0` when `σ = 0` and `EI > 0` strictly above the incumbent.
