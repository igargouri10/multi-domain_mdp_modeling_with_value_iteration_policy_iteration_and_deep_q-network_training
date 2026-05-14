# V3 Decisions

*Mini-2 repository, EECS 590 Sp26. This file is the artifact V3 grades
under "Final Content": it lists every post-V2 class topic and records,
for each one, whether I implemented it, why, and where to find the
implementation. The intent is transparency, not justification by
volume — V3 explicitly penalizes "implements everything" as much as
"implements nothing."*

## TL;DR

For Mini-2's foundation environment — three single-protagonist, fully
observed, finite MDPs (Windy Chasm, Lidar cone, Circuit design) — I
implemented **one** V3 addition:

* **Bayesian hyperparameter optimization** (GP + Expected Improvement)
  applied to the V2 pure-NumPy DQN on Problem 1 (Windy Chasm).

I deliberately did **not** implement, on the same grounds the V3
prompt already flags:

* POMDP wrapper for Windy Chasm and recurrent Q-networks.
* Multi-agent / swarm coordination.
* A third BO sweep on Problems 2 (Lidar) or 3 (Circuit Design).

The V2 deliverables (Windy Chasm MDP + variants, Lidar shaping, Circuit
design, NumPy DQN, replay buffer, technical-challenges.md, project
log, written report) are kept as-is. The cleanup pass below brings the
repo's structural readiness up to V3's bar.

---

## 1. Implemented: Bayesian hyperparameter optimization on the NumPy DQN

**Code.** `src/mini2_rl/hpo/` (already scaffolded before this turn;
re-used with one addition: an `objective_signal` knob on
`DQNWindyChasmObjective`). CLI entry point at
`scripts/bayesian_hpo.py`. Unit tests in `tests/test_hpo.py` (9 new
assertions covering the search space, GP, EI, BO-on-quadratic, and a
DQN-objective smoke test). Trial logs in `data/hpo/`. Report figures
in `docs/report/figures/fig_v3_bo_*.pdf`.

**Why this is the right V3 addition.** The V3 prompt itself states:
*"An environment centered on a main protagonist does not necessarily
benefit from swarm/coordination implementations, but very much
benefits from Bayesian hyperparameter tuning."* All three Mini-2
problems are single-protagonist, so this is the post-V2 topic where a
careful implementation produces a *qualitative*, not just cosmetic,
improvement. The V2 DQN baseline in particular reaches a tail-window
mean of ~−24 on Windy Chasm against value iteration's −14.998 — a gap
that is partly a hyperparameter problem.

**Empirical findings (and the honest negative result).** I ran two BO
sweeps (n_init=4 LHS + n_iter=12 EI = 16 trials each; 250 training
episodes per trial; same seed; same search space) that differ only in
the scalar BO maximizes:

| Run | Objective signal | Best objective | Best greedy eval at start | Best params (lr / batch / ε-decay / target-sync) |
|---|---|---|---|---|
| 1 | tail_mean (last 50 ep) | −44.69 | −63.28 (poor!) | 1.86e-4 / 101 / 4154 / 193 |
| 2 | eval_return_start (greedy from start) | −12.20 | −12.20 | 1.91e-3 / 51 / 11354 / 121 |

Two important observations from these tables.

*First, the proxy mismatch.* The `tail_mean` and `eval_return_start`
rankings disagree substantially. The trial that maximized `tail_mean`
had `eval_return_start = −63.28`, well below the −24 the V2 hand-tuned
run achieves. The trial that maximized `eval_return_start` (−12.20)
beats both the V2 hand-tune *and* the value-iteration baseline of
−14.998 (one lucky rollout, but optimum-class). This is exactly the
*proxy-misspecification* warning that practical BO papers emphasize
(Snoek et al. 2012 §3.3). I am keeping both runs in the report so the
finding is visible rather than hidden.

*Second, EI did not beat the LHS on either run.* The best-by-objective
trial in both runs is one of the four initial LHS points, and the
twelve EI iterations did not improve on it. There are two plausible
causes — DQN training-seed noise dominates the EI signal at 250
episodes per trial; and 16 trials in a 4-D search space is in the
"random search competes with BO" regime that Bergstra & Bengio (2012)
characterize. Lifting the per-trial budget (or running multiple seeds
per trial) is logged below as the obvious next step.

**Design choices, with rejected alternatives.** Same as Mini-1's V3
work, ported here for completeness:

| Choice | What I did | Why I rejected the alternative |
|---|---|---|
| Surrogate model | GP with RBF kernel and zero-mean prior (Rasmussen & Williams 2006). | TPE (Bergstra et al. 2011) is simpler but loses the smooth posterior EI uses; random-forest surrogates need a vectorized-tree implementation. |
| Acquisition fn | Expected Improvement (Mockus 1989) with $\xi=0.01$ exploration bias. | UCB needs a $\beta$ schedule the user has to tune by hand — defeats the purpose of HPO. |
| Kernel hyperparams | Median-pairwise-distance length scale; signal variance = Var(y); tiny noise jitter. | Type-II MLE on the kernel is unreliable at $n \le 25$ observations (Snoek et al. 2012 §3) and would pull in scipy. |
| Acquisition optim. | Random search of 4–8 K unit-cube candidates. | L-BFGS-B on EI would pull in scipy; in 4-D random restarts are within $10^{-3}$ of the boundary. |
| Inner objective | Pure-NumPy DQN, 250 episodes / 60 max steps per trial. | Torch DQN gives the same asymptotic policy at the same (or higher) wall-clock cost; using torch would also have re-broken the no-extra-deps story V2 established. |

---

## 2. Skipped: POMDP wrapper for Windy Chasm + recurrent Q-network

**Why skipped.** Windy Chasm is **fully observed** by construction —
the agent sees its own (x, y) and the wind is a known stochastic
function of the column. Wrapping it as a POMDP would either (a)
artificially mask the wind, which makes the problem strictly harder
without showing off any new RL idea, or (b) leave the observation
identical to the state, which is a no-op.

**What I would have done.** A `BeliefMDP` wrapper that maintains a
categorical posterior over a hidden state component (e.g. the wind
strength `B`), plus a small GRU-conditioned Q-network. Kaelbling
et al. (1998) is the canonical reference and is already cited from
Problem 2's discussion of reward shaping; adding it on the V3 side
without a POMDP problem would not help the reader.

---

## 3. Skipped: BO sweep on Problems 2 (Lidar) and 3 (Circuit Design)

**Why skipped.** Each additional problem multiplies the V3 compute
budget by ~16 trials of an inner training loop. Problem 2 (Lidar)
has only one obviously interesting BO knob (`lambda_shape`), and
Problem 3 (Circuit Design) has discrete component-placement actions
that don't expose a continuous hyperparameter that a smooth GP
surrogate would help with. Spending the V3 budget on these would
*dilute* the headline finding rather than strengthen it.

**Pointer.** The framework is generic: writing a
`LidarShapingObjective` or `CircuitGreedyObjective` that satisfies the
`(params) -> (scalar, metrics)` contract is ~30 LoC. Logged as future
work below.

---

## 4. Skipped: multi-agent / swarm coordination

**Why skipped.** V3 says verbatim that "an environment centered on a
main protagonist does not necessarily benefit from swarm/coordination
implementations." All three Mini-2 problems are single-protagonist.
Building MADDPG or QMIX on Windy Chasm would require inventing a
contrived multi-agent task and the resulting algorithm choice would
be unrelated to anything else in the repo. Taking V3 at its word.

---

## 5. Structural-readiness changes

* Removed empty `notebooks/` scaffold, `__pycache__/` artifacts, and
  stale LaTeX intermediates (`.aux`, `.bbl`, `.blg`, `.log`, `.out`).
  All four classes are still in `.gitignore`.
* Added `.gitkeep` to `checkpoints/`, `data/`, and `data/hpo/`.
* Created the missing `scripts/` directory with the V3 BO CLI.
* Added `DECISIONS.md` (this file).
* Added `tests/test_hpo.py` with 9 V3 unit tests.
* Added BO references (Mockus 1989, Snoek et al. 2012, Frazier 2018,
  Rasmussen-Williams 2006, Bergstra-Bengio 2012, Bergstra et al. 2011,
  McKay-Beckman-Conover 1979, Garreau et al. 2018) to
  `docs/report/references.bib`.
* Extended `docs/technical-challenges.md` with the engineering issues
  hit during the V3 BO experiment (notably the proxy-mismatch finding
  in §1 above).
* Extended `README.md` (new V3 section) and the LaTeX report (new V3
  section) so the new work is discoverable.

---

## 6. What I'd add if I had another week

In rough order of payoff:

1. **Multi-fidelity BO (Klein et al. 2017).** Train each inner DQN
   for a small budget early and a full budget late. This is the most
   direct fix for "EI didn't beat LHS at 16 trials."
2. **Proper seed-variance accounting.** Train each candidate config
   under K independent seeds and use the seed-mean as the BO
   objective. This collapses both the proxy mismatch and the
   seed-noise issue into a single (more honest) signal.
3. **Apply the BO framework to Problem 2's `lambda_shape` and
   Problem 3's greedy-policy hyperparameters** so the V3 finding
   generalises across all three Mini-2 problems.

---

*Author: Ismail Gargouri. No human collaborator on the V3 work; I
consulted Sutton & Barto (2018), Rasmussen & Williams (2006), Snoek
et al. (2012), Frazier (2018), and Bergstra & Bengio (2012). All
references appear in `docs/report/references.bib` and are cited in the
V3 section of `docs/report/report.pdf`.*
