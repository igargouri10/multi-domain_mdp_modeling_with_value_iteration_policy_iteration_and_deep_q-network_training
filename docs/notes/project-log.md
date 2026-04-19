# Project log — Mini-2

A chronological, decision-level log of the Mini-2 build. Entries are
dated; each block ends with the artefact it produced.

## 2026-04-18 — Scope alignment and scaffold
- Confirmed with stakeholder: implement Problem 1 CHALLENGE parts (a)–(d),
  discuss (e)–(f) in the report. Implement Problems 2 and 3 in full. DQN
  on Problem 1. New `mini2_rl` repo with the Mini-1 base classes vendored.
  References in IEEE format.
- Scaffolded the repo skeleton (`src/`, `docs/`, `data/`, `tests/`,
  `scripts/`, `setup.py`, `requirements.txt`).
- Artefact: empty repo with `MarkovProcess` / `MRP` / `MDP` copied from
  Mini-1 and a new `mdp/dp.py` with `policy_iteration` and
  `value_iteration`.

## 2026-04-18 — Problem 1 base: Windy Chasm MDP
- Implemented `WindyChasmSpec` + `WindyChasmMDP` in
  `src/mini2_rl/problems/windy_chasm.py`.
- State space: 95 interior cells + GOAL + CRASH = 97.
- Smoke test: VI 95 sweeps, PI 3 sweeps, `V*(start) = −14.998`, max
  VI/PI disagreement `9.47 × 10⁻¹¹`.
- Artefact: `windy_chasm.py`.

## 2026-04-18 — Problem 1 challenge variants
- Implemented four separate variants in `windy_chasm_variants.py`:
  - (a) fine grid 60×21 with `step_cost / subdiv`
  - (b) continuous-x with `dx=0.05` and `p(x) = B^E(x)` evaluated on the
    real x-coordinate
  - (c) velocity: fixed `v`, step cost scaled
  - (d) integral-wind formula: `(p(j) + p(j+vt))/2 · t + p(j)(1 − vt)`
- Results JSON records `|S|`, sweeps, and `V*(start)` for each.

## 2026-04-18 — Problem 2: Lidar cone
- Encoded the PDF's diamond corridor on a 13×7 grid (`|r − 6| + |c| ≤ 5`).
- Initial bug: goal at `(6, 4)` left `(6, 5)` as a dead-end crash sink;
  fixed to `(6, 5)` the narrow tip (`V*(start) = 26.83`).
- Added reward shaping `−λ |Δ(s)|` with `Δ = 2r − 12` and verified
  policy invariance across `λ ∈ [0, 1]`.

## 2026-04-18 — Problem 3: Circuit design
- Implemented `CircuitDesignSpec`, `CircuitSimulator`, `CircuitDesignMDP`
  in `problems/circuit_design.py`.
- Simulator handles the PDF's insulation between `(1, 1)` and `(2, 1)`.
- Tabular MDP reachable from the start state has 391 states under the
  default spec (2-component budget, 6-step horizon).
- VI 7 sweeps, PI 3 sweeps; `V*(start) = −20.77` (agent cannot build XOR
  with only 2 components, but the framework is verified).
- Added `state_space_size(H, W)` for the complexity analysis in the
  report.

## 2026-04-18 — DQN on Problem 1
- Sandbox cannot `pip install torch`; wrote `agents/dqn_numpy.py` with
  hand-rolled backprop for a two-hidden-layer MLP, Adam, and a uniform
  replay buffer. Target network synced every 500 grad steps.
- 3000 episodes on Windy Chasm, 60-step cap: returns improve from −64
  (episode 0) to −24 (100-ep MA at ep 3000). Greedy eval from start
  `= −0.01` on a single rollout (stochastic).
- Artefact: `dqn_learning_curve.pdf`.

## 2026-04-18 — Experiment driver and report
- Wrote `experiments/run_all.py` that emits `data/results.json` plus
  seven figures.
- Wrote `docs/report/report.tex` (10 pages) with IEEE-format
  `thebibliography` block and 20 references. Compiled under
  `pdflatex × 2` to resolve cross-references.
- Artefact: `docs/report/report.pdf` (10 pages, ~520 KB).
