# mini2_rl — EECS 590 Mini-Project 2 code and report

This repository contains the code, experiments, and LaTeX report for
Mini-Project 2 of EECS 590 (Advanced Reinforcement Learning, Spring 2026).
It augments the Mini-1 abstract Markov-process hierarchy with a finite
Markov Decision Process (MDP) class, and solves the three problems from
`Mini_2.pdf`:

1. **Windy Chasm** — a `20 × 7` stochastic gridworld with multiplicative
   wind; policy and value iteration; four challenge variants that refine
   the grid toward a continuous-space MDP.
2. **Lidar-cone corridor** — a reward-shaping demo in the sense of Ng,
   Harada, and Russell (ICML 1999).
3. **Circuit design via sequential decisions** — an agent places wires
   and NAND gates on a `3 × 3` grid to synthesise XOR; complexity analysis
   for the `4 × 4` case.

The V2 deliverable is a pure-NumPy DQN agent (Mnih *et al.*, 2015) trained
on Problem 1. The **V3** deliverable is a from-scratch Gaussian-process
Bayesian-optimization loop (Mockus 1989; Snoek *et al.*, 2012) that tunes
that DQN's hyperparameters on the Problem 1 MDP. See `DECISIONS.md` at
the repo root for the V3 implemented-vs-skipped justification.

## Layout

```
mini2_rl_repo/
├── src/mini2_rl/
│   ├── mdp/                # abstract classes + DP solvers
│   ├── problems/           # Windy Chasm, lidar cone, circuit design
│   ├── agents/             # numpy DQN (+ optional torch DQN)
│   ├── buffers/            # replay buffer
│   ├── nn/                 # MLP architectures (torch path)
│   ├── hpo/                # V3: GP-BO with Expected Improvement
│   └── experiments/        # end-to-end figure/JSON driver
├── scripts/
│   └── bayesian_hpo.py     # V3: BO CLI for the NumPy DQN
├── docs/
│   ├── report/             # LaTeX source + compiled PDF
│   ├── notes/              # project log
│   └── technical-challenges.md
├── DECISIONS.md            # V3 transparency: what was/wasn't done
├── data/                   # results.json + V3 hpo/ trial logs
├── tests/                  # smoke + V3 BO unit tests
└── requirements.txt
```

## Install

```bash
pip install -r requirements.txt      # numpy, matplotlib
# optional: pip install -e .[drl]    # for the torch DQN
```

## Reproducing every figure in the report

```bash
cd mini2_rl_repo
PYTHONPATH=src python -m mini2_rl.experiments.run_all
```

This writes:

* `data/results.json` — all numerical results (VI/PI sweep counts, V*
  values, DQN learning curve, circuit-design complexity table).
* `docs/report/figures/*.pdf` — every figure that the LaTeX report
  consumes.

Then rebuild the report:

```bash
cd docs/report && pdflatex report.tex && pdflatex report.tex
```

## Key results

| Problem | `|S|` | VI sweeps | PI sweeps | `V*` (start) |
|---|---|---|---|---|
| Windy Chasm (`20×7`) | 97 | 95 | 3 | −14.998 |
| Windy Chasm fine (`60×21`) | 1123 | 116 | — | −4.17 |
| Windy Chasm continuous-x | 2263 | 33 | — | −33.19 |
| Lidar cone (`13×7`) | 37 | 15 | 4 | +26.83 |
| Circuit design subset | 391 | 7 | 3 | −20.77 |
| DQN on Windy Chasm | 97 | — | — | −24 (last 50 ep MA); VI target −14.998 |

Values come from a single run of `run_all.py`.

## V3: Bayesian hyperparameter optimization

V3 adds a from-scratch Gaussian-process Bayesian-optimization loop
(`src/mini2_rl/hpo/`) that tunes the V2 NumPy DQN on Problem 1. The
rationale, in one sentence: V3 flags Bayesian HPO as the right tool
for a single-protagonist environment, all three Mini-2 problems are
single-protagonist, so this is where the V3 effort goes. The full
reasoning, including the post-V2 topics I considered and rejected
(POMDPs, multi-agent, more BO sweeps), is in `DECISIONS.md`.

Two BO sweeps were run, differing only in the scalar BO maximizes:

| Run | Signal | Best objective | Best greedy-eval-at-start |
|---|---|---|---|
| 1 | `tail_mean` (last 50 train episodes) | −44.69 | −63.28 (poor!) |
| 2 | `eval_return_start` (greedy from start) | −12.20 | −12.20 (≈ optimal) |

The two rankings disagree — the "best" config under `tail_mean` has
a *worse* greedy policy than several other configs in the same sweep.
This proxy mismatch is the headline finding and is discussed in the
V3 section of `docs/report/report.pdf` and in `DECISIONS.md`.

Reproduce either run:

```bash
PYTHONPATH=src python scripts/bayesian_hpo.py \
    --n-init 4 --n-iter 12 --episodes 250 --max-steps 60 \
    --objective-signal eval_return_start \
    --log data/hpo/dqn_windy_chasm_evalstart.json \
    --fig docs/report/figures/fig_v3_bo_evalstart.pdf
```

References for the BO methodology (Mockus 1989; Snoek, Larochelle &
Adams 2012; Frazier 2018; Rasmussen & Williams 2006; Bergstra &
Bengio 2012; Bergstra et al. 2011) are in
`docs/report/references.bib`.

## Author and collaborations

* Author: Ismail Gargouri (`igargouri1@gmail.com`).
* No human collaborators on any part of this repository.
* No pre-existing RL frameworks (Stable-Baselines3, RLlib, CleanRL,
  Optuna, scikit-optimize) were used; the DQN and the BO loop are
  written from scratch.
* Third-party libraries: NumPy, Matplotlib, optionally PyTorch.
  Versions pinned in `requirements.txt`.
