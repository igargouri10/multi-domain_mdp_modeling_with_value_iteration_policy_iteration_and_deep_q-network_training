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
on Problem 1.

## Layout

```
mini2_rl_repo/
├── src/mini2_rl/
│   ├── mdp/                # abstract classes + DP solvers
│   ├── problems/           # Windy Chasm, lidar cone, circuit design
│   ├── agents/             # numpy DQN (+ optional torch DQN)
│   ├── buffers/            # replay buffer
│   ├── nn/                 # MLP architectures (torch path)
│   └── experiments/        # end-to-end figure/JSON driver
├── docs/
│   ├── report/             # LaTeX source + compiled PDF
│   ├── notes/              # project log
│   └── technical-challenges.md
├── data/                   # results.json (numerical outputs)
├── tests/                  # (add pytest smoke tests here)
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
