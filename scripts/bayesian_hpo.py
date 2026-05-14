r"""Run Bayesian hyperparameter optimization on the V2 NumPy DQN (Mini-2 V3).

Tunes lr (log-uniform), batch_size (int), epsilon_decay_steps (log-uniform),
and target_sync (int) on the Problem 1 Windy Chasm MDP. The objective is
selectable via --objective-signal: tail_mean (default; mean over the last
N training episodes) or eval_return_start (greedy-policy return from the
start state at the end of training).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from mini2_rl.hpo import (
    BayesianOptimizer, SearchSpace,
    LogUniformParam, IntParam,
    DQNWindyChasmObjective,
)
from mini2_rl.problems.windy_chasm import WindyChasmSpec


def build_space() -> SearchSpace:
    return SearchSpace(
        params=[
            LogUniformParam("lr",                  low=1e-4, high=5e-3),
            IntParam       ("batch_size",          low=16,   high=128),
            LogUniformParam("epsilon_decay_steps", low=2_000, high=20_000),
            IntParam       ("target_sync",         low=50,   high=500),
        ]
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-init", type=int, default=4)
    ap.add_argument("--n-iter", type=int, default=12)
    ap.add_argument("--episodes", type=int, default=250)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--tail-window", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--xi", type=float, default=0.01)
    ap.add_argument("--log", default=None)
    ap.add_argument("--fig", default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--objective-signal",
                    choices=["tail_mean", "eval_return_start"],
                    default="tail_mean")
    args = ap.parse_args()

    objective = DQNWindyChasmObjective(
        spec=WindyChasmSpec(),
        episodes=args.episodes,
        max_steps=args.max_steps,
        tail_window=args.tail_window,
        objective_signal=args.objective_signal,
    )

    space = build_space()
    bo = BayesianOptimizer(
        space=space,
        objective=objective,
        n_init=args.n_init,
        n_iter=args.n_iter,
        seed=args.seed,
        xi=args.xi,
        verbose=args.verbose,
    )
    bo.run()

    payload = {
        "version": "v3",
        "task": "windy_chasm",
        "algo": "dqn_numpy",
        "objective_signal": args.objective_signal,
        "n_init": args.n_init,
        "n_iter": args.n_iter,
        "episodes_per_trial": args.episodes,
        "max_steps_per_episode": args.max_steps,
        "tail_window": args.tail_window,
        "seed": args.seed,
        "xi": args.xi,
        "search_space": [
            {"name": p.name, "kind": type(p).__name__,
             "low": p.low, "high": p.high}
            for p in space.params
        ],
        "best_trial": bo.best_trial.to_dict() if bo.best_trial else None,
        "running_best": bo.running_best(),
        "trials": bo.history(),
    }
    if args.log:
        os.makedirs(os.path.dirname(os.path.abspath(args.log)) or ".", exist_ok=True)
        with open(args.log, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        print("[bayesian_hpo] wrote", args.log)
    else:
        json.dump(payload["best_trial"], sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")

    if args.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        running = bo.running_best()
        objs = [t.objective for t in bo.trials]
        x = np.arange(1, len(running) + 1)
        if args.objective_signal == "tail_mean":
            ylabel = "tail-mean (last %d ep) DQN return" % args.tail_window
        else:
            ylabel = "greedy-eval return from start"

        os.makedirs(os.path.dirname(os.path.abspath(args.fig)) or ".", exist_ok=True)
        fig, ax = plt.subplots(figsize=(6.0, 3.5))
        ax.plot(x, objs, "o", color="tab:gray", alpha=0.6, label="trial objective")
        ax.plot(x, running, "-", color="tab:blue", lw=2.0, label="running best")
        ax.axvline(args.n_init + 0.5, color="k", ls=":", lw=1.0,
                   label="end of LHS / start of EI")
        ax.set_xlabel("BO trial")
        ax.set_ylabel(ylabel)
        ax.set_title("V3 Bayesian HPO on NumPy-DQN / Windy Chasm (signal="
                     + args.objective_signal + ")")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.fig, bbox_inches="tight")
        plt.close(fig)
        print("[bayesian_hpo] wrote", args.fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
