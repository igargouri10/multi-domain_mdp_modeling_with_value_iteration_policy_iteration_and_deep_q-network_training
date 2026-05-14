r"""Bayesian hyperparameter optimization for mini2_rl (V3).

This subpackage adds a Gaussian-process Bayesian-optimization (BO) loop
that tunes the pure-NumPy DQN from V2 on the Windy Chasm MDP from
Problem 1. The V3 prompt singles out Bayesian HPO as the appropriate
post-V2 tool for a single-protagonist environment; the Mini-2 problems
are all single-protagonist (Windy Chasm, Lidar cone, circuit design),
so this is where the V3 effort goes. See ``DECISIONS.md`` for the full
implemented-vs-skipped justification.

Submodules
----------
``space``       Search-space primitives (uniform / log-uniform / int).
``gp``          Pure-NumPy Gaussian-process regression with the RBF kernel.
``optimizer``   The BO loop with Expected-Improvement acquisition.
``objectives``  Wrappers that turn a DQN training run into a scalar
                BO objective.

References
----------
* J. Mockus, *Bayesian Approach to Global Optimization*, Kluwer, 1989.
* J. Snoek, H. Larochelle, and R. P. Adams, "Practical Bayesian
  optimization of machine learning algorithms," NeurIPS, 2012.
* P. I. Frazier, "A tutorial on Bayesian optimization," arXiv:1807.02811,
  2018.
* C. E. Rasmussen and C. K. I. Williams, *Gaussian Processes for Machine
  Learning*, MIT Press, 2006.
"""

from .space import SearchSpace, UniformParam, LogUniformParam, IntParam
from .gp import GaussianProcess, rbf_kernel
from .optimizer import BayesianOptimizer, BOTrial, expected_improvement
from .objectives import DQNWindyChasmObjective

__all__ = [
    "SearchSpace",
    "UniformParam",
    "LogUniformParam",
    "IntParam",
    "GaussianProcess",
    "rbf_kernel",
    "BayesianOptimizer",
    "BOTrial",
    "expected_improvement",
    "DQNWindyChasmObjective",
]
