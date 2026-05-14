r"""Bayesian-optimization loop with Expected-Improvement acquisition.

Algorithm (Snoek, Larochelle, and Adams, 2012, Section 3; Frazier, 2018,
Section 4):

1. Draw an initial design of ``n_init`` Latin-Hypercube points and
   evaluate the objective at each.
2. For ``n_iter`` iterations:
    a. Fit a Gaussian-process surrogate on the unit-cube design matrix.
    b. Choose the next sample by maximizing Expected Improvement.
    c. Evaluate the (noisy) objective and append to the design matrix.
3. Return all trials and the running best.

EI is maximized by a random search of unit-cube candidates (no scipy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .gp import GaussianProcess
from .space import SearchSpace


def expected_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    f_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    r"""Expected Improvement, maximization convention.

    .. math::
       \text{EI}(x) = (\mu(x) - f^* - \xi)\,\Phi(z) + \sigma(x)\,\phi(z),
       \quad z = \frac{\mu(x) - f^* - \xi}{\sigma(x)}.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    eps = 1e-12

    imp = mu - f_best - xi
    z = imp / (sigma + eps)

    phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    Phi = 0.5 * (1.0 + np.vectorize(math.erf)(z / np.sqrt(2.0)))

    ei = imp * Phi + sigma * phi
    ei[sigma < 1e-9] = 0.0
    return ei


@dataclass
class BOTrial:
    iteration: int
    source: str                 # "init" or "ei"
    params_unit: List[float]
    params_native: Dict[str, Any]
    objective: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    wall_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BayesianOptimizer:
    """Sequential GP-BO for noisy scalar objectives, maximization convention."""

    space: SearchSpace
    objective: Callable[[Dict[str, Any]], Any]
    n_init: int = 4
    n_iter: int = 12
    seed: int = 0
    xi: float = 0.01
    n_candidates_per_iter: int = 4096
    verbose: bool = False

    trials: List[BOTrial] = field(default_factory=list, init=False)
    best_trial: Optional[BOTrial] = field(default=None, init=False)
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def _evaluate(self, u: np.ndarray, source: str, iteration: int) -> BOTrial:
        import time

        native = self.space.to_native_dict(u)
        t0 = time.perf_counter()
        result = self.objective(native)
        wall = time.perf_counter() - t0

        if isinstance(result, tuple) and len(result) == 2:
            obj, metrics = result
            obj = float(obj)
            metrics = dict(metrics)
        else:
            obj = float(result)
            metrics = {}

        trial = BOTrial(
            iteration=iteration,
            source=source,
            params_unit=list(map(float, u.tolist())),
            params_native=native,
            objective=obj,
            metrics=metrics,
            wall_seconds=float(wall),
        )
        self.trials.append(trial)

        if self.best_trial is None or obj > self.best_trial.objective:
            self.best_trial = trial

        if self.verbose:
            best_so_far = self.best_trial.objective if self.best_trial else float("-inf")
            print(
                f"[BO iter {iteration:3d}] src={source:4s}  "
                f"obj={obj:+.4f}  best={best_so_far:+.4f}  "
                f"wall={wall:.1f}s  params={native}"
            )
        return trial

    def run(self) -> "BayesianOptimizer":
        U_init = self.space.sample_lhs(self.n_init, self.rng)
        for i in range(self.n_init):
            self._evaluate(U_init[i], source="init", iteration=i)

        for t in range(self.n_iter):
            U_obs = np.array([trial.params_unit for trial in self.trials], dtype=float)
            y_obs = np.array([trial.objective for trial in self.trials], dtype=float)

            length = GaussianProcess.heuristic_length_scale(U_obs)
            signal = max(float(np.var(y_obs)), 1e-3)
            gp = GaussianProcess(
                length_scale=length, signal_var=signal, noise_var=1e-4
            ).fit(U_obs, y_obs)

            U_cand = self.rng.random((self.n_candidates_per_iter, self.space.dim))
            mu, sigma = gp.predict(U_cand, return_std=True)
            ei = expected_improvement(mu, sigma, f_best=float(y_obs.max()), xi=self.xi)
            best_idx = int(np.argmax(ei))
            u_next = U_cand[best_idx]

            min_dist = float(np.min(np.linalg.norm(U_obs - u_next, axis=1)))
            if min_dist < 1e-3:
                u_next = np.clip(
                    u_next + self.rng.normal(scale=0.05, size=u_next.shape),
                    0.0, 1.0,
                )

            self._evaluate(u_next, source="ei", iteration=self.n_init + t)

        return self

    def history(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.trials]

    def running_best(self) -> List[float]:
        best = -np.inf
        out: List[float] = []
        for t in self.trials:
            best = max(best, t.objective)
            out.append(float(best))
        return out
