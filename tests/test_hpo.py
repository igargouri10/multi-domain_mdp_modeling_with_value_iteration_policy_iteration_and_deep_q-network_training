r"""Smoke + correctness tests for the V3 Bayesian-HPO subpackage (Mini-2).

The first eight tests are pure-NumPy and run in well under a second.
The last test trains a tiny DQN end-to-end (~one second on CPU) so we
catch regressions in the objective wrapper.
"""

from __future__ import annotations

import math
import numpy as np

from mini2_rl.hpo import (
    BayesianOptimizer, SearchSpace,
    UniformParam, LogUniformParam, IntParam,
    GaussianProcess, rbf_kernel, expected_improvement,
    DQNWindyChasmObjective,
)
from mini2_rl.problems.windy_chasm import WindyChasmSpec


# ---------------------------------------------------------------------------
def test_uniform_param_roundtrip() -> None:
    p = UniformParam("x", 0.0, 10.0)
    for u in [0.0, 0.25, 0.5, 1.0]:
        x = p.to_native(u)
        u2 = p.to_unit(x)
        assert abs(u - u2) < 1e-9, (u, x, u2)


def test_log_uniform_param_roundtrip() -> None:
    p = LogUniformParam("lr", 1e-4, 1.0)
    for u in [0.0, 0.5, 1.0]:
        x = p.to_native(u)
        u2 = p.to_unit(x)
        assert abs(u - u2) < 1e-9, (u, x, u2)


def test_int_param_range() -> None:
    p = IntParam("batch_size", 16, 128)
    seen = sorted({p.to_native(u) for u in np.linspace(0, 1, 1024)})
    assert seen[0] == 16 and seen[-1] == 128, seen
    assert all(16 <= v <= 128 for v in seen)


def test_search_space_lhs_is_a_permutation() -> None:
    space = SearchSpace([
        UniformParam("a", 0, 1),
        UniformParam("b", 0, 1),
    ])
    rng = np.random.default_rng(0)
    U = space.sample_lhs(8, rng)
    for j in range(space.dim):
        strata = (U[:, j] * 8).astype(int)
        assert sorted(strata.tolist()) == list(range(8)), strata


# ---------------------------------------------------------------------------
def test_rbf_kernel_symmetry_and_diagonal() -> None:
    X = np.random.default_rng(0).random((4, 3))
    K = rbf_kernel(X, X, length_scale=0.5, signal_var=1.7)
    assert np.allclose(K, K.T)
    assert np.allclose(np.diag(K), 1.7)


def test_gp_interpolates_noise_free() -> None:
    rng = np.random.default_rng(1)
    X = rng.random((6, 2))
    y = np.sin(2 * np.pi * X[:, 0]) + 0.3 * X[:, 1]
    gp = GaussianProcess(length_scale=0.4, signal_var=1.0, noise_var=1e-8)
    gp.fit(X, y)
    mu, _ = gp.predict(X)
    assert np.allclose(mu, y, atol=1e-3)


def test_expected_improvement_zero_when_sigma_zero() -> None:
    mu = np.array([0.5, 1.5])
    sigma = np.array([0.0, 0.0])
    ei = expected_improvement(mu, sigma, f_best=1.0)
    assert np.allclose(ei, 0.0)


def test_bo_finds_optimum_of_quadratic() -> None:
    """BO should beat random search on a 2-D bowl with a small budget."""
    space = SearchSpace([UniformParam("x", 0, 1), UniformParam("y", 0, 1)])

    def obj(p):
        x, y = p["x"], p["y"]
        return -((x - 0.7) ** 2 + (y - 0.3) ** 2), {}

    bo = BayesianOptimizer(
        space=space, objective=obj,
        n_init=4, n_iter=12, seed=0, n_candidates_per_iter=2048,
    ).run()

    assert bo.best_trial is not None
    bx = bo.best_trial.params_native["x"]
    by = bo.best_trial.params_native["y"]
    dist = math.hypot(bx - 0.7, by - 0.3)
    assert dist < 0.12, (bx, by, dist)


def test_dqn_objective_smoke() -> None:
    """End-to-end: one tiny DQN trial via the objective wrapper."""
    obj = DQNWindyChasmObjective(
        spec=WindyChasmSpec(),
        episodes=20, max_steps=30, tail_window=10,
    )
    scalar, metrics = obj({
        "lr": 5e-4, "batch_size": 32,
        "epsilon_decay_steps": 5_000, "target_sync": 200,
    })
    assert math.isfinite(scalar)
    assert "tail_mean_return" in metrics
    assert "eval_return_start" in metrics
