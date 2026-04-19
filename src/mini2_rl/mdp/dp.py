r"""Dynamic-programming solvers over a finite MDP.

Functional wrappers over the Bellman expectation / optimality operators. The
Mini-2 preliminary explicitly asks for *policy iteration* and *value iteration*
to be part of the core class hierarchy, so we expose both here as pure
functions that take an :class:`MarkovDecisionProcess` and return the pair
``(V*, pi*)``.

The algorithms and convergence statements follow R. A. Howard,
*Dynamic Programming and Markov Processes* (1960), M. L. Puterman,
*Markov Decision Processes* (1994), D. P. Bertsekas and J. N. Tsitsiklis,
*Neuro-Dynamic Programming* (1996), and R. S. Sutton and A. G. Barto,
*Reinforcement Learning: An Introduction*, 2nd ed. (2018, §4.3–§4.4).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .base import MarkovDecisionProcess


__all__ = [
    "q_from_v",
    "greedy_policy_from_q",
    "policy_iteration",
    "value_iteration",
]


def q_from_v(mdp: MarkovDecisionProcess, V: np.ndarray) -> np.ndarray:
    r"""One-step lookahead :math:`Q(s, a) = R^a_s + \gamma \sum_{s'} P^a_{ss'} V(s')`."""
    return mdp.R + mdp.gamma * (mdp.P @ V)  # (|A|, |S|)


def greedy_policy_from_q(Q: np.ndarray) -> np.ndarray:
    r"""Return the deterministic greedy policy as a ``(|S|, |A|)`` one-hot matrix."""
    m, n = Q.shape  # (|A|, |S|)
    best = np.argmax(Q, axis=0)
    pi = np.zeros((n, m))
    pi[np.arange(n), best] = 1.0
    return pi


def policy_iteration(
    mdp: MarkovDecisionProcess,
    *,
    max_sweeps: int = 10_000,
) -> Tuple[np.ndarray, np.ndarray, int]:
    r"""Howard's policy iteration.

    Returns ``(V, pi, sweeps)`` where ``V`` is the optimal value vector,
    ``pi`` is a ``(|S|, |A|)`` one-hot greedy policy, and ``sweeps`` is the
    number of evaluation+improvement passes.
    """
    n, m = mdp.num_states, mdp.num_actions
    policy = np.full((n, m), 1.0 / m)
    V = np.zeros(n)
    for sweeps in range(1, max_sweeps + 1):
        # Policy evaluation: solve V^pi analytically via induced MRP.
        mrp = mdp.induced_mrp(policy)
        A = np.eye(n) - mrp.gamma * mrp.P
        V = np.linalg.solve(A, mrp.R)
        # Policy improvement.
        Q = q_from_v(mdp, V)
        new_policy = greedy_policy_from_q(Q)
        if np.array_equal(policy, new_policy):
            return V, policy, sweeps
        policy = new_policy
    return V, policy, sweeps


def value_iteration(
    mdp: MarkovDecisionProcess,
    *,
    tol: float = 1e-10,
    max_sweeps: int = 10_000,
) -> Tuple[np.ndarray, np.ndarray, int]:
    r"""Bellman value iteration.

    Iterates :math:`V_{k+1}(s) = \max_a \bigl(R^a_s + \gamma \sum_{s'} P^a_{ss'} V_k(s')\bigr)`
    until the sup-norm update drops below ``tol``. Returns
    ``(V, pi, sweeps)`` where ``pi`` is the greedy policy read off from ``V``.

    For :math:`\gamma < 1` the Bellman optimality operator is a sup-norm
    contraction with modulus :math:`\gamma`, so convergence is geometric at
    rate :math:`\gamma^k` (S. Banach, *Fund. Math.* 3, 1922).
    """
    n = mdp.num_states
    V = np.zeros(n)
    for sweep in range(1, max_sweeps + 1):
        Q = q_from_v(mdp, V)
        V_new = Q.max(axis=0)
        delta = np.max(np.abs(V_new - V))
        V = V_new
        if delta < tol:
            return V, greedy_policy_from_q(q_from_v(mdp, V)), sweep
    return V, greedy_policy_from_q(q_from_v(mdp, V)), sweep
