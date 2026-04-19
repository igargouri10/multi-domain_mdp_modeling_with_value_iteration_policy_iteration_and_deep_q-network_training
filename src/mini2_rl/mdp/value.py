r"""Value-convergence solvers for MRPs (Mini-1, Task 3).

We provide three methods that all compute the state-value function

.. math::
   V = \mathcal{R} + \gamma P V,
   \qquad \Longrightarrow \qquad V = (I - \gamma P)^{-1} \mathcal{R},

for a finite MRP. Their trade-offs are discussed at length in the report,
but briefly:

* :func:`solve_value_matrix` -- closed-form via a linear solve. Costs
  :math:`\mathcal{O}(|S|^3)` and :math:`\mathcal{O}(|S|^2)` memory. The
  constant factor is small because ``numpy.linalg.solve`` dispatches to LAPACK
  ``dgesv``. Handles :math:`\gamma < 1` cleanly; fails gracefully for
  :math:`\gamma = 1` when :math:`I - P` is singular.
* :func:`solve_value_iterative` -- fixed-point iteration of the Bellman
  expectation operator, :math:`V_{k+1} = \mathcal{R} + \gamma P V_k`. Costs
  :math:`\mathcal{O}(K|S|^2)` for ``K`` iterations, but scales to sparse
  ``P`` because each iteration is a single mat-vec. Guaranteed to converge
  for :math:`\gamma < 1` because :math:`T` is a :math:`\gamma`-contraction in
  the sup-norm (Banach fixed-point theorem; see Sutton & Barto 2018, §4.1).
* :func:`solve_value_eigendecomp` -- an educational alternative using the
  spectral decomposition of :math:`P`. It is *not* recommended in practice
  because it is unstable for defective / non-diagonalizable :math:`P` (the
  classic degenerate case :math:`P = \begin{pmatrix}0 & 1\\ 1 & 0\end{pmatrix}`
  is only *periodic*, but near-repeated eigenvalues still produce ill-conditioned
  eigenvector matrices). The function detects near-singular eigenbases and
  falls back to the direct solver.

References
----------
Howard (1960), *Dynamic Programming and Markov Processes*; Puterman (1994),
*Markov Decision Processes*; Sutton & Barto (2018), Ch. 4.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .base import MarkovRewardProcess


__all__ = [
    "solve_value_matrix",
    "solve_value_iterative",
    "solve_value_eigendecomp",
]


# ---------------------------------------------------------------------------
def solve_value_matrix(mrp: MarkovRewardProcess) -> np.ndarray:
    r"""Closed-form :math:`V = (I - \gamma P)^{-1} \mathcal{R}`.

    Raises :class:`numpy.linalg.LinAlgError` if :math:`I - \gamma P` is
    singular, which happens iff :math:`\gamma = 1` and :math:`P` has an
    eigenvalue of 1 (i.e. the chain is recurrent and has no discounting).
    In that case :func:`solve_value_iterative` with a termination absorbing
    state, or an average-reward formulation, should be used instead.
    """
    n = mrp.num_states
    A = np.eye(n) - mrp.gamma * mrp.P
    return np.linalg.solve(A, mrp.R)


# ---------------------------------------------------------------------------
def solve_value_iterative(
    mrp: MarkovRewardProcess,
    *,
    tol: float = 1e-10,
    max_iter: int = 10_000,
    V0: np.ndarray | None = None,
    return_history: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    r"""Iterate :math:`V_{k+1} = \mathcal{R} + \gamma P V_k` to a fixed point.

    Parameters
    ----------
    tol : float
        Sup-norm tolerance for declaring convergence.
    max_iter : int
        Safety cap on the number of iterations.
    V0 : np.ndarray or None
        Optional initial value estimate (defaults to zeros).
    return_history : bool
        If ``True``, also return a ``(K+1, |S|)`` history for plotting.

    Notes
    -----
    For :math:`\gamma < 1` the Bellman operator is a sup-norm contraction of
    modulus :math:`\gamma`, so the iterates satisfy
    :math:`\|V_k - V^*\|_\infty \le \gamma^k \|V_0 - V^*\|_\infty`.
    In particular, a tolerance of ``tol`` is hit in
    :math:`\lceil \log(\text{tol}/\|V_0-V^*\|_\infty) / \log \gamma \rceil`
    iterations.
    """
    V = np.zeros(mrp.num_states) if V0 is None else np.array(V0, dtype=float)
    history = [V.copy()] if return_history else None
    for _ in range(max_iter):
        V_new = mrp.bellman_operator(V)
        delta = np.max(np.abs(V_new - V))
        V = V_new
        if return_history:
            history.append(V.copy())
        if delta < tol:
            break
    if return_history:
        return V, np.asarray(history)
    return V


# ---------------------------------------------------------------------------
def solve_value_eigendecomp(
    mrp: MarkovRewardProcess, *, cond_thresh: float = 1e10
) -> np.ndarray:
    r"""Spectral solver using :math:`P = Q \Lambda Q^{-1}`.

    If :math:`P` is diagonalizable with eigenvectors ``Q`` and eigenvalues
    :math:`\lambda_i`, then

    .. math::
       V = Q \, \text{diag}\!\Big(\tfrac{1}{1 - \gamma \lambda_i}\Big) Q^{-1} \mathcal{R}.

    The method has three failure modes we actively guard against:

    1. **Non-diagonalizable (defective) matrices**. ``numpy.linalg.eig``
       returns a non-orthogonal basis that may fail to span the full
       state space; we detect this through the condition number of ``Q``.
    2. **Periodic chains** such as :math:`\binom{0\;1}{1\;0}` (the classic
       degenerate example from the assignment). Eigenvalues :math:`\{+1, -1\}`
       are well-defined but the corresponding eigenvectors involve complex
       arithmetic for intermediate sums; we work in ``complex128`` and
       take the real part of the final answer.
    3. **Exact resonance** :math:`\gamma \lambda_i = 1` which makes
       :math:`1/(1 - \gamma \lambda_i)` blow up. We raise an explicit error.

    If the condition number of the eigenbasis exceeds ``cond_thresh`` we fall
    back to :func:`solve_value_matrix`, preserving numerical safety.
    """
    eigvals, Q = np.linalg.eig(mrp.P.astype(complex))
    try:
        Qinv = np.linalg.inv(Q)
    except np.linalg.LinAlgError:
        return solve_value_matrix(mrp)
    cond = np.linalg.cond(Q)
    if not np.isfinite(cond) or cond > cond_thresh:
        # degenerate / near-defective -- fall back
        return solve_value_matrix(mrp)
    denom = 1.0 - mrp.gamma * eigvals
    if np.any(np.abs(denom) < 1e-12):
        raise np.linalg.LinAlgError(
            "gamma * eigenvalue == 1; spectral solver is singular"
        )
    D = np.diag(1.0 / denom)
    V_complex = Q @ D @ Qinv @ mrp.R.astype(complex)
    # The imaginary part should be zero up to roundoff for a real MRP.
    if np.max(np.abs(V_complex.imag)) > 1e-6:
        # non-negligible imaginary part -> fall back
        return solve_value_matrix(mrp)
    return V_complex.real
