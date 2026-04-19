"""Abstract base classes for finite Markov processes (Mini-1, Task 2).

We use the "coding choice" of a sentinel value to represent walls/blocked cells
(Mini-1 task 4 asks for a coding choice for -infinity). We use ``numpy.nan``
internally for the mask, and a boolean ``blocked`` mask for the structural
information, because:

* ``-np.inf`` in reward entries propagates correctly in value computations if
  we never place it on a reachable state; Mini-1 says to translate masks into
  ``S`` and ``R``, so walls should simply be *removed* from the state space.
* Using NaN in masks makes the "not a cell" condition explicit and survives
  arithmetic only where we actively check for it.

References
----------
Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed., 2018
(Ch. 3: Finite MDPs). Bellman, "A Markovian Decision Process," 1957.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Callable, Hashable, Iterable, List, Optional, Sequence, Tuple

import numpy as np


TERMINAL_SENTINEL: object = object()
"""Sentinel object used in transition dictionaries to mark the absorbing state."""


# ---------------------------------------------------------------------------
#  MarkovProcess  (plain Markov chain -- no rewards, no actions)
# ---------------------------------------------------------------------------
class MarkovProcess(abc.ABC):
    """Finite Markov process :math:`(\\mathcal{S}, P)`.

    Stores a list of hashable states and a row-stochastic transition matrix
    :math:`P \\in \\mathbb{R}^{|\\mathcal{S}| \\times |\\mathcal{S}|}`.

    The storage choice is an explicit dense NumPy array. For the gridworld
    sizes used in Mini-1 (state spaces in the tens to low hundreds) this is
    substantially faster than a dict-of-dict sparse representation, and it
    lets us reuse optimized BLAS routines for the matrix-form Bellman solver.
    A sparse-matrix subclass can be swapped in without changing the interface.
    """

    def __init__(
        self,
        states: Sequence[Hashable],
        transition_matrix: np.ndarray,
        *,
        terminal_states: Optional[Iterable[Hashable]] = None,
        name: str = "MarkovProcess",
    ) -> None:
        states = list(states)
        P = np.asarray(transition_matrix, dtype=float)
        n = len(states)
        if P.shape != (n, n):
            raise ValueError(
                f"Transition matrix shape {P.shape} does not match |S|={n}"
            )
        if np.any(P < -1e-12):
            raise ValueError("Transition matrix contains negative entries")
        row_sums = P.sum(axis=1)
        # Absorbing / terminal rows: allow either a self-loop row or a zero row
        # (which encodes "process terminates"). Everything else must sum to 1.
        bad = []
        for i, r in enumerate(row_sums):
            if not (np.isclose(r, 1.0, atol=1e-8) or np.isclose(r, 0.0, atol=1e-8)):
                bad.append(i)
        if bad:
            raise ValueError(
                f"Rows {bad} of P are not row-stochastic (sums={row_sums[bad]})."
            )
        self._states: List[Hashable] = states
        self._state_index = {s: i for i, s in enumerate(states)}
        self._P = P
        self._terminal = set(terminal_states or ())
        for t in self._terminal:
            if t not in self._state_index:
                raise KeyError(f"Terminal state {t!r} not in state list")
        self.name = name

    # -- properties ---------------------------------------------------------
    @property
    def states(self) -> List[Hashable]:
        return list(self._states)

    @property
    def num_states(self) -> int:
        return len(self._states)

    @property
    def P(self) -> np.ndarray:
        """Transition matrix :math:`P_{ss'} = \\Pr(S_{t+1}=s' \\mid S_t=s)`."""
        return self._P

    def is_terminal(self, s: Hashable) -> bool:
        return s in self._terminal

    def index(self, s: Hashable) -> int:
        return self._state_index[s]

    # -- sampling -----------------------------------------------------------
    def step(
        self, s: Hashable, rng: Optional[np.random.Generator] = None
    ) -> Hashable:
        """Sample a successor state from ``s``."""
        if rng is None:
            rng = np.random.default_rng()
        probs = self._P[self.index(s)]
        if probs.sum() == 0.0:  # absorbing-zero row -> stay / terminate
            return s
        return self._states[rng.choice(len(self._states), p=probs)]

    # -- lightweight repr ---------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.name} |S|={self.num_states}>"


# ---------------------------------------------------------------------------
#  MarkovRewardProcess  (adds a reward vector R)
# ---------------------------------------------------------------------------
class MarkovRewardProcess(MarkovProcess):
    """Finite MRP :math:`(\\mathcal{S}, P, \\mathcal{R}, \\gamma)`.

    Rewards are stored as an expected reward *vector* :math:`\\mathcal{R}(s)`
    (length ``|S|``) because the Bellman expectation equation
    :math:`V = \\mathcal{R} + \\gamma P V` only requires the state-indexed
    expected reward (Sutton & Barto 2018, Eq. 3.13). If state-pair rewards
    :math:`r(s, s')` are supplied, we collapse them into
    :math:`\\mathcal{R}(s) = \\sum_{s'} P_{ss'} r(s, s')` at construction time.
    """

    def __init__(
        self,
        states: Sequence[Hashable],
        transition_matrix: np.ndarray,
        rewards: np.ndarray,
        *,
        gamma: float = 0.9,
        terminal_states: Optional[Iterable[Hashable]] = None,
        name: str = "MarkovRewardProcess",
    ) -> None:
        super().__init__(
            states, transition_matrix,
            terminal_states=terminal_states, name=name,
        )
        R = np.asarray(rewards, dtype=float)
        if R.ndim == 2:
            # state-pair reward r(s, s') -> collapse to expected R(s)
            if R.shape != self._P.shape:
                raise ValueError(
                    f"Pair-reward shape {R.shape} != P shape {self._P.shape}"
                )
            R = np.einsum("ij,ij->i", self._P, R)
        elif R.shape != (self.num_states,):
            raise ValueError(
                f"Reward shape {R.shape} != (|S|,) = ({self.num_states},)"
            )
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"Discount gamma={gamma} must lie in [0, 1]")
        self._R = R
        self.gamma = float(gamma)

    @property
    def R(self) -> np.ndarray:
        return self._R

    # Convenience -----------------------------------------------------------
    def bellman_operator(self, V: np.ndarray) -> np.ndarray:
        r"""Apply the Bellman expectation operator :math:`T V = \mathcal{R} + \gamma P V`."""
        return self._R + self.gamma * (self._P @ V)


# ---------------------------------------------------------------------------
#  MarkovDecisionProcess  (Mini-1 Task 6: actions)
# ---------------------------------------------------------------------------
class MarkovDecisionProcess(abc.ABC):
    """Finite MDP :math:`(\\mathcal{S}, \\mathcal{A}, P, \\mathcal{R}, \\gamma)`.

    Storage convention: ``P[a]`` is the (|S|, |S|) transition matrix under
    action ``a``, and ``R[a]`` is the expected-reward vector (|S|,) under
    action ``a``. We deliberately store per-action matrices rather than a
    single ``(|S|, |A|, |S|)`` tensor so that:

    * Actions available from different states can vary (set unsupported
      actions to a self-loop with zero reward).
    * We can share common tiles between actions (e.g. two actions with the
      same transition model differ only in ``R``).
    * Sparse per-action matrices can be used without reshaping.
    """

    def __init__(
        self,
        states: Sequence[Hashable],
        actions: Sequence[Hashable],
        transitions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        *,
        gamma: float = 0.9,
        terminal_states: Optional[Iterable[Hashable]] = None,
        name: str = "MarkovDecisionProcess",
    ) -> None:
        self._states = list(states)
        self._actions = list(actions)
        self._state_index = {s: i for i, s in enumerate(self._states)}
        self._action_index = {a: i for i, a in enumerate(self._actions)}
        n, m = len(self._states), len(self._actions)
        if len(transitions) != m or len(rewards) != m:
            raise ValueError("transitions / rewards must have |A| entries")
        P = np.stack([np.asarray(t, dtype=float) for t in transitions], axis=0)
        R = np.stack([np.asarray(r, dtype=float) for r in rewards], axis=0)
        if P.shape != (m, n, n):
            raise ValueError(f"P stack shape {P.shape} != (|A|,|S|,|S|) = {(m,n,n)}")
        if R.shape != (m, n):
            # allow per-action state-pair rewards
            if R.shape == (m, n, n):
                R = np.einsum("aij,aij->ai", P, R)
            else:
                raise ValueError(f"R stack shape {R.shape} != (|A|,|S|)")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"Discount gamma={gamma} must lie in [0, 1]")
        self._P = P
        self._R = R
        self.gamma = float(gamma)
        self._terminal = set(terminal_states or ())
        self.name = name

    # -- properties ---------------------------------------------------------
    @property
    def states(self) -> List[Hashable]:
        return list(self._states)

    @property
    def actions(self) -> List[Hashable]:
        return list(self._actions)

    @property
    def num_states(self) -> int:
        return len(self._states)

    @property
    def num_actions(self) -> int:
        return len(self._actions)

    @property
    def P(self) -> np.ndarray:
        """Transition tensor, ``P[a, s, s'] = Pr(s' | s, a)``."""
        return self._P

    @property
    def R(self) -> np.ndarray:
        """Expected-reward matrix, ``R[a, s]``."""
        return self._R

    def is_terminal(self, s: Hashable) -> bool:
        return s in self._terminal

    def index(self, s: Hashable) -> int:
        return self._state_index[s]

    # -- induce an MRP from a policy ---------------------------------------
    def induced_mrp(
        self, policy: np.ndarray, *, name: Optional[str] = None
    ) -> MarkovRewardProcess:
        r"""Collapse an MDP into an MRP under a stochastic policy.

        ``policy`` has shape ``(|S|, |A|)`` with rows summing to 1. The
        induced transition matrix is :math:`P^{\pi}_{ss'} = \sum_a \pi(a|s) P_{ss'}^{a}`
        and the induced reward is :math:`\mathcal{R}^{\pi}(s) = \sum_a \pi(a|s) \mathcal{R}^a(s)`.
        """
        policy = np.asarray(policy, dtype=float)
        if policy.shape != (self.num_states, self.num_actions):
            raise ValueError(
                f"policy shape {policy.shape} != (|S|,|A|)={self.num_states, self.num_actions}"
            )
        if not np.allclose(policy.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("Policy rows must sum to 1")
        # P_pi[s, s'] = sum_a pi(a|s) P[a, s, s']
        P_pi = np.einsum("sa,asn->sn", policy, self._P)
        R_pi = np.einsum("sa,as->s", policy, self._R)
        return MarkovRewardProcess(
            self._states, P_pi, R_pi,
            gamma=self.gamma, terminal_states=self._terminal,
            name=name or f"{self.name}|pi",
        )

    # -- sampling -----------------------------------------------------------
    def step(
        self,
        s: Hashable,
        a: Hashable,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[Hashable, float]:
        if rng is None:
            rng = np.random.default_rng()
        si, ai = self._state_index[s], self._action_index[a]
        probs = self._P[ai, si]
        if probs.sum() == 0.0:
            s_next = s
        else:
            s_next = self._states[rng.choice(len(self._states), p=probs)]
        r = float(self._R[ai, si])
        return s_next, r

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.name} |S|={self.num_states} |A|={self.num_actions}>"
