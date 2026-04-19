r"""Problem 1 — Navigating a Windy Chasm.

Mini-2 Problem 1 in full. The drone moves on a grid of height ``H`` (``i``
direction) and width ``W`` (``j`` direction). By default ``H = 20`` and
``W = 7``. The drone starts at ``(0, ⌊W/2⌋) = (0, 3)``.

Dynamics
--------
Given the current non-terminal cell ``(i, j)`` and an action
``a ∈ {forward, left, right}``:

1.  The drone moves deterministically to ``(i', j')``:

        forward: (i+1, j)
        left:    (i,   j-1)
        right:   (i,   j+1)

2.  The wind acts on ``j'`` with a *p*-parameterised three-outcome
    distribution.  Let ``c = (W-1)/2`` be the centre column (``3`` for
    ``W = 7``) and

    .. math::
        E(j) = \frac{1}{1 + (j-c)^2}, \qquad p(j) = B^{E(j)},

    where ``B = p(c)`` is the baseline wind strength.  The wind pushes
    symmetrically by ±1 or ±2 columns:

    * with probability ``p(j')`` the wind shifts to ``j' ± 1`` (50/50);
    * with probability ``(1-p(j')) p(j')^2`` the wind shifts to ``j' ± 2``
      (50/50);
    * with the remaining probability ``(1-p(j'))(1 - p(j')^2)`` the drone
      stays at ``j'``.

    The three branches sum to one, as claimed in the assignment PDF, because
    ``p + (1-p) p^2 + (1-p)(1-p^2) = p + (1-p) = 1``.

3.  Terminal check on the final cell ``(i', j'')`` (before wind if the
    deterministic move itself lands on a terminal column; otherwise after
    wind):

        goal:  i' = H-1               reward +R   absorbing
        crash: j'' ∈ {0, W-1} (out)   reward -r   absorbing
        else:  step reward -1         transition to (i', j'')

We expose two APIs for this environment:

* :class:`WindyChasmSpec` — a lightweight dataclass capturing the
  hyperparameters (``H, W, B, R, r``) so we can run parameter sweeps.
* :class:`WindyChasmMDP` — a :class:`MarkovDecisionProcess` subclass with
  the full ``(|A|, |S|, |S|)`` transition stack, ready to plug into any of
  the value solvers or DP routines.

References to the design choices appear in ``docs/report/report.pdf`` §III.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Hashable, List, Tuple

import numpy as np

from ..mdp.base import MarkovDecisionProcess


__all__ = [
    "Action",
    "WindyChasmSpec",
    "WindyChasmMDP",
    "wind_distribution",
]


class Action(IntEnum):
    FORWARD = 0
    LEFT = 1
    RIGHT = 2


@dataclass
class WindyChasmSpec:
    """Hyperparameters of the windy-chasm environment."""

    H: int = 20
    W: int = 7
    B: float = 0.4        # baseline wind strength p(centre)
    R_goal: float = 50.0   # reward for reaching (H-1, n)
    r_crash: float = 50.0  # penalty for crashing (magnitude; sign applied at use)
    step_cost: float = 1.0  # magnitude; applied as -step_cost per step
    gamma: float = 0.95

    @property
    def centre(self) -> int:
        return (self.W - 1) // 2


# ---------------------------------------------------------------------------
def wind_distribution(j_mid: int, spec: WindyChasmSpec) -> Dict[int, float]:
    r"""Probability distribution over the final column after the wind acts.

    Given the post-action column ``j_mid`` (before the wind), returns a
    mapping ``{j_final: probability}``. The distribution is *symmetric* in
    ±1 and ±2 shifts, which matches the centre-line description in the PDF
    and extends it consistently to ``j_mid ≠ centre``.
    """
    c = spec.centre
    if j_mid == c:
        p = spec.B
    else:
        E = 1.0 / (1.0 + (j_mid - c) ** 2)
        p = spec.B ** E
    dist: Dict[int, float] = {}
    # ±1 shifts — probability p split 50/50
    dist[j_mid + 1] = dist.get(j_mid + 1, 0.0) + p / 2
    dist[j_mid - 1] = dist.get(j_mid - 1, 0.0) + p / 2
    # ±2 shifts — probability (1-p) p^2 split 50/50
    w2 = (1.0 - p) * (p ** 2)
    dist[j_mid + 2] = dist.get(j_mid + 2, 0.0) + w2 / 2
    dist[j_mid - 2] = dist.get(j_mid - 2, 0.0) + w2 / 2
    # stay probability
    dist[j_mid] = dist.get(j_mid, 0.0) + (1.0 - p) * (1.0 - p ** 2)
    return dist


# ---------------------------------------------------------------------------
class WindyChasmMDP(MarkovDecisionProcess):
    r"""Finite MDP realisation of the windy-chasm environment.

    State encoding
    --------------
    Non-terminal cells are pairs ``(i, j)`` with ``0 ≤ i ≤ H-2`` and
    ``1 ≤ j ≤ W-2``. We add two absorbing states, ``GOAL`` and ``CRASH``,
    that the goal row and the side columns funnel into.

    Actions
    -------
    ``Action.FORWARD`` (+1 to ``i``), ``Action.LEFT`` (-1 to ``j``),
    ``Action.RIGHT`` (+1 to ``j``).

    Reward kernel
    -------------
    Expected-reward-vector representation ``R[a, s]`` with rewards set to
    ``-step_cost`` for non-terminal transitions, ``+R_goal`` when the next
    state is ``GOAL`` (probability-weighted), and ``-r_crash`` when the
    next state is ``CRASH``. This matches the :math:`R(s, a) = \mathbb{E}
    [R_{t+1} \mid S_t = s, A_t = a]` convention used in the Mini-2
    preliminary.
    """

    GOAL: Tuple[str, str] = ("GOAL", "")
    CRASH: Tuple[str, str] = ("CRASH", "")

    def __init__(self, spec: WindyChasmSpec) -> None:
        self.spec = spec
        states: List[Hashable] = []
        # Non-terminal cells
        for i in range(spec.H - 1):          # i = 0 .. H-2
            for j in range(1, spec.W - 1):   # j = 1 .. W-2
                states.append((i, j))
        states.append(self.GOAL)
        states.append(self.CRASH)
        state_index = {s: k for k, s in enumerate(states)}
        self._state_index_public = state_index
        n = len(states)
        actions = list(Action)
        m = len(actions)
        P = np.zeros((m, n, n))
        R = np.zeros((m, n))
        goal_idx = state_index[self.GOAL]
        crash_idx = state_index[self.CRASH]
        # Absorbing self-loops for terminals
        for a_idx in range(m):
            P[a_idx, goal_idx, goal_idx] = 1.0
            P[a_idx, crash_idx, crash_idx] = 1.0

        # Build transitions for non-terminals
        for s, s_idx in state_index.items():
            if s in (self.GOAL, self.CRASH):
                continue
            i, j = s
            for a in actions:
                # deterministic move
                if a == Action.FORWARD:
                    ip, jp = i + 1, j
                elif a == Action.LEFT:
                    ip, jp = i, j - 1
                else:
                    ip, jp = i, j + 1
                ip = min(ip, spec.H - 1)  # clamp i to grid
                # check immediate crash/goal after the deterministic move
                if ip == spec.H - 1:
                    # goal row — any column counts; skip wind
                    P[a.value, s_idx, goal_idx] += 1.0
                    R[a.value, s_idx] += spec.R_goal
                    continue
                if jp <= 0 or jp >= spec.W - 1:
                    P[a.value, s_idx, crash_idx] += 1.0
                    R[a.value, s_idx] += -spec.r_crash
                    continue
                # Otherwise apply wind at jp
                dist = wind_distribution(jp, spec)
                total_exp_r = 0.0
                for jf, prob in dist.items():
                    if prob == 0.0:
                        continue
                    if ip == spec.H - 1 or jf <= 0 or jf >= spec.W - 1:
                        # Out-of-bounds or terminal column → crash
                        # (goal row already handled above)
                        P[a.value, s_idx, crash_idx] += prob
                        total_exp_r += prob * (-spec.r_crash)
                    else:
                        P[a.value, s_idx, state_index[(ip, jf)]] += prob
                        total_exp_r += prob * (-spec.step_cost)
                R[a.value, s_idx] = total_exp_r

        super().__init__(
            states, actions,
            transitions=[P[a] for a in range(m)],
            rewards=[R[a] for a in range(m)],
            gamma=spec.gamma,
            terminal_states=[self.GOAL, self.CRASH],
            name=f"WindyChasm(H={spec.H},W={spec.W},B={spec.B})",
        )

    # ------------------------------------------------------------------
    #  Convenience helpers for the experiments / report
    # ------------------------------------------------------------------
    def start_state(self) -> Tuple[int, int]:
        return (0, self.spec.centre)

    def value_grid(self, V: np.ndarray) -> np.ndarray:
        """Reshape a value vector into an (H, W) grid with NaN outside cells."""
        g = np.full((self.spec.H, self.spec.W), np.nan)
        for s, i in self._state_index_public.items():
            if s in (self.GOAL, self.CRASH):
                continue
            g[s[0], s[1]] = V[i]
        return g

    def policy_grid(self, policy: np.ndarray) -> np.ndarray:
        """Return an (H, W) array of action integers (NaN outside the MDP)."""
        g = np.full((self.spec.H, self.spec.W), -1, dtype=int)
        for s, i in self._state_index_public.items():
            if s in (self.GOAL, self.CRASH):
                continue
            g[s[0], s[1]] = int(np.argmax(policy[i]))
        return g
