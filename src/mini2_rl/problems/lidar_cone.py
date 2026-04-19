r"""Problem 2 — Robot Motion Control via Reward Shaping (lidar cone corridor).

The Mini-2 PDF proposes a diamond-shaped corridor on a 13-row × 7-column
grid (the canonical example reprinted at the top of the problem).  The grid
narrows from six traversable cells in the widest row (row 6) to a single
cell at rows 1 and 11.  The robot starts at ``A`` in row 6, col 0 and must
traverse the corridor *along its axis of symmetry* (row 6).  "Forward" in
this problem means increasing the column index; hardware noise causes
random ±1 drift in the row index each step.

State representation (answer to question (a))
---------------------------------------------
We propose a *Markovian* state

    s = (row, col)                              (observed position)

together with the *derived* lidar asymmetry channel

    Δ(s) = d_up(s) - d_down(s) = 2·row - 12,

which is what a forward-looking lidar cone would return as the difference
between the upper and lower sector distances.  ``Δ`` is a deterministic
function of ``row`` for this grid, so it carries no extra information —
but it makes the reward shaping transparent (see (b)) and it is what a
*noisy*  lidar-driven estimator would approximate on the real robot.

The task is Markov in the observed ``(row, col)`` because the wall
geometry and noise distribution depend only on the current cell.  In the
real robot (continuous pose, sensor latency, control-loop lag) this
Markov assumption is approximate; we discuss the consequences in the
report and in :func:`policy_iteration_discussion` below.

Actions
-------
Three discrete encoder-threshold adjustments:

* ``FORWARD``       — intended col += 1; drifts ±1 row with ``p_drift``.
* ``CORRECT_UP``    — intended row -= 1 (move toward row 0).
* ``CORRECT_DOWN``  — intended row += 1 (move toward row 12).

Reward function (answer to question (b))
----------------------------------------

.. math::
    R(s, a) = -c_\text{step}
              - \lambda |\Delta(s)|
              + R_\text{goal} \mathbf{1}[s' = \text{goal}]
              - r_\text{crash} \mathbf{1}[s' \in \text{wall}].

The ``-λ|Δ(s)|`` term is reward shaping (Ng, Harada, and Russell, ICML 1999)
and is bias-free with respect to the optimal policy because ``Δ`` is a
state function and reward shaping by a state-only potential does not change
the optimal policy.  Intuitively, it directly encodes the PDF's ask:
encourage forward motion by default and penalise off-axis drift.

Reward shaping reference: A. Y. Ng, D. Harada, and S. J. Russell, "Policy
invariance under reward transformations: theory and application to reward
shaping," in *Proc. 16th Int. Conf. Machine Learning (ICML)*, 1999.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Tuple

import numpy as np

from ..mdp.base import MarkovDecisionProcess


__all__ = [
    "LidarAction",
    "LidarConeSpec",
    "LidarConeMDP",
    "CANONICAL_CORRIDOR",
    "corridor_walkable_mask",
]


class LidarAction(IntEnum):
    FORWARD = 0
    CORRECT_UP = 1
    CORRECT_DOWN = 2


# The PDF's canonical diamond corridor.  We encode it as a 2-D mask with
# True = walkable, False = wall.  Rows 0..12, cols 0..6.
def _build_canonical_corridor() -> np.ndarray:
    H, W = 13, 7
    mask = np.zeros((H, W), dtype=bool)
    # Walkable cells: |row - 6| + |col - 0| ≤ 5 — a diamond of radius 5 with
    # vertex at (6, 0), expanding to (6, 5) as col increases … but the PDF
    # draws a diamond pointing *right*, so the maximum col depends on row
    # distance to 6.
    for r in range(H):
        for c in range(W):
            # Walkable iff (6 - |r - 6|) >= c + 1 after the PDF's wall
            # layout.  Equivalently, the ceiling at col c is row c and the
            # floor is row 12 - c, giving walkable rows ``c < r < 12 - c``.
            if c < r < 12 - c:
                mask[r, c] = True
    return mask


CANONICAL_CORRIDOR: np.ndarray = _build_canonical_corridor()


def corridor_walkable_mask(corridor: np.ndarray | None = None) -> np.ndarray:
    return CANONICAL_CORRIDOR.copy() if corridor is None else corridor.copy()


# ---------------------------------------------------------------------------
@dataclass
class LidarConeSpec:
    mask: np.ndarray | None = None        # corridor walkable mask
    start: Tuple[int, int] = (6, 0)
    goal: Tuple[int, int] = (6, 5)         # narrow tip of the cone
    p_drift: float = 0.15                  # probability of ±1 row drift
    lambda_shape: float = 0.2              # coefficient on |Δ| shaping term
    step_cost: float = 1.0
    crash_penalty: float = 20.0
    goal_reward: float = 50.0
    gamma: float = 0.95

    def resolved_mask(self) -> np.ndarray:
        return corridor_walkable_mask(self.mask)


# ---------------------------------------------------------------------------
class LidarConeMDP(MarkovDecisionProcess):
    """Finite MDP for the lidar-cone corridor (Problem 2)."""

    CRASH = ("CRASH", "")

    def __init__(self, spec: LidarConeSpec) -> None:
        self.spec = spec
        mask = spec.resolved_mask()
        H, W = mask.shape
        self.H, self.W = H, W
        self.mask = mask
        centre_row = H // 2
        # Non-terminal states = walkable cells that are not the goal.
        states: List = []
        for r in range(H):
            for c in range(W):
                if mask[r, c] and (r, c) != spec.goal:
                    states.append((r, c))
        states.append(spec.goal)
        states.append(self.CRASH)
        state_index = {s: k for k, s in enumerate(states)}
        self._state_index_public = state_index
        n = len(states)
        actions = list(LidarAction)
        m = len(actions)
        P = np.zeros((m, n, n))
        R = np.zeros((m, n))
        goal_idx = state_index[spec.goal]
        crash_idx = state_index[self.CRASH]
        # Absorbing terminals
        for a in range(m):
            P[a, goal_idx, goal_idx] = 1.0
            P[a, crash_idx, crash_idx] = 1.0

        def asymmetry(r: int) -> int:
            return 2 * r - 2 * centre_row

        def resolve(r: int, c: int) -> int:
            if not (0 <= r < H and 0 <= c < W) or not mask[r, c]:
                return crash_idx
            if (r, c) == spec.goal:
                return goal_idx
            return state_index[(r, c)]

        for s in states[:-2]:  # non-terminals
            s_idx = state_index[s]
            r, c = s
            # Each action has a nominal target cell; drift only on FORWARD.
            for a in actions:
                dist: Dict[int, float] = {}
                if a == LidarAction.FORWARD:
                    # (1 - p_drift) probability straight forward
                    nominal = (r, c + 1)
                    up = (r - 1, c + 1)
                    down = (r + 1, c + 1)
                    dist[resolve(*nominal)] = (
                        dist.get(resolve(*nominal), 0.0) + (1.0 - spec.p_drift)
                    )
                    dist[resolve(*up)] = (
                        dist.get(resolve(*up), 0.0) + spec.p_drift / 2
                    )
                    dist[resolve(*down)] = (
                        dist.get(resolve(*down), 0.0) + spec.p_drift / 2
                    )
                elif a == LidarAction.CORRECT_UP:
                    target = (r - 1, c)
                    dist[resolve(*target)] = 1.0
                else:  # CORRECT_DOWN
                    target = (r + 1, c)
                    dist[resolve(*target)] = 1.0

                total_r = 0.0
                for next_idx, prob in dist.items():
                    if prob == 0.0:
                        continue
                    P[a.value, s_idx, next_idx] += prob
                    if next_idx == goal_idx:
                        total_r += prob * spec.goal_reward
                    elif next_idx == crash_idx:
                        total_r += prob * (-spec.crash_penalty)
                    else:
                        # non-terminal step: base cost + shaping
                        total_r += prob * (
                            -spec.step_cost
                            - spec.lambda_shape * abs(asymmetry(r))
                        )
                R[a.value, s_idx] = total_r

        super().__init__(
            states, actions,
            transitions=[P[a] for a in range(m)],
            rewards=[R[a] for a in range(m)],
            gamma=spec.gamma,
            terminal_states=[spec.goal, self.CRASH],
            name=f"LidarCone(H={H},W={W})",
        )

    # Helpers --------------------------------------------------------------
    def value_grid(self, V: np.ndarray) -> np.ndarray:
        g = np.full((self.H, self.W), np.nan)
        for s, idx in self._state_index_public.items():
            if s in (self.CRASH,) or not isinstance(s, tuple):
                continue
            r, c = s
            g[r, c] = V[idx]
        return g

    def policy_grid(self, policy: np.ndarray) -> np.ndarray:
        g = np.full((self.H, self.W), -1, dtype=int)
        for s, idx in self._state_index_public.items():
            if s == self.CRASH or s == self.spec.goal:
                continue
            r, c = s
            g[r, c] = int(np.argmax(policy[idx]))
        return g


# ---------------------------------------------------------------------------
def policy_iteration_discussion() -> str:
    """Return the plain-text answer to Problem 2 question (c).

    The text is embedded here so it can be used both by the LaTeX report
    and printed from the ``run_all.py`` driver for sanity-checking.
    """
    return (
        "Problem 2 (c) Policy-iteration challenges on the real robot:\n"
        "  - Empirical P(s'|s,a) is estimated from rollouts. With sensor\n"
        "    noise (lidar fog, motor jitter, Kalman filter drift) the\n"
        "    discretised state aliases multiple underlying continuous poses,\n"
        "    inflating the variance of the empirical transition counts.\n"
        "  - Hardware variability means P drifts between training days.\n"
        "    Policy iteration assumes a stationary P, so naive Howard-1960\n"
        "    iterations may never stabilise.\n"
        "  Mitigations: (i) Bayesian posterior over P with a Dirichlet\n"
        "    conjugate prior (Strehl and Littman, JCSS 2008), (ii) robust\n"
        "    value iteration over a set of plausible P's (Iyengar,\n"
        "    Math. OR 2005; Wiesemann et al., OR 2013), (iii) model-free\n"
        "    methods (Q-learning, Actor-Critic) that sidestep the model\n"
        "    estimation problem entirely."
    )
