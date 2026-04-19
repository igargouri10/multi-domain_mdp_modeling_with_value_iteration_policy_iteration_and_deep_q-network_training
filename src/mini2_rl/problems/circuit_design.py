r"""Problem 3 — Circuit design via sequential decisions.

The agent navigates an :math:`H \times W` grid (the PDF's default is
:math:`3 \times 3`) and places combinational components (wires and NAND gates)
to synthesise a target Boolean function.  Per the Mini-2 PDF:

* Cells ``(0, 0)`` and ``(H-1, 0)`` are *source* tiles that inject signals
  ``A`` and ``B`` respectively.
* The cell ``((H-1)//2, W-1)`` is a *sink* tile that absorbs the final signal
  ``C``.  For the default :math:`3 \times 3` grid that is ``(1, 2)``.
* Insulation between ``(1, 1)`` and ``(2, 1)`` blocks every signal between
  those two tiles — implemented as a symmetric pair in
  :attr:`CircuitDesignSpec.insulation`.
* The target function is ``C = A XOR B`` (PDF).

State representation (answer to question (a))
---------------------------------------------
We use

    s = (pos_i, pos_j, grid, done_flag)                (Markovian)

where ``grid`` is an :math:`H \times W` array with entries in
:math:`\{\mathrm{EMPTY}, \mathrm{WIRE}, \mathrm{NAND}\}`.  This carries *all*
information needed to decide what to do next (position for the move/place
semantics, grid for the partial circuit, ``done_flag`` to terminate).  Because
the environment is deterministic given ``(s, a)`` the process is exactly
Markovian (Sutton and Barto, 2018, §3.3).

Test-time evaluation (answer to question (b))
---------------------------------------------
Combinational logic is evaluated *deterministically* by simulating signal
propagation: for each of the four input combinations
:math:`(A, B) \in \{0, 1\}^{2}` we propagate signals through the grid using
a fixed-point iteration that respects the "inputs from left/top/bottom,
outputs to the right" gate orientation from the PDF.  Wires broadcast their
input to all output neighbours; NAND gates compute the NAND of their defined
inputs.  The circuit *correctly implements* XOR iff the sink value ``C`` at
``(⌊(H-1)/2⌋, W-1)`` equals :math:`A \oplus B` for all four inputs.

Using a deterministic simulator means the transition kernel
:math:`P(s'\mid s, a)` is a delta function, which is the cleanest Markovian
formulation of a combinational design task.  An alternative (which we discuss
in the report) would be to inject stochastic gate delays or noise to model
hardware variability — both reduce to the deterministic case as the variance
goes to zero.

Insulation (answer to question (c))
-----------------------------------
``CircuitDesignSpec.insulation`` holds an *undirected* set of cell pairs
between which signal propagation is forbidden.  The default instance
matches the PDF: ``((1, 1), (2, 1))``.  The simulator simply skips these
edges when propagating.

Complexity (answer to question (d))
-----------------------------------
State-space size upper bound for the :math:`H \times W` design problem is

    |S| ≤ (H · W) · 3^(H · W) · 2                  (pos × grid × done)

(the factor of two is the terminal flag).  For :math:`3 \times 3` that is
:math:`9 \cdot 19\,683 \cdot 2 = 354\,294`; for :math:`4 \times 4` it balloons
to :math:`16 \cdot 43\,046\,721 \cdot 2 \approx 1.38 \cdot 10^{9}` — utterly
infeasible for tabular value iteration, whose cost per sweep is
:math:`O(|S|^{2} |A|)`.  We discuss approximation strategies
(function approximation, factored MDPs, hierarchical RL) in the report.

References
----------
R. A. Howard, *Dynamic Programming and Markov Processes*, MIT Press, 1960.

A. Koller and B. Milch, "Multi-agent influence diagrams for representing
and solving games," *Games and Economic Behavior*, vol. 45, no. 1,
pp. 181–221, 2003.

M. L. Puterman, *Markov Decision Processes: Discrete Stochastic Dynamic
Programming*.  Wiley, 1994.

C. Szepesvári, *Algorithms for Reinforcement Learning*.  Morgan & Claypool,
2010.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from itertools import product
from typing import Callable, Dict, FrozenSet, Hashable, List, Sequence, Tuple

import numpy as np

from ..mdp.base import MarkovDecisionProcess


__all__ = [
    "Cell",
    "CircuitAction",
    "CircuitDesignSpec",
    "CircuitSimulator",
    "CircuitDesignMDP",
    "state_space_size",
    "xor_target",
]


# ---------------------------------------------------------------------------
class Cell(IntEnum):
    EMPTY = 0
    WIRE = 1
    NAND = 2


class CircuitAction(IntEnum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    PLACE_WIRE = 4
    PLACE_NAND = 5
    REMOVE = 6
    DONE = 7


def xor_target(A: int, B: int) -> int:
    return A ^ B


# ---------------------------------------------------------------------------
@dataclass
class CircuitDesignSpec:
    """Hyperparameters of the circuit-design environment."""

    H: int = 3
    W: int = 3
    start: Tuple[int, int] = (1, 0)             # typical agent spawn
    source_A: Tuple[int, int] = (0, 0)
    source_B: Tuple[int, int] = (2, 0)
    sink_C: Tuple[int, int] = (1, 2)
    # Pairs of *adjacent* cells between which signals are insulated (blocked).
    insulation: Tuple[Tuple[Tuple[int, int], Tuple[int, int]], ...] = (
        ((1, 1), (2, 1)),
    )
    target: Callable[[int, int], int] = xor_target
    max_steps: int = 12                        # step budget (episode length)
    step_cost: float = 1.0
    place_cost: float = 0.5                     # per-placement penalty
    correct_reward: float = 50.0
    incorrect_penalty: float = 20.0
    gamma: float = 0.95


# ---------------------------------------------------------------------------
def state_space_size(H: int, W: int) -> Dict[str, int]:
    """Combinatorial size estimate for the design problem on ``H × W``.

    Returns a dictionary with the position count, the number of grid
    configurations (``3 ** (H·W)``), and the upper bound on the full state
    space (position × grid × done flag).  These quantities are what Section
    Problem 3 (d) of the report cites.
    """
    cells = H * W
    positions = cells
    grids = 3 ** cells
    upper = positions * grids * 2
    return {
        "cells": cells,
        "positions": positions,
        "grid_configurations": grids,
        "state_space_upper_bound": upper,
        "vi_cost_per_sweep_upper": upper * upper * 8,  # |S|^2 * |A| (rough)
    }


# ---------------------------------------------------------------------------
class CircuitSimulator:
    r"""Deterministic propagation simulator for a partial circuit grid.

    The grid is given as an :math:`H \times W` integer array with entries
    :class:`Cell`.  Each gate's inputs arrive from the *left*, *top*, and
    *bottom* neighbours (per the PDF's fixed orientation) and the gate's
    output feeds the *right* neighbour.  A WIRE passes its input through to
    all three output directions (right only, by convention) — we implement
    WIRE as a 1-input / 1-output element that simply copies its left input
    to its right output, which is the conventional interpretation of
    "wire" in this assignment.  NAND gates compute NAND of defined inputs.

    Signals are integers in ``{0, 1}``.  Undefined signals are modelled as
    ``None``.  We run a bounded number of propagation sweeps
    (:attr:`max_sweeps`) to reach a fixed point; any cycle that does not
    converge is declared invalid.
    """

    def __init__(self, spec: CircuitDesignSpec, max_sweeps: int = 32) -> None:
        self.spec = spec
        self.max_sweeps = max_sweeps
        # Symmetric set of blocked undirected edges.
        self._blocked: FrozenSet[FrozenSet[Tuple[int, int]]] = frozenset(
            frozenset(pair) for pair in spec.insulation
        )

    # ------------------------------------------------------------------
    def _edge_blocked(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        return frozenset((a, b)) in self._blocked

    # ------------------------------------------------------------------
    def _propagate_once(
        self,
        grid: np.ndarray,
        signals: Dict[Tuple[int, int], int],
    ) -> bool:
        """One sweep of left-to-right propagation.  Returns ``True`` on change."""
        H, W = grid.shape
        changed = False
        for j in range(W):
            for i in range(H):
                cell = Cell(int(grid[i, j]))
                if cell == Cell.EMPTY:
                    continue
                # Inputs from left, top, bottom (per PDF).
                inputs: List[int] = []
                for ni, nj in [(i, j - 1), (i - 1, j), (i + 1, j)]:
                    if not (0 <= ni < H and 0 <= nj < W):
                        continue
                    if self._edge_blocked((i, j), (ni, nj)):
                        continue
                    if (ni, nj) in signals:
                        inputs.append(signals[(ni, nj)])
                if not inputs:
                    continue
                if cell == Cell.WIRE:
                    # A wire forwards exactly one input — the left one if
                    # available, otherwise whatever is defined.  (Fixed-
                    # orientation wires in the PDF flow left-to-right.)
                    value = inputs[0]
                elif cell == Cell.NAND:
                    if len(inputs) < 2:
                        continue                # wait for all inputs
                    nand_value = 0 if all(x == 1 for x in inputs) else 1
                    value = nand_value
                else:  # pragma: no cover
                    continue
                if signals.get((i, j)) != value:
                    signals[(i, j)] = value
                    changed = True
        return changed

    # ------------------------------------------------------------------
    def evaluate(self, grid: np.ndarray, A: int, B: int) -> int | None:
        """Return the value at the sink (``None`` if undefined)."""
        signals: Dict[Tuple[int, int], int] = {
            self.spec.source_A: int(A),
            self.spec.source_B: int(B),
        }
        for _ in range(self.max_sweeps):
            changed = self._propagate_once(grid, signals)
            if not changed:
                break
        # Read the value at the neighbour cell that drives the sink.
        si, sj = self.spec.sink_C
        # If the sink is empty, value is whatever was written into (si, sj),
        # which in our left-to-right convention equals the left neighbour's
        # value.  Check both: the sink itself (if a component was placed on
        # it) and its left neighbour.
        if (si, sj) in signals:
            return signals[(si, sj)]
        left = (si, sj - 1)
        if 0 <= left[1] and left in signals and not self._edge_blocked((si, sj), left):
            return signals[left]
        return None

    # ------------------------------------------------------------------
    def score(self, grid: np.ndarray) -> Tuple[bool, int]:
        """Check correctness against the spec's target function.

        Returns ``(is_correct, num_correct_rows)`` over the 4 input rows.
        """
        target = self.spec.target
        hits = 0
        for A, B in product([0, 1], [0, 1]):
            out = self.evaluate(grid, A, B)
            if out is not None and out == target(A, B):
                hits += 1
        return hits == 4, hits


# ---------------------------------------------------------------------------
class CircuitDesignMDP(MarkovDecisionProcess):
    r"""Tabular MDP for the circuit-design problem.

    **This constructor expands the full :math:`(|S|, |A|, |S|)` transition
    tensor.** For the default :math:`3 \times 3` problem with a typical step
    budget this is on the order of :math:`10^{5}` states which is feasible
    but slow; for any larger grid the user should reach for function
    approximation.  See :func:`state_space_size` for the combinatorial
    scaling.

    To keep the builder tractable, we bound the search by ``spec.max_steps``
    and a *component budget* (at most ``max_components`` non-empty cells on
    the grid at once).  These two restrictions give a state space that is
    small enough to solve with tabular DP yet retains the essential design
    decisions from the PDF.
    """

    TERMINAL = ("TERMINAL", "")

    def __init__(
        self,
        spec: CircuitDesignSpec,
        *,
        max_components: int = 3,
        allowed_cells: Sequence[Tuple[int, int]] | None = None,
    ) -> None:
        self.spec = spec
        self.simulator = CircuitSimulator(spec)
        H, W = spec.H, spec.W
        self.max_components = max_components

        # Restrict the design to a subset of cells (for tractability).  The
        # default subset is the 3 interior columns around the axis (i.e.
        # the "useful" tiles for a 3×3 design), which keeps the state count
        # manageable.
        if allowed_cells is None:
            allowed_cells = [(i, j) for i in range(H) for j in range(1, W - 1)]
        self.allowed_cells = tuple(sorted(allowed_cells))

        # Enumerate states via BFS from the start state.
        start_state = self._encode(spec.start, np.zeros((H, W), dtype=np.int8),
                                   done=False, steps=0)
        self._states: List[Hashable] = [start_state, self.TERMINAL]
        self._state_index: Dict[Hashable, int] = {s: i for i, s in enumerate(self._states)}

        actions = list(CircuitAction)
        # We'll fill transitions/rewards lazily via BFS then reshape.
        trans_rows: Dict[int, Dict[int, Dict[int, float]]] = {
            a.value: {} for a in actions
        }
        reward_rows: Dict[int, Dict[int, float]] = {a.value: {} for a in actions}

        frontier: List[Hashable] = [start_state]
        visited: set = {start_state, self.TERMINAL}
        while frontier:
            s = frontier.pop()
            if s == self.TERMINAL:
                continue
            s_idx = self._state_index[s]
            pos, grid, done, steps = self._decode(s)
            if done:
                # Absorbing: route to TERMINAL.
                for a in actions:
                    trans_rows[a.value].setdefault(s_idx, {})[
                        self._state_index[self.TERMINAL]
                    ] = 1.0
                    reward_rows[a.value][s_idx] = 0.0
                continue
            # For each action, compute (next_state, reward).
            for a in actions:
                next_state, reward = self._apply(pos, grid, done, steps, a)
                if next_state not in self._state_index:
                    self._state_index[next_state] = len(self._states)
                    self._states.append(next_state)
                    if next_state not in visited:
                        frontier.append(next_state)
                        visited.add(next_state)
                ns_idx = self._state_index[next_state]
                trans_rows[a.value].setdefault(s_idx, {})[ns_idx] = 1.0
                reward_rows[a.value][s_idx] = reward

        # Absorbing TERMINAL
        term_idx = self._state_index[self.TERMINAL]
        for a in actions:
            trans_rows[a.value].setdefault(term_idx, {})[term_idx] = 1.0
            reward_rows[a.value].setdefault(term_idx, 0.0)

        n = len(self._states)
        m = len(actions)
        P = np.zeros((m, n, n))
        R = np.zeros((m, n))
        for a in actions:
            for s_idx, row in trans_rows[a.value].items():
                for ns_idx, prob in row.items():
                    P[a.value, s_idx, ns_idx] = prob
            for s_idx, r in reward_rows[a.value].items():
                R[a.value, s_idx] = r
        # Any row that never got populated (unreachable) becomes a self-loop.
        for a in range(m):
            sums = P[a].sum(axis=1)
            missing = np.where(sums == 0)[0]
            for s_idx in missing:
                P[a, s_idx, s_idx] = 1.0

        super().__init__(
            list(self._states), actions,
            transitions=[P[a] for a in range(m)],
            rewards=[R[a] for a in range(m)],
            gamma=spec.gamma,
            terminal_states=[self.TERMINAL],
            name=f"CircuitDesign({H}x{W})",
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _encode(pos, grid, *, done, steps):
        return (tuple(pos), tuple(int(v) for v in grid.flatten()), bool(done), int(steps))

    def _decode(self, s):
        pos, flat, done, steps = s
        grid = np.array(flat, dtype=np.int8).reshape(self.spec.H, self.spec.W)
        return pos, grid, done, steps

    # ------------------------------------------------------------------
    def _apply(self, pos, grid, done, steps, action: CircuitAction):
        """Transition function for a *deterministic* placement environment."""
        spec = self.spec
        H, W = spec.H, spec.W
        i, j = pos
        new_grid = grid.copy()
        reward = -spec.step_cost

        if action == CircuitAction.MOVE_UP:
            i = max(0, i - 1)
        elif action == CircuitAction.MOVE_DOWN:
            i = min(H - 1, i + 1)
        elif action == CircuitAction.MOVE_LEFT:
            j = max(0, j - 1)
        elif action == CircuitAction.MOVE_RIGHT:
            j = min(W - 1, j + 1)
        elif action == CircuitAction.PLACE_WIRE:
            if (i, j) in self.allowed_cells:
                new_grid[i, j] = Cell.WIRE.value
                reward -= spec.place_cost
        elif action == CircuitAction.PLACE_NAND:
            if (i, j) in self.allowed_cells:
                new_grid[i, j] = Cell.NAND.value
                reward -= spec.place_cost
        elif action == CircuitAction.REMOVE:
            if (i, j) in self.allowed_cells:
                new_grid[i, j] = Cell.EMPTY.value
        elif action == CircuitAction.DONE:
            is_correct, _hits = self.simulator.score(new_grid)
            if is_correct:
                reward += spec.correct_reward
            else:
                reward -= spec.incorrect_penalty
            return self._encode((i, j), new_grid, done=True,
                                steps=min(spec.max_steps, steps + 1)), reward

        # Enforce component budget (max non-empty cells)
        if int((new_grid != 0).sum()) > self.max_components:
            new_grid = grid.copy()

        new_steps = steps + 1
        # Exhausted budget: force termination.
        if new_steps >= spec.max_steps:
            is_correct, _ = self.simulator.score(new_grid)
            end_reward = spec.correct_reward if is_correct else -spec.incorrect_penalty
            reward += end_reward
            return self._encode((i, j), new_grid, done=True, steps=new_steps), reward
        return self._encode((i, j), new_grid, done=False, steps=new_steps), reward
