"""Core Markov process / MRP / MDP abstractions.

These modules are vendored from the Mini-1 repository (``mini1_rl.mdp.base``
and ``mini1_rl.mdp.value``) so Mini-2 can be handed in as a self-contained
deliverable. Any improvements made here should be ported back to Mini-1.
"""

from .base import (
    MarkovProcess,
    MarkovRewardProcess,
    MarkovDecisionProcess,
    TERMINAL_SENTINEL,
)
from .value import (
    solve_value_matrix,
    solve_value_iterative,
    solve_value_eigendecomp,
)
from .dp import (
    policy_iteration,
    value_iteration,
    q_from_v,
    greedy_policy_from_q,
)

__all__ = [
    "MarkovProcess",
    "MarkovRewardProcess",
    "MarkovDecisionProcess",
    "TERMINAL_SENTINEL",
    "solve_value_matrix",
    "solve_value_iterative",
    "solve_value_eigendecomp",
    "policy_iteration",
    "value_iteration",
    "q_from_v",
    "greedy_policy_from_q",
]
