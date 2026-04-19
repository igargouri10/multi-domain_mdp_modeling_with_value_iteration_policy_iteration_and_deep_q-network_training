"""Mini-2 agents (DP + DQN).

Two DQN backends are provided:

* :mod:`.dqn` — a PyTorch implementation that mirrors Mnih *et al.* (2015).
  Activated only if torch is importable.
* :mod:`.dqn_numpy` — a pure-NumPy reference implementation.  Always
  available; used by the report's figures so results are fully reproducible
  without a deep-learning framework.
"""

from .base import BaseAgent, AgentRecord
from .dqn_numpy import DQNNumpy, DQNConfig, train_dqn_on_mdp

try:
    from .dqn import DQNAgent  # noqa: F401
    HAS_TORCH_DQN = True
except ImportError:
    HAS_TORCH_DQN = False

__all__ = [
    "BaseAgent", "AgentRecord",
    "DQNNumpy", "DQNConfig", "train_dqn_on_mdp",
    "HAS_TORCH_DQN",
]
if HAS_TORCH_DQN:
    __all__.append("DQNAgent")
