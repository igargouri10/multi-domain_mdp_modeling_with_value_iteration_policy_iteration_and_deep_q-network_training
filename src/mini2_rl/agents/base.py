r"""Base Agent class (V1, requirement 5) and lightweight run-record dataclass.

All Mini-1 / V1 / V2 algorithms follow the same interface so that the top-level
train/evaluate script can swap them out with no conditional branching:

.. code-block:: text

   agent = make_agent(...)
   agent.train(env, ...)          # further training
   agent.evaluate(env, ...)       # Monte-Carlo evaluation of the current policy
   agent.save(dir)                # dumps best value/policy kernel pair
   agent.load(dir)

The abstract contract keeps two artefacts: ``V`` (or ``Q``) the value
function, and ``policy`` the action-selection kernel (a ``(|S|, |A|)`` array
for tabular agents, a neural network for DRL agents).
"""

from __future__ import annotations

import abc
import json
import os
import pickle
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class AgentRecord:
    """Lightweight metadata + metrics saved alongside every checkpoint."""
    algo: str
    task: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    episodes_trained: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


class BaseAgent(abc.ABC):
    """Common interface for every agent in the repository."""

    algo: str = "BaseAgent"

    def __init__(self, **hyper: Any) -> None:
        self.hyperparameters: Dict[str, Any] = dict(hyper)
        self.record = AgentRecord(
            algo=self.algo, task="unknown", hyperparameters=dict(hyper)
        )

    # ------------------------------------------------------------------
    @abc.abstractmethod
    def train(self, env: Any, *, episodes: int = 1, **kwargs: Any) -> Dict[str, Any]:
        """Further-train the agent and return a metrics dict."""

    @abc.abstractmethod
    def evaluate(self, env: Any, *, episodes: int = 10, **kwargs: Any) -> Dict[str, Any]:
        """Evaluate the current policy without updating anything."""

    # ------------------------------------------------------------------
    # Checkpointing: store the value function + policy + record.
    # ------------------------------------------------------------------
    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        state = self.state_dict()
        with open(os.path.join(directory, "state.pkl"), "wb") as fh:
            pickle.dump(state, fh)
        with open(os.path.join(directory, "record.json"), "w") as fh:
            fh.write(self.record.to_json())

    def load(self, directory: str) -> None:
        with open(os.path.join(directory, "state.pkl"), "rb") as fh:
            state = pickle.load(fh)
        self.load_state_dict(state)
        with open(os.path.join(directory, "record.json")) as fh:
            d = json.load(fh)
        self.record = AgentRecord(**d)

    # Subclasses override these two.
    def state_dict(self) -> Dict[str, Any]:  # pragma: no cover
        return {"hyperparameters": self.hyperparameters}

    def load_state_dict(self, state: Dict[str, Any]) -> None:  # pragma: no cover
        self.hyperparameters = state["hyperparameters"]
