r"""DQN-on-WindyChasm objective for the Mini-2 V3 BO loop.

The objective wraps a single training run of the V2 pure-NumPy DQN on
the Problem 1 Windy Chasm MDP and returns a scalar that the BO loop
maximizes.

The contract is ``f(params_dict) -> (scalar, metrics_dict)``, where
``params_dict`` carries native (un-normalized) values for the BO
search variables and ``scalar`` is the objective. Two scalar choices
are exposed via ``objective_signal``:

* ``"tail_mean"`` (default) --- mean return over the last ``tail_window``
  *training* episodes. Cheap; integrates over many rollouts; but
  dominated by exploration noise from the residual epsilon.
* ``"eval_return_start"`` --- single greedy rollout from the start
  state at the end of training. One-rollout signal; less noisy with
  respect to the *current policy* but high variance with respect to
  Q-network seeds.

The first V3 run used ``tail_mean`` and revealed a proxy mismatch
(see ``DECISIONS.md`` and the V3 report section); the second run uses
``eval_return_start`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..problems.windy_chasm import WindyChasmSpec, WindyChasmMDP
from ..agents.dqn_numpy import DQNConfig, train_dqn_on_mdp


@dataclass
class DQNWindyChasmObjective:
    """Train a NumPy DQN on Windy Chasm under params; return a scalar return."""

    spec: WindyChasmSpec = field(default_factory=WindyChasmSpec)
    episodes: int = 250
    max_steps: int = 60
    tail_window: int = 50
    fixed_hidden: Tuple[int, ...] = (64, 64)
    fixed_buffer_capacity: int = 20_000
    fixed_min_buffer: int = 500
    seed_offset_eval: int = 7919
    objective_signal: str = "tail_mean"

    _trial_count: int = field(default=0, init=False, repr=False)
    _mdp: Optional[Any] = field(default=None, init=False, repr=False)
    _start_idx: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._mdp = WindyChasmMDP(self.spec)
        self._start_idx = self._mdp._state_index_public[self._mdp.start_state()]

    def __call__(self, params: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        self._trial_count += 1
        seed = 1_000 + self._trial_count

        cfg = DQNConfig(
            hidden=self.fixed_hidden,
            lr=float(params["lr"]),
            buffer_capacity=self.fixed_buffer_capacity,
            batch_size=int(params["batch_size"]),
            min_buffer=self.fixed_min_buffer,
            epsilon_start=1.0,
            epsilon_end=float(params.get("epsilon_end", 0.05)),
            epsilon_decay_steps=int(params["epsilon_decay_steps"]),
            target_sync=int(params["target_sync"]),
            grad_clip=5.0,
            seed=seed,
        )

        out = train_dqn_on_mdp(
            self._mdp, cfg,
            episodes=self.episodes,
            max_steps=self.max_steps,
            start_state_index=self._start_idx,
        )

        returns = out["returns"]
        losses = out["losses"]
        tail = returns[-self.tail_window:] if len(returns) >= 1 else returns
        tail_mean = float(np.mean(tail)) if len(tail) else float("-inf")
        tail_std = float(np.std(tail)) if len(tail) else 0.0
        late_loss = float(np.mean(losses[-200:])) if len(losses) >= 1 else float("nan")
        eval_start = float(out["eval_return_start"])

        metrics = {
            "tail_mean_return": tail_mean,
            "tail_std_return":  tail_std,
            "tail_window":      int(self.tail_window),
            "eval_return_start": eval_start,
            "late_loss_mean":   late_loss,
            "episodes":         int(self.episodes),
        }
        if self.objective_signal == "tail_mean":
            scalar = tail_mean
        elif self.objective_signal == "eval_return_start":
            scalar = eval_start
        else:
            raise ValueError(
                "objective_signal must be 'tail_mean' or 'eval_return_start', "
                "got " + repr(self.objective_signal)
            )
        return scalar, metrics
