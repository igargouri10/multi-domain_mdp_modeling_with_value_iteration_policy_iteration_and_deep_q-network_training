r"""Deep Q-Network agent (V2 requirement -- one DRL algorithm).

Follows Mnih et al. (2015) with:
* Online Q-network Q_theta(s, a).
* Target network Q_{theta^-} synchronised every ``target_sync`` steps (or as
  an exponential moving average if ``ema_tau`` is provided).
* Replay buffer with random uniform sampling.
* epsilon-greedy exploration with linear decay.

The grid-world foundation environment only has a few tens of states, so a
tabular agent is vastly more sample-efficient. DQN is implemented here so we
satisfy the V2 "at least one DRL algorithm" requirement and so we have a path
forward for when the foundation environment grows. For the actual comparisons
in the report we still use the tabular agents.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from ..nn.architectures import MLP
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

from ..buffers import ReplayBuffer, Transition
from ..mdp.base import MarkovDecisionProcess
from .base import BaseAgent


def _one_hot(i: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    v[i] = 1.0
    return v


class DQNAgent(BaseAgent):
    algo = "dqn"

    def __init__(
        self,
        *,
        hidden: tuple = (64, 64),
        lr: float = 1e-3,
        gamma_override: Optional[float] = None,
        buffer_capacity: int = 10_000,
        batch_size: int = 64,
        min_buffer: int = 200,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 5_000,
        target_sync: int = 100,
        ema_tau: Optional[float] = None,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            hidden=hidden, lr=lr, gamma_override=gamma_override,
            buffer_capacity=buffer_capacity, batch_size=batch_size,
            min_buffer=min_buffer,
            epsilon_start=epsilon_start, epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
            target_sync=target_sync, ema_tau=ema_tau,
            seed=seed, device=device,
        )
        self.online: Optional[nn.Module] = None
        self.target: Optional[nn.Module] = None
        self.optim: Optional[Any] = None
        self.buffer: Optional[ReplayBuffer] = None
        self._steps = 0

    # ------------------------------------------------------------------
    def _build(self, env: MarkovDecisionProcess) -> None:
        if not HAS_TORCH:
            raise RuntimeError(
                "PyTorch is not installed; install with `pip install torch`."
            )
        hp = self.hyperparameters
        torch.manual_seed(hp["seed"])
        in_dim = env.num_states
        out_dim = env.num_actions
        self.online = MLP(in_dim, out_dim, hidden=hp["hidden"]).to(hp["device"])
        self.target = MLP(in_dim, out_dim, hidden=hp["hidden"]).to(hp["device"])
        self.target.load_state_dict(self.online.state_dict())
        self.optim = torch.optim.Adam(self.online.parameters(), lr=hp["lr"])
        self.buffer = ReplayBuffer(
            hp["buffer_capacity"],
            state_shape=(in_dim,),
            seed=hp["seed"],
        )

    def _epsilon(self) -> float:
        hp = self.hyperparameters
        frac = min(1.0, self._steps / max(1, hp["epsilon_decay_steps"]))
        return hp["epsilon_start"] + frac * (hp["epsilon_end"] - hp["epsilon_start"])

    # ------------------------------------------------------------------
    def train(
        self,
        env: MarkovDecisionProcess,
        *,
        episodes: int = 200,
        max_steps: int = 200,
        rng: Optional[np.random.Generator] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        if self.online is None:
            self._build(env)
        hp = self.hyperparameters
        device = hp["device"]
        rng = rng or np.random.default_rng(hp["seed"])
        gamma = hp["gamma_override"] if hp["gamma_override"] is not None else env.gamma

        returns_log = []
        loss_log = []
        for ep in range(episodes):
            s = int(rng.integers(env.num_states))
            ep_return = 0.0
            for _ in range(max_steps):
                if env.is_terminal(env.states[s]):
                    break
                # epsilon-greedy
                if rng.random() < self._epsilon():
                    a = int(rng.integers(env.num_actions))
                else:
                    with torch.no_grad():
                        obs = torch.from_numpy(_one_hot(s, env.num_states)).to(device)
                        a = int(self.online(obs).argmax().item())
                probs = env.P[a, s]
                s_next = int(rng.choice(env.num_states, p=probs))
                r = float(env.R[a, s])
                done = env.is_terminal(env.states[s_next])
                self.buffer.push(Transition(
                    _one_hot(s, env.num_states),
                    a, r,
                    _one_hot(s_next, env.num_states),
                    done,
                ))
                s = s_next; ep_return += r; self._steps += 1

                if len(self.buffer) >= hp["min_buffer"]:
                    batch = self.buffer.sample(hp["batch_size"])
                    ob = torch.from_numpy(batch["state"]).to(device)
                    ac = torch.from_numpy(batch["action"]).to(device)
                    rw = torch.from_numpy(batch["reward"]).to(device)
                    ob_next = torch.from_numpy(batch["next_state"]).to(device)
                    d = torch.from_numpy(batch["done"]).to(device)
                    q = self.online(ob).gather(1, ac.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        q_next = self.target(ob_next).max(dim=1).values
                        target = rw + gamma * q_next * (1.0 - d)
                    loss = ((q - target) ** 2).mean()
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    loss_log.append(float(loss.item()))
                    # sync target
                    if hp["ema_tau"] is not None:
                        tau = hp["ema_tau"]
                        with torch.no_grad():
                            for p_t, p_o in zip(self.target.parameters(), self.online.parameters()):
                                p_t.data.mul_(1 - tau).add_(tau * p_o.data)
                    elif self._steps % hp["target_sync"] == 0:
                        self.target.load_state_dict(self.online.state_dict())
            returns_log.append(ep_return)
        self.record.episodes_trained += episodes
        self.record.metrics = {
            "avg_return_last_50": float(np.mean(returns_log[-50:])) if returns_log else 0.0,
            "final_loss": float(np.mean(loss_log[-100:])) if loss_log else 0.0,
        }
        return self.record.metrics

    # ------------------------------------------------------------------
    def evaluate(
        self, env: MarkovDecisionProcess, *, episodes: int = 30,
        max_steps: int = 200, **_: Any,
    ) -> Dict[str, Any]:
        if self.online is None:
            raise RuntimeError("Call train() first")
        device = self.hyperparameters["device"]
        rng = np.random.default_rng(self.hyperparameters["seed"] + 1)
        returns = []
        for _ in range(episodes):
            s = int(rng.integers(env.num_states))
            ep_return = 0.0
            for _ in range(max_steps):
                if env.is_terminal(env.states[s]):
                    break
                with torch.no_grad():
                    obs = torch.from_numpy(_one_hot(s, env.num_states)).to(device)
                    a = int(self.online(obs).argmax().item())
                probs = env.P[a, s]
                s_next = int(rng.choice(env.num_states, p=probs))
                ep_return += float(env.R[a, s])
                s = s_next
            returns.append(ep_return)
        return {"avg_return": float(np.mean(returns)) if returns else 0.0}

    def state_dict(self) -> Dict[str, Any]:
        if self.online is None:
            return {"hyperparameters": self.hyperparameters}
        return {
            "hyperparameters": self.hyperparameters,
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.hyperparameters = state["hyperparameters"]
        if "online" in state:
            # Caller is expected to call _build with the env before load.
            self.online.load_state_dict(state["online"])
            self.target.load_state_dict(state["target"])
