r"""Pure-NumPy Deep Q-Network agent for the V2 DRL requirement.

Implements Mnih et al. (2015) with manually-differentiated MLP backpropagation
so the agent runs without a deep-learning framework.  The model is a two-
hidden-layer MLP :math:`Q_\theta(s, \cdot): \mathbb{R}^{|S|} \to \mathbb{R}^{|A|}`
trained with target-network Bellman fitting and a uniform replay buffer.

Design choices and references
-----------------------------
* **Target network** synchronised every :math:`N` gradient steps.
  V. Mnih *et al.*, "Human-level control through deep reinforcement
  learning," *Nature*, vol. 518, pp. 529–533, 2015.
* **Uniform experience replay** decouples the online trajectory from the
  gradient updates, breaking the i.i.d. assumption violation that online
  fitting would suffer from.  L.-J. Lin, "Self-improving reactive agents
  based on reinforcement learning, planning and teaching,"
  *Machine Learning*, vol. 8, pp. 293–321, 1992.
* **epsilon-greedy exploration** with linear decay from
  :math:`\epsilon_{0}` to :math:`\epsilon_{\infty}` over
  ``epsilon_decay_steps`` environment interactions.
* **Adam optimiser** (D. Kingma and J. Ba, "Adam: a method for stochastic
  optimisation," ICLR 2015).  We implement Adam by hand.
* **ReLU activations** (X. Glorot, A. Bordes, and Y. Bengio, "Deep sparse
  rectifier neural networks," AISTATS 2011).

Why hand-rolled instead of PyTorch?  Anthropic's sandbox has no GPU and no
torch wheel available; pure NumPy is reproducible, fast enough for the
95-state Windy Chasm MDP, and leaves no framework-version drift risk for the
report's "reproduce this result" claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


__all__ = ["DQNConfig", "DQNNumpy", "train_dqn_on_mdp"]


# ---------------------------------------------------------------------------
@dataclass
class DQNConfig:
    hidden: Tuple[int, ...] = (64, 64)
    lr: float = 5e-4
    gamma_override: Optional[float] = None
    buffer_capacity: int = 20_000
    batch_size: int = 64
    min_buffer: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000
    target_sync: int = 250
    grad_clip: float = 5.0
    seed: int = 0


# ---------------------------------------------------------------------------
def _init_mlp(dims: List[int], rng: np.random.Generator) -> List[Dict[str, np.ndarray]]:
    layers: List[Dict[str, np.ndarray]] = []
    for a, b in zip(dims[:-1], dims[1:]):
        # Glorot / Xavier uniform
        limit = np.sqrt(6.0 / (a + b))
        W = rng.uniform(-limit, limit, size=(a, b)).astype(np.float32)
        b_ = np.zeros((b,), dtype=np.float32)
        layers.append({"W": W, "b": b_})
    return layers


def _forward(layers: List[Dict[str, np.ndarray]], x: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, np.ndarray]]]:
    cache: List[Dict[str, np.ndarray]] = []
    h = x
    for k, layer in enumerate(layers):
        z = h @ layer["W"] + layer["b"]
        cache.append({"x": h, "z": z})
        if k < len(layers) - 1:
            h = np.maximum(0.0, z)  # ReLU
        else:
            h = z                    # linear head
    return h, cache


def _backward(
    layers: List[Dict[str, np.ndarray]],
    cache: List[Dict[str, np.ndarray]],
    dout: np.ndarray,
) -> List[Dict[str, np.ndarray]]:
    grads: List[Dict[str, np.ndarray]] = [None] * len(layers)   # type: ignore
    d = dout
    for k in reversed(range(len(layers))):
        h = cache[k]["x"]
        z = cache[k]["z"]
        if k < len(layers) - 1:
            # we applied ReLU after matmul on forward, so the derivative of
            # the activation *was* applied at layer k+1's input; here we're
            # at the pre-activation of layer k, which is linear -> ReLU.
            pass
        W = layers[k]["W"]
        grads[k] = {
            "W": h.T @ d / h.shape[0],
            "b": d.mean(axis=0),
        }
        if k > 0:
            d_prev_pre = d @ W.T
            # Apply ReLU' at layer k-1's post-activation (which fed into k)
            prev_z = cache[k - 1]["z"]
            d = d_prev_pre * (prev_z > 0).astype(np.float32)
    return grads


# ---------------------------------------------------------------------------
class DQNNumpy:
    """Hand-rolled DQN for one-hot-encoded finite MDP states."""

    def __init__(self, n_states: int, n_actions: int, config: DQNConfig) -> None:
        self.n_states, self.n_actions = n_states, n_actions
        self.cfg = config
        rng = np.random.default_rng(config.seed)
        dims = [n_states, *config.hidden, n_actions]
        self.online = _init_mlp(dims, rng)
        self.target = [{"W": l["W"].copy(), "b": l["b"].copy()} for l in self.online]
        # Adam state
        self.m = [{"W": np.zeros_like(l["W"]), "b": np.zeros_like(l["b"])} for l in self.online]
        self.v = [{"W": np.zeros_like(l["W"]), "b": np.zeros_like(l["b"])} for l in self.online]
        self.t_step = 0
        self.rng = rng
        # Replay
        self.buf_s = np.zeros((config.buffer_capacity, n_states), dtype=np.float32)
        self.buf_a = np.zeros(config.buffer_capacity, dtype=np.int64)
        self.buf_r = np.zeros(config.buffer_capacity, dtype=np.float32)
        self.buf_sp = np.zeros((config.buffer_capacity, n_states), dtype=np.float32)
        self.buf_d = np.zeros(config.buffer_capacity, dtype=np.float32)
        self._buf_size = 0
        self._buf_pos = 0

    # ------------------------------------------------------------------
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        q, _ = _forward(self.online, obs)
        return q

    def act(self, obs: np.ndarray, eps: float) -> int:
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        q = self.q_values(obs[None, :])[0]
        return int(np.argmax(q))

    def push(self, s: np.ndarray, a: int, r: float, sp: np.ndarray, done: bool) -> None:
        i = self._buf_pos
        self.buf_s[i] = s; self.buf_a[i] = a
        self.buf_r[i] = r; self.buf_sp[i] = sp
        self.buf_d[i] = 1.0 if done else 0.0
        self._buf_pos = (self._buf_pos + 1) % self.cfg.buffer_capacity
        self._buf_size = min(self._buf_size + 1, self.cfg.buffer_capacity)

    def _sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        idx = self.rng.integers(0, self._buf_size, size=batch_size)
        return (
            self.buf_s[idx], self.buf_a[idx], self.buf_r[idx],
            self.buf_sp[idx], self.buf_d[idx],
        )

    # ------------------------------------------------------------------
    def _adam_step(self, grads: List[Dict[str, np.ndarray]]) -> None:
        self.t_step += 1
        lr, beta1, beta2, eps = self.cfg.lr, 0.9, 0.999, 1e-8
        for k, layer in enumerate(self.online):
            for key in ("W", "b"):
                g = grads[k][key]
                # grad clip
                gnorm = np.linalg.norm(g)
                if gnorm > self.cfg.grad_clip:
                    g = g * (self.cfg.grad_clip / (gnorm + 1e-12))
                self.m[k][key] = beta1 * self.m[k][key] + (1 - beta1) * g
                self.v[k][key] = beta2 * self.v[k][key] + (1 - beta2) * (g * g)
                m_hat = self.m[k][key] / (1 - beta1 ** self.t_step)
                v_hat = self.v[k][key] / (1 - beta2 ** self.t_step)
                layer[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def _learn(self, gamma: float) -> float:
        s, a, r, sp, d = self._sample(self.cfg.batch_size)
        # Target: r + gamma * max Q_target(s', .) * (1 - done)
        q_tgt_sp, _ = _forward(self.target, sp)
        target = r + gamma * q_tgt_sp.max(axis=1) * (1.0 - d)
        # Predicted Q(s, a)
        q_online, cache = _forward(self.online, s)
        pred = q_online[np.arange(len(a)), a]
        td = pred - target
        # MSE gradient
        dL_dq = np.zeros_like(q_online)
        dL_dq[np.arange(len(a)), a] = td * (2.0 / len(a))
        grads = _backward(self.online, cache, dL_dq)
        self._adam_step(grads)
        return float((td ** 2).mean())

    def sync_target(self) -> None:
        self.target = [{"W": l["W"].copy(), "b": l["b"].copy()} for l in self.online]


# ---------------------------------------------------------------------------
def train_dqn_on_mdp(
    mdp,
    config: DQNConfig,
    *,
    episodes: int = 300,
    max_steps: int = 100,
    start_state_index: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Train a :class:`DQNNumpy` on a finite MDP using one-hot state features.

    Returns a dict with the per-episode returns and losses, ready to plot.
    """
    n = mdp.num_states
    m = mdp.num_actions
    gamma = config.gamma_override if config.gamma_override is not None else mdp.gamma
    agent = DQNNumpy(n, m, config)
    rng = np.random.default_rng(config.seed + 1)
    returns_log: List[float] = []
    loss_log: List[float] = []
    eye = np.eye(n, dtype=np.float32)

    def terminal_idx(i: int) -> bool:
        return mdp.is_terminal(mdp.states[i])

    start = start_state_index if start_state_index is not None else 0

    for ep in range(episodes):
        s = start
        ep_return = 0.0
        for t in range(max_steps):
            if terminal_idx(s):
                break
            eps = agent.cfg.epsilon_start + min(
                1.0, agent.t_step / max(1, agent.cfg.epsilon_decay_steps)
            ) * (agent.cfg.epsilon_end - agent.cfg.epsilon_start)
            a = agent.act(eye[s], eps)
            probs = mdp.P[a, s]
            sp = int(rng.choice(n, p=probs))
            r = float(mdp.R[a, s])
            done = terminal_idx(sp)
            agent.push(eye[s], a, r, eye[sp], done)
            s = sp
            ep_return += r
            if agent._buf_size >= agent.cfg.min_buffer:
                loss = agent._learn(gamma)
                loss_log.append(loss)
                if agent.t_step % agent.cfg.target_sync == 0:
                    agent.sync_target()
        returns_log.append(ep_return)
        if verbose and (ep + 1) % 25 == 0:
            print(f"  episode {ep + 1:4d}  eps={eps:.3f}  "
                  f"return_last25={np.mean(returns_log[-25:]):.2f}")
    # Final greedy evaluation from the start state
    eval_return = 0.0
    s = start
    for _ in range(max_steps):
        if terminal_idx(s):
            break
        q = agent.q_values(eye[s][None, :])[0]
        a = int(np.argmax(q))
        probs = mdp.P[a, s]
        sp = int(rng.choice(n, p=probs))
        eval_return += float(mdp.R[a, s])
        s = sp
    return {
        "returns": np.array(returns_log),
        "losses": np.array(loss_log),
        "eval_return_start": float(eval_return),
        "agent": agent,
    }
