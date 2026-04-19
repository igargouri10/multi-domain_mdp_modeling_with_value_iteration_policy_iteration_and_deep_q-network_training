r"""Replay buffer with bounded size and disk persistence (V2 requirement).

Directory layout expected on disk::

    data/replay_buffers/
      <algo>/
        <task>/
          <policy_freshness>/
            chunk_0000.npz, chunk_0001.npz, ...

The class enforces a hard size cap (``max_transitions``); beyond it, the
oldest chunk is deleted or overwritten by the :func:`rotate` method. A small
CLI script in ``scripts/replace_old_experience.py`` exposes the same logic.

References
----------
Lin, "Self-improving reactive agents based on reinforcement learning"
(1992, *Machine Learning*, vol. 8) -- original experience replay.
Mnih et al., "Human-level control through deep reinforcement learning"
(*Nature* 518, 2015) -- large-scale DQN replay.
"""

from __future__ import annotations

import os
import pickle
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        *,
        state_shape: Optional[Tuple[int, ...]] = None,
        seed: int = 0,
        storage_root: Optional[str] = None,
        algo: str = "dqn",
        task: str = "gridworld",
        freshness: str = "current",
    ) -> None:
        assert capacity > 0, "capacity must be positive"
        self.capacity = capacity
        self._buffer: Deque[Transition] = deque(maxlen=capacity)
        self._rng = np.random.default_rng(seed)
        self.state_shape = state_shape
        self.storage_root = storage_root
        self.algo = algo
        self.task = task
        self.freshness = freshness

    # ------------------------------------------------------------------
    def push(self, transition: Transition) -> None:
        assert (
            self.state_shape is None
            or transition.state.shape == self.state_shape
        ), (
            f"Expected state shape {self.state_shape}, got {transition.state.shape}"
        )
        self._buffer.append(transition)
        assert len(self._buffer) <= self.capacity

    def __len__(self) -> int:
        return len(self._buffer)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        assert len(self._buffer) >= batch_size, "not enough transitions"
        idx = self._rng.integers(0, len(self._buffer), size=batch_size)
        batch = [self._buffer[i] for i in idx]
        return {
            "state": np.stack([t.state for t in batch]),
            "action": np.array([t.action for t in batch], dtype=np.int64),
            "reward": np.array([t.reward for t in batch], dtype=np.float32),
            "next_state": np.stack([t.next_state for t in batch]),
            "done": np.array([t.done for t in batch], dtype=np.float32),
        }

    # ------------------------------------------------------------------
    #  On-disk persistence
    # ------------------------------------------------------------------
    def _chunk_dir(self) -> str:
        assert self.storage_root is not None
        return os.path.join(self.storage_root, self.algo, self.task, self.freshness)

    def save_chunk(self, tag: str = "latest") -> str:
        d = self._chunk_dir()
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"chunk_{tag}.pkl")
        with open(path, "wb") as fh:
            pickle.dump(list(self._buffer), fh)
        return path

    def load_chunk(self, path: str) -> None:
        with open(path, "rb") as fh:
            items = pickle.load(fh)
        for t in items:
            self.push(t)

    def rotate(self, keep_newest: int) -> None:
        """Drop the oldest transitions, keeping only ``keep_newest``."""
        assert keep_newest >= 0, "keep_newest must be non-negative"
        while len(self._buffer) > keep_newest:
            self._buffer.popleft()
