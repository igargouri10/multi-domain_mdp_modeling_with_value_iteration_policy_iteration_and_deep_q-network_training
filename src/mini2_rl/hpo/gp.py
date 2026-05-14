r"""Pure-NumPy Gaussian-process regression with an RBF kernel.

Textbook implementation of the GP surrogate (Rasmussen and Williams,
2006, Section 2.2). No scipy, no sklearn; the rest of mini2_rl is
already on the "pure NumPy" path because the V2 sandbox lacks a
``torch`` wheel.

The posterior at a query point :math:`x_\star` is

.. math::

   \mu(x_\star) = k_\star^\top \bigl(K + \sigma_n^2 I\bigr)^{-1} y \\
   \sigma^2(x_\star) = k(x_\star, x_\star)
       - k_\star^\top \bigl(K + \sigma_n^2 I\bigr)^{-1} k_\star

with RBF kernel
:math:`k(x, x') = \sigma_f^2 \exp(-\|x - x'\|^2 / (2\ell^2))`.

Kernel hyperparameters ``(sigma_f, ell, sigma_n)`` are fixed at
inference time rather than tuned by type-II MLE; on the small
~25-point datasets BO produces in V3, MLE on the kernel is unreliable
(Snoek et al. 2012, Section 3), so we set the signal variance equal
to the empirical variance of the observed targets and the length
scale via the median-pairwise-distance heuristic (Garreau et al. 2017).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


def rbf_kernel(
    X1: np.ndarray, X2: np.ndarray, length_scale: float, signal_var: float
) -> np.ndarray:
    """RBF/Gaussian kernel matrix; ``X1`` is ``(n1, d)``, ``X2`` is ``(n2, d)``."""
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    sq1 = np.sum(X1 ** 2, axis=1, keepdims=True)
    sq2 = np.sum(X2 ** 2, axis=1, keepdims=True).T
    sqd = sq1 + sq2 - 2.0 * X1 @ X2.T
    sqd = np.maximum(sqd, 0.0)
    return signal_var * np.exp(-0.5 * sqd / (length_scale ** 2))


@dataclass
class GaussianProcess:
    """Zero-mean GP regression with a fixed RBF kernel."""

    length_scale: float = 0.2
    signal_var: float = 1.0
    noise_var: float = 1e-4

    _X: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _y: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _L: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _y_mean: float = field(default=0.0, init=False, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcess":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError(f"Inconsistent shapes: X={X.shape}, y={y.shape}.")

        self._y_mean = float(y.mean())
        y_centered = y - self._y_mean

        n = X.shape[0]
        K = rbf_kernel(X, X, self.length_scale, self.signal_var)
        K_jitter = K + (self.noise_var + 1e-8) * np.eye(n)

        L = None
        jitter = 1e-8
        for _ in range(6):
            try:
                L = np.linalg.cholesky(K_jitter)
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
                K_jitter = K + (self.noise_var + jitter) * np.eye(n)
        if L is None:
            raise np.linalg.LinAlgError("GP Cholesky failed even with jitter.")

        self._X = X
        self._y = y_centered
        self._L = L
        return self

    def predict(
        self, X_star: np.ndarray, *, return_std: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self._X is None or self._L is None or self._y is None:
            raise RuntimeError("Fit the GP before calling predict().")
        X_star = np.atleast_2d(np.asarray(X_star, dtype=float))

        K_s = rbf_kernel(self._X, X_star, self.length_scale, self.signal_var)
        v = np.linalg.solve(self._L, self._y)
        alpha = np.linalg.solve(self._L.T, v)
        mu = K_s.T @ alpha + self._y_mean

        if not return_std:
            return mu, None

        w = np.linalg.solve(self._L, K_s)
        var = self.signal_var - np.sum(w * w, axis=0)
        var = np.maximum(var, 1e-12)
        return mu, np.sqrt(var)

    @staticmethod
    def heuristic_length_scale(X: np.ndarray) -> float:
        """Median heuristic for the RBF length scale (Garreau et al. 2017)."""
        X = np.asarray(X, dtype=float)
        if X.shape[0] < 2:
            return 0.2
        diffs = X[:, None, :] - X[None, :, :]
        d = np.sqrt(np.sum(diffs ** 2, axis=-1))
        triu = d[np.triu_indices_from(d, k=1)]
        med = float(np.median(triu))
        return max(med, 1e-3)
