"""
Standard scaling (z-score) for lattice lengths a, b, c.

Angles alpha, beta, gamma are never scaled — only lengths get (x - mean) / std.
"""
from __future__ import annotations

from typing import Any

import numpy as np

EPSILON = 1e-5


class ABCScaler:
    """StandardScaler on 3-D [a, b, c] lengths."""

    def __init__(self) -> None:
        self.means: np.ndarray | None = None
        self.stds: np.ndarray | None = None

    def fit(self, abc: np.ndarray) -> ABCScaler:
        X = np.asarray(abc, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError(f"expected (M, 3) for [a,b,c], got {X.shape}")
        if len(X) == 0:
            raise ValueError("cannot fit ABCScaler on empty data")
        self.means = X.mean(axis=0)
        self.stds = X.std(axis=0, ddof=0) + EPSILON
        return self

    def transform(self, abc: np.ndarray) -> np.ndarray:
        if self.means is None or self.stds is None:
            raise RuntimeError("ABCScaler not fitted")
        X = np.asarray(abc, dtype=np.float64)
        single = X.ndim == 1
        if single:
            X = X.reshape(1, 3)
        out = (X - self.means) / self.stds
        return out[0] if single else out

    def to_dict(self) -> dict[str, Any]:
        if self.means is None or self.stds is None:
            raise RuntimeError("ABCScaler not fitted")
        return {
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
            "epsilon": EPSILON,
            "feature_names": ["a", "b", "c"],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ABCScaler:
        obj = cls()
        obj.means = np.asarray(d["means"], dtype=np.float64)
        obj.stds = np.asarray(d["stds"], dtype=np.float64)
        return obj


def abc_to_L(abc: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Build 6-D L = [a,b,c,alpha,beta,gamma]."""
    a, b, c = [float(x) for x in np.asarray(abc, dtype=np.float64).reshape(3)]
    alpha, beta, gamma = [float(x) for x in np.asarray(angles, dtype=np.float64).reshape(3)]
    return np.array([a, b, c, alpha, beta, gamma], dtype=np.float64)


def L_to_dict(L: np.ndarray, suffix: str = "") -> dict[str, float]:
    a, b, c, alpha, beta, gamma = [float(x) for x in L]
    s = f"_{suffix}" if suffix else ""
    return {
        f"a{s}": a,
        f"b{s}": b,
        f"c{s}": c,
        f"alpha{s}": alpha,
        f"beta{s}": beta,
        f"gamma{s}": gamma,
    }
