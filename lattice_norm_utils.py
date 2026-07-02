"""
Lattice length normalization for Task 02 structure codes (L in .npz).

Two methods (angles alpha, beta, gamma unchanged):

1. scale_N  — CDVAE scale_length:
     a', b', c' = a, b, c / N^(1/3)

2. scale_V  — volume-based:
     V = a*b*c * sqrt(1 - cos²α - cos²β - cos²γ + 2 cosα cosβ cosγ)
     a', b', c' = a, b, c / V^(1/3)

Triclinic volume formula (all angles ≠ 90° allowed).
"""
from __future__ import annotations

import numpy as np

EPSILON = 1e-12


def triclinic_volume(L: np.ndarray) -> float:
    """
    Volume (Å³) from L = [a, b, c, alpha, beta, gamma] (angles in degrees).

    V = abc * sqrt(1 - cos²α - cos²β - cos²γ + 2 cosα cosβ cosγ)
    """
    L = np.asarray(L, dtype=np.float64).reshape(6)
    a, b, c, alpha, beta, gamma = L
    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError(f"Lattice lengths must be > 0, got a={a}, b={b}, c={c}")

    ar, br, gr = np.deg2rad([alpha, beta, gamma])
    ca, cb, cg = np.cos([ar, br, gr])
    radicand = 1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg
    if radicand < -EPSILON:
        raise ValueError(
            f"Invalid triclinic cell: volume radicand={radicand:.6e} "
            f"(angles α={alpha}, β={beta}, γ={gamma})"
        )
    radicand = max(float(radicand), 0.0)
    return float(a * b * c * np.sqrt(radicand))


def scale_length_by_N(L: np.ndarray, N: int) -> np.ndarray:
    """a,b,c <- a,b,c / N^(1/3); angles unchanged (CDVAE scale_length)."""
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")
    L = np.asarray(L, dtype=np.float64).reshape(6)
    a, b, c, alpha, beta, gamma = L
    inv = float(N) ** (1.0 / 3.0)
    return np.array([a / inv, b / inv, c / inv, alpha, beta, gamma], dtype=np.float64)


def scale_length_by_V(L: np.ndarray, volume: float | None = None) -> tuple[np.ndarray, float]:
    """a,b,c <- a,b,c / V^(1/3); angles unchanged."""
    L = np.asarray(L, dtype=np.float64).reshape(6)
    a, b, c, alpha, beta, gamma = L
    V = float(volume) if volume is not None else triclinic_volume(L)
    if V <= 0:
        raise ValueError(f"Volume must be positive, got {V}")
    inv = V ** (1.0 / 3.0)
    L_scaled = np.array([a / inv, b / inv, c / inv, alpha, beta, gamma], dtype=np.float64)
    return L_scaled, V


def normalize_lattice_both(L: np.ndarray, N: int) -> dict[str, np.ndarray | float]:
    """Return raw L, both scaled variants, and volume."""
    L = np.asarray(L, dtype=np.float64).reshape(6)
    V = triclinic_volume(L)
    L_N = scale_length_by_N(L, N)
    L_V, _ = scale_length_by_V(L, volume=V)
    return {
        "L": L,
        "L_scaled_N": L_N,
        "L_scaled_V": L_V,
        "volume": V,
        "N": int(N),
    }


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
