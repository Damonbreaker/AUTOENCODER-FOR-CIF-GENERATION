#!/usr/bin/env python3
"""
CDVAE-style crystal representation M = (N, A, F, L, composition).

Extracted from tasks/02_structure_M/structure_M.py and structure_M_all.py.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, NamedTuple, Tuple

import gemmi
import numpy as np


class CrystalM(NamedTuple):
    """Crystal representation M = (N, A, F, L, composition)."""

    N: int
    A: np.ndarray  # (N,) int — atomic numbers
    F: np.ndarray  # (N, 3) float — fractional x,y,z
    L: np.ndarray  # (6,) float — a,b,c, alpha,beta,gamma (degrees)
    composition: str


def site_frac(site: Any) -> Tuple[float, float, float]:
    """Return (x, y, z) fractional coords for one gemmi site."""
    fract = getattr(site, "fract", None)
    if fract is not None:
        return (float(fract.x), float(fract.y), float(fract.z))
    frac = getattr(site, "frac", None)
    if frac is not None:
        return (float(frac.x), float(frac.y), float(frac.z))
    raise AttributeError("gemmi site has neither .fract nor .frac")


def composition_from_sites(structure: gemmi.SmallStructure) -> str:
    """Build a Hill-style formula string by counting elements in structure.sites."""
    counts: Counter = Counter(site.element.name for site in structure.sites)
    order: list[str] = []
    for el in ("C", "H"):
        if el in counts:
            order.append(el)
    for el in sorted(counts):
        if el not in order:
            order.append(el)
    parts: list[str] = []
    for el in order:
        n = counts[el]
        parts.append(f"{el}{n}" if n != 1 else el)
    return "".join(parts)


def composition_from_A(A: np.ndarray) -> str:
    """Hill-style formula from atomic numbers (works for any element via gemmi)."""
    counts: Counter = Counter(gemmi.Element(int(z)).name for z in A)
    order: list[str] = []
    for el in ("C", "H"):
        if el in counts:
            order.append(el)
    for el in sorted(counts):
        if el not in order:
            order.append(el)
    parts: list[str] = []
    for el in order:
        n = counts[el]
        parts.append(f"{el}{n}" if n != 1 else el)
    return "".join(parts)


def structure_to_M(structure: gemmi.SmallStructure) -> CrystalM:
    """Convert gemmi SmallStructure → CrystalM (N, A, F, L, composition)."""
    sites = structure.sites
    N = len(sites)
    A = np.array([site.element.atomic_number for site in sites], dtype=np.int64)
    F = np.array([site_frac(site) for site in sites], dtype=np.float64)
    cell = structure.cell
    L = np.array(
        [cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma],
        dtype=np.float64,
    )
    composition = composition_from_sites(structure)
    return CrystalM(N=N, A=A, F=F, L=L, composition=composition)


def check_M(m: CrystalM) -> None:
    """Raise ValueError if M = (N, A, F, L, composition) is internally inconsistent."""
    if m.N <= 0:
        raise ValueError(f"N must be positive, got {m.N}")
    if m.A.shape != (m.N,):
        raise ValueError(f"A shape {m.A.shape} != ({m.N},)")
    if m.F.shape != (m.N, 3):
        raise ValueError(f"F shape {m.F.shape} != ({m.N}, 3)")
    if m.L.shape != (6,):
        raise ValueError(f"L shape {m.L.shape} != (6,)")

    a, b, c, alpha, beta, gamma = m.L
    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError(f"Lattice lengths must be > 0, got a={a}, b={b}, c={c}")
    if not (0 < alpha < 180 and 0 < beta < 180 and 0 < gamma < 180):
        raise ValueError(f"Lattice angles must be in (0, 180), got {alpha}, {beta}, {gamma}")
    if not m.composition:
        raise ValueError("composition string is empty")

    expected = composition_from_A(m.A)
    if expected != m.composition:
        raise ValueError(
            f"composition mismatch: from A got {expected!r}, stored {m.composition!r}"
        )


def save_M(path: Path, m: CrystalM, out_dir: Path) -> Path:
    """Save CrystalM arrays to a compressed .npz file. Returns output path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}.npz"
    np.savez_compressed(
        out_path,
        N=np.int64(m.N),
        A=m.A,
        F=m.F,
        L=m.L,
        composition=np.array(m.composition),
        source_cif=str(path),
    )
    return out_path


def load_M(npz_path: Path) -> CrystalM:
    """Load CrystalM from a .npz file written by save_M."""
    data = np.load(npz_path, allow_pickle=False)
    return CrystalM(
        N=int(data["N"]),
        A=data["A"],
        F=data["F"],
        L=data["L"],
        composition=str(data["composition"]),
    )


def print_M_summary(path: Path, m: CrystalM) -> None:
    """Print N, A, F, L, composition for one crystal."""
    a, b, c, alpha, beta, gamma = m.L
    print("=" * 72)
    print(f"file        : {path}")
    print(f"composition : {m.composition}")
    print(f"N           : {m.N}")
    print(f"A (Z)       : {m.A.tolist()}")
    print(f"L           : a={a:.4f}, b={b:.4f}, c={c:.4f} Angstrom")
    print(f"              alpha={alpha:.2f}, beta={beta:.2f}, gamma={gamma:.2f} deg")
    print(f"F shape     : {m.F.shape}  (N atoms × 3 frac coords)")
    print("first 3 fractional coords (x, y, z):")
    for i in range(min(3, m.N)):
        x, y, z = m.F[i]
        znum = int(m.A[i])
        print(f"  site {i}: Z={znum:2d}  {x:.6f}  {y:.6f}  {z:.6f}")
    if m.N > 3:
        print(f"  ... ({m.N - 3} more sites)")
