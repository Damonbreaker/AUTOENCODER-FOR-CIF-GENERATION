#!/usr/bin/env python3
"""Task 02 structure .npz → ASE Atoms for UMA (physical cell, full PBC)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from ase import Atoms
    from ase.geometry import cellpar_to_cell
except ImportError as exc:  # pragma: no cover
    raise ImportError("UMA encoder requires ASE: pip install ase") from exc


@dataclass(frozen=True)
class StructureNpzRecord:
    npz_path: Path
    source_cif: str
    N: int
    composition: str
    A: np.ndarray
    F: np.ndarray
    L: np.ndarray


def load_structure_npz(npz_path: Path) -> StructureNpzRecord:
    with np.load(npz_path, allow_pickle=False) as data:
        n = int(data["N"])
        a = np.asarray(data["A"], dtype=np.int64)
        f = np.asarray(data["F"], dtype=np.float64)
        lvec = np.asarray(data["L"], dtype=np.float64).reshape(6)
        composition = str(data["composition"])
        source_cif = str(data["source_cif"]) if "source_cif" in data else ""
    if a.shape != (n,):
        raise ValueError(f"{npz_path.name}: A shape {a.shape} != ({n},)")
    if f.shape != (n, 3):
        raise ValueError(f"{npz_path.name}: F shape {f.shape} != ({n}, 3)")
    return StructureNpzRecord(
        npz_path=npz_path,
        source_cif=source_cif,
        N=n,
        composition=composition,
        A=a,
        F=f,
        L=lvec,
    )


def record_to_ase_atoms(record: StructureNpzRecord) -> Atoms:
    """Convert Task 02 M=(A,F,L) to ASE Atoms with 3D periodic boundary conditions."""
    a, b, c, alpha, beta, gamma = [float(x) for x in record.L]
    cell = cellpar_to_cell([a, b, c, alpha, beta, gamma])
    cart = record.F @ cell
    atoms = Atoms(
        numbers=record.A,
        positions=cart,
        cell=cell,
        pbc=[True, True, True],
    )
    return atoms


def list_structure_npz_in_batch(batch_dir: Path, batch_label: str) -> list[Path]:
    folder = batch_dir / f"batch_{batch_label}"
    if not folder.is_dir():
        raise FileNotFoundError(f"missing Task 02 batch dir: {folder}")
    paths = sorted(folder.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no .npz files in {folder}")
    return paths


def uma_node_out_path(struct_npz: Path, batch_out: Path) -> Path:
    stem = struct_npz.stem + "_uma_node.pt"
    return batch_out / stem
