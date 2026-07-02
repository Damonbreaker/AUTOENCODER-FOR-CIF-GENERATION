#!/usr/bin/env python3
"""Load Task 05 UMA node_emb_l0 + Task 02 normalized lattice labels for training."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

TASK08_DIR = Path(__file__).resolve().parent
TASK02_DIR = TASK08_DIR.parent / "02_structure_M"
TASK05_DIR = TASK08_DIR.parent / "05_encoder_gnn"
if str(TASK05_DIR) not in sys.path:
    sys.path.insert(0, str(TASK05_DIR))

from uma_split_utils import (  # noqa: E402
    SplitManifest,
    list_uma_node_pts,
    resolve_or_create_split,
)

DEFAULT_BATCH_LABEL = "N_0000_0050"
DEFAULT_UMA_IN_DIR = TASK05_DIR / "outputs" / "uma_node_by_batch"
DEFAULT_LATTICE_ROOT = TASK02_DIR / "outputs" / "normalized_lattice_by_N"
LATTICE_DIM = 6

# (atom_matrix [N,d], true_N scalar, true_L_norm [6])
Sample = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def torch_load(path: Path, map_location: str | torch.device = "cpu") -> dict:
    """torch.load compatible with PyTorch 2.0 (no weights_only kwarg)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _matrix_from_payload(data: dict) -> torch.Tensor:
    if "node_emb_l0" in data:
        return data["node_emb_l0"].float()
    if "node_emb" in data:
        emb = data["node_emb"]
        if emb.dim() == 3:
            return emb[:, 0, :].float()
        return emb.float()
    raise KeyError("missing node_emb_l0 / node_emb")


def resolve_lattice_npz(
    uma_pt: Path,
    data: dict,
    *,
    lattice_root: Path,
    batch_label: str,
) -> Path:
    """Map *_uma_node.pt -> Task 02 normalized_lattice_by_N batch npz."""
    if "source_npz" in data:
        stem = Path(str(data["source_npz"])).stem
    else:
        stem = uma_pt.name.replace("_uma_node.pt", "")
    return lattice_root / f"batch_{batch_label}" / f"{stem}.npz"


def load_L_norm(lattice_npz: Path) -> torch.Tensor:
    with np.load(lattice_npz, allow_pickle=False) as arr:
        if "L_norm" not in arr:
            raise KeyError(f"{lattice_npz.name}: missing L_norm")
        L = np.asarray(arr["L_norm"], dtype=np.float32).reshape(LATTICE_DIM)
    return torch.from_numpy(L)


def load_uma_sample(
    pt_path: Path,
    *,
    lattice_root: Path | None = DEFAULT_LATTICE_ROOT,
    batch_label: str = DEFAULT_BATCH_LABEL,
) -> Sample:
    """One crystal: node_emb_l0 [N,d], true N, normalized lattice L_norm [6]."""
    pt_path = Path(pt_path)
    data = torch_load(pt_path, map_location="cpu")
    atom_matrix = _matrix_from_payload(data)
    n_atoms = int(data["N"]) if "N" in data else int(atom_matrix.shape[0])
    if atom_matrix.shape[0] != n_atoms:
        raise ValueError(
            f"{pt_path.name}: node_emb_l0 rows {atom_matrix.shape[0]} != N={n_atoms}"
        )
    true_n = torch.tensor(float(n_atoms), dtype=torch.float32)

    if lattice_root is None:
        true_l = torch.zeros(LATTICE_DIM, dtype=torch.float32)
    else:
        lattice_npz = resolve_lattice_npz(
            pt_path, data, lattice_root=Path(lattice_root), batch_label=batch_label
        )
        if not lattice_npz.is_file():
            raise FileNotFoundError(
                f"missing normalized lattice label: {lattice_npz}\n"
                f"Run Task 02 normalize_lattice_by_batch for {batch_label}."
            )
        true_l = load_L_norm(lattice_npz)

    return atom_matrix.float(), true_n, true_l


def infer_d_model(pt_path: Path) -> int:
    data = torch_load(pt_path, map_location="cpu")
    return int(_matrix_from_payload(data).shape[-1])


def build_uma_splits(
    *,
    uma_in_dir: Path,
    batch_label: str = DEFAULT_BATCH_LABEL,
    split_seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    manifest_path: Path | None = None,
    resplit: bool = False,
    limit: int = 0,
    lattice_root: Path | None = DEFAULT_LATTICE_ROOT,
) -> tuple[list[str], list[str], list[str], SplitManifest, int]:
    """
    Discover *_uma_node.pt, split 70/15/15, return path lists (lazy load each epoch).
    """
    batch_in = Path(uma_in_dir) / f"batch_{batch_label}"
    uma_paths = list_uma_node_pts(batch_in)
    if limit > 0:
        uma_paths = uma_paths[:limit]

    if manifest_path is None:
        manifest_path = batch_in / "split_manifest.json"

    manifest = resolve_or_create_split(
        uma_paths,
        Path(manifest_path),
        train_frac=train_frac,
        val_frac=val_frac,
        seed=split_seed,
        batch_label=batch_label,
        resplit=resplit,
    )

    train_paths = manifest.splits["train"]
    val_paths = manifest.splits["val"]
    test_paths = manifest.splits["test"]
    d_model = infer_d_model(Path(train_paths[0]))

    if lattice_root is not None:
        sample_pt = Path(train_paths[0])
        sample_data = torch_load(sample_pt, map_location="cpu")
        sample_lat = resolve_lattice_npz(
            sample_pt,
            sample_data,
            lattice_root=Path(lattice_root),
            batch_label=batch_label,
        )
        if not sample_lat.is_file():
            raise FileNotFoundError(
                f"lattice labels missing for training: {sample_lat}\n"
                f"Expected Task 02 dir: {lattice_root}/batch_{batch_label}/"
            )

    return train_paths, val_paths, test_paths, manifest, d_model
