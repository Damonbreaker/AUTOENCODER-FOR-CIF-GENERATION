#!/usr/bin/env python3
"""UMA batch file listing and 70/15/15 split manifest (no PyG / heavy deps)."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

DEFAULT_BATCH_LABEL = "N_0000_0050"


def list_uma_node_pts(batch_in: Path) -> list[Path]:
    paths = sorted(batch_in.glob("*_uma_node.pt"))
    if not paths:
        raise FileNotFoundError(f"no *_uma_node.pt files in {batch_in}")
    return paths


@dataclass(frozen=True)
class SplitManifest:
    batch_label: str
    split_seed: int
    train_frac: float
    val_frac: float
    test_frac: float
    splits: dict[str, list[str]]

    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.splits.items()}

    def to_dict(self) -> dict[str, Any]:
        c = self.counts()
        return {
            "batch_label": self.batch_label,
            "split_seed": self.split_seed,
            "train_frac": self.train_frac,
            "val_frac": self.val_frac,
            "test_frac": self.test_frac,
            "counts": c,
            "total": sum(c.values()),
            "splits": self.splits,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SplitManifest:
        return cls(
            batch_label=str(data["batch_label"]),
            split_seed=int(data["split_seed"]),
            train_frac=float(data["train_frac"]),
            val_frac=float(data["val_frac"]),
            test_frac=float(data["test_frac"]),
            splits={k: list(v) for k, v in data["splits"].items()},
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> SplitManifest:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def split_uma_paths(
    uma_paths: list[Path],
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
    batch_label: str = DEFAULT_BATCH_LABEL,
) -> SplitManifest:
    """Deterministic 70/15/15 split over sorted UMA file paths."""
    if batch_label != DEFAULT_BATCH_LABEL:
        raise ValueError(f"only {DEFAULT_BATCH_LABEL} is supported, got {batch_label}")
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    if not 0.0 <= val_frac < 1.0 or train_frac + val_frac >= 1.0:
        raise ValueError("invalid val_frac or train+val >= 1")

    n = len(uma_paths)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()

    n_train = int(math.floor(n * train_frac))
    n_val = int(math.floor(n * val_frac))

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    def _pick(idxs: list[int]) -> list[str]:
        return [str(uma_paths[i].resolve()) for i in idxs]

    test_frac = 1.0 - train_frac - val_frac
    return SplitManifest(
        batch_label=batch_label,
        split_seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=round(test_frac, 6),
        splits={
            "train": _pick(train_idx),
            "val": _pick(val_idx),
            "test": _pick(test_idx),
        },
    )


def resolve_or_create_split(
    uma_paths: list[Path],
    manifest_path: Path,
    *,
    train_frac: float,
    val_frac: float,
    seed: int,
    batch_label: str,
    resplit: bool,
) -> SplitManifest:
    """Load existing split_manifest.json on resume; create once otherwise."""
    if manifest_path.is_file() and not resplit:
        manifest = SplitManifest.load(manifest_path)
        expected = {str(p.resolve()) for p in uma_paths}
        saved = set(manifest.splits["train"] + manifest.splits["val"] + manifest.splits["test"])
        if saved != expected:
            raise RuntimeError(
                f"split_manifest.json does not match current UMA inputs under batch_{batch_label}.\n"
                "Delete split_manifest.json and re-run, or pass --resplit."
            )
        print(f"Loaded existing split: {manifest.counts()}", flush=True)
        return manifest

    manifest = split_uma_paths(
        uma_paths,
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
        batch_label=batch_label,
    )
    manifest.save(manifest_path)
    print(f"Created split (seed={seed}): {manifest.counts()}", flush=True)
    return manifest
