#!/usr/bin/env python3
"""
Attention-based compression of UMA node embeddings: (N, d) -> (1, d).

Reads Task 05 UMA outputs for N_0000_0050 only (*_uma_node.pt, node_emb_l0),
splits crystals 70/15/15 train/val/test, then writes compressed embeddings per split.

Usage:
  cd tasks/05_encoder_gnn
  python uma_attn_compress_N_0000_0050.py
  python uma_attn_compress_N_0000_0050.py --limit 10 --device cpu

Pure PyTorch only — no PyG / DGL.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

TASK_DIR = Path(__file__).resolve().parent
DEFAULT_UMA_IN_DIR = TASK_DIR / "outputs" / "uma_node_by_batch"
DEFAULT_OUT_DIR = TASK_DIR / "outputs" / "uma_attn_compress_by_batch"

SplitName = Literal["train", "val", "test"]

if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from uma_mh_pool import load_uma_node_matrix  # noqa: E402
from uma_split_utils import (  # noqa: E402
    DEFAULT_BATCH_LABEL,
    SplitManifest,
    list_uma_node_pts,
    resolve_or_create_split,
    split_uma_paths,
)


class AtomAttentionCompressor(nn.Module):
    """
    Compress UMA node_emb_l0 [N, d] -> crystal vector [1, d] via attention (steps 1–4).

    1. Mean pool:  t = mean(X)
    2. Project:    q = t W_q, K = X W_k, V = X W_v
    3. Attention:  a = softmax(q K^T / sqrt(d))
    4. Message:    m = a V   <- compressed output (no step-5 combine)
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.scale = math.sqrt(d_model)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        atom_features: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if atom_features.dim() != 2:
            raise ValueError(f"Expected [N, d], got {tuple(atom_features.shape)}")
        if atom_features.size(-1) != self.d_model:
            raise ValueError(
                f"Expected d_model={self.d_model}, got {atom_features.size(-1)}"
            )

        target = atom_features.mean(dim=0, keepdim=True)
        query = self.W_q(target)
        keys = self.W_k(atom_features)
        values = self.W_v(atom_features)
        scores = (query @ keys.transpose(0, 1)) / self.scale
        attention_weights = F.softmax(scores, dim=-1)
        message = attention_weights @ values

        if return_attention:
            return message, attention_weights
        return message


def build_compressor(in_dim: int) -> AtomAttentionCompressor:
    return AtomAttentionCompressor(d_model=in_dim)


def load_compressor_weights(model: AtomAttentionCompressor, weights_path: Path) -> None:
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "compressor_state_dict" in ckpt:
        model.load_state_dict(ckpt["compressor_state_dict"])
        return
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
        return
    if isinstance(ckpt, dict):
        try:
            model.load_state_dict(ckpt)
            return
        except RuntimeError:
            pass
    raise ValueError(f"could not load AtomAttentionCompressor weights from {weights_path}")


def compress_uma_matrix(
    x: torch.Tensor,
    model: AtomAttentionCompressor,
) -> tuple[torch.Tensor, torch.Tensor]:
    crystal, attn = model(x, return_attention=True)
    return crystal, attn


def uma_attn_compress_out_path(uma_node_pt: Path, split_out: Path) -> Path:
    name = uma_node_pt.name.replace("_uma_node.pt", "_uma_attn_compress.pt")
    if name == uma_node_pt.name:
        name = f"{uma_node_pt.stem}_uma_attn_compress.pt"
    return split_out / name


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def process_uma_attn_compress_batch(
    batch_label: str,
    in_dir: Path,
    out_dir: Path,
    *,
    compressor_weights: Path | None = None,
    init_seed: int = 42,
    split_seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    device: str = "cpu",
    limit: int = 0,
    overwrite: bool = False,
    resplit: bool = False,
    verbose_first: bool = True,
) -> dict[str, Any]:
    if batch_label != DEFAULT_BATCH_LABEL:
        raise ValueError(
            f"This pipeline supports only {DEFAULT_BATCH_LABEL} (0 ≤ N < 50), got {batch_label}"
        )

    batch_in = in_dir / f"batch_{batch_label}"
    batch_out = out_dir / f"batch_{batch_label}"
    batch_out.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        (batch_out / split_name).mkdir(parents=True, exist_ok=True)

    uma_paths = list_uma_node_pts(batch_in)
    if limit:
        uma_paths = uma_paths[:limit]

    manifest_path = batch_out / "split_manifest.json"
    manifest = resolve_or_create_split(
        uma_paths,
        manifest_path,
        train_frac=train_frac,
        val_frac=val_frac,
        seed=split_seed,
        batch_label=batch_label,
        resplit=resplit,
    )

    torch_device = torch.device(device)
    sample_x = load_uma_node_matrix(uma_paths[0])
    in_dim = int(sample_x.shape[-1])

    if compressor_weights is None:
        torch.manual_seed(init_seed)
    model = build_compressor(in_dim)
    if compressor_weights is not None:
        load_compressor_weights(model, compressor_weights)
    model = model.to(torch_device).eval()

    weights_out = batch_out / "atom_attn_compressor_weights.pt"
    if compressor_weights is None and not weights_out.is_file():
        torch.save(
            {
                "compressor_state_dict": model.state_dict(),
                "pool_mode": "atom_attention_compressor",
                "in_dim": in_dim,
                "out_dim": in_dim,
                "init_seed": init_seed,
                "note": "shared init weights for all splits",
            },
            weights_out,
        )

    summary_path = batch_out / "summary.csv"
    index_path = batch_out / "index.jsonl"
    summary_fields = [
        "split",
        "batch_label",
        "source_uma_pt",
        "attn_compress_pt",
        "N",
        "in_dim",
        "out_dim",
        "wall_time_sec",
    ]

    stats: dict[str, dict[str, int]] = {
        s: {"processed": 0, "skipped": 0, "errors": 0} for s in ("train", "val", "test")
    }
    t0 = time.perf_counter()
    global_idx = 0
    total = sum(len(v) for v in manifest.splits.values())

    with summary_path.open("w", newline="", encoding="utf-8") as sf, index_path.open(
        "w", encoding="utf-8"
    ) as jf:
        writer = csv.DictWriter(sf, fieldnames=summary_fields)
        writer.writeheader()

        for split_name in ("train", "val", "test"):
            split_out = batch_out / split_name
            for uma_pt_str in manifest.splits[split_name]:
                global_idx += 1
                uma_pt = Path(uma_pt_str)
                out_pt = uma_attn_compress_out_path(uma_pt, split_out)
                if out_pt.is_file() and not overwrite:
                    stats[split_name]["skipped"] += 1
                    continue

                t_one = time.perf_counter()
                try:
                    x = load_uma_node_matrix(uma_pt)
                    with torch.no_grad():
                        crystal, attn = compress_uma_matrix(x.to(torch_device), model)

                    if verbose_first and stats[split_name]["processed"] == 0 and split_name == "train":
                        print("\n=== First training sample (shape verification) ===")
                        print(f"Split:                   {split_name}")
                        print(f"Source UMA file:         {uma_pt.name}")
                        print(f"Input node_emb_l0:       {tuple(x.shape)}")
                        print(f"Attention weights:       {tuple(attn.shape)}")
                        print(f"Compressed crystal_emb:  {tuple(crystal.shape)}")
                        print(f"Attention weights sum:   {attn.sum().item():.6f}")
                        print("=" * 52)
                        verbose_first = False

                    payload = torch.load(uma_pt, map_location="cpu", weights_only=False)
                    record = {
                        "crystal_emb": crystal.squeeze(0).cpu(),
                        "attention_weights": attn.squeeze(0).cpu(),
                        "split": split_name,
                        "source_uma_pt": str(uma_pt),
                        "source_npz": payload.get("source_npz"),
                        "batch_label": batch_label,
                        "pool_mode": "atom_attention_compressor",
                        "in_dim": in_dim,
                        "out_dim": in_dim,
                        "N": int(x.shape[0]),
                    }
                    torch.save(record, out_pt)

                    row = {
                        "split": split_name,
                        "batch_label": batch_label,
                        "source_uma_pt": str(uma_pt),
                        "attn_compress_pt": str(out_pt),
                        "N": int(x.shape[0]),
                        "in_dim": in_dim,
                        "out_dim": in_dim,
                        "wall_time_sec": round(time.perf_counter() - t_one, 4),
                    }
                    writer.writerow(row)
                    jf.write(json.dumps(row) + "\n")
                    stats[split_name]["processed"] += 1

                    if global_idx % 500 == 0 or global_idx == 1:
                        print(
                            f"  [{global_idx}/{total}] {split_name}  "
                            f"N={x.shape[0]}  ->  {out_pt.name}",
                            flush=True,
                        )
                except Exception as exc:
                    stats[split_name]["errors"] += 1
                    print(f"ERROR [{split_name}] {uma_pt.name}: {exc}", flush=True)

    elapsed = time.perf_counter() - t0
    return {
        "batch_label": batch_label,
        "in_dir": str(batch_in),
        "out_dir": str(batch_out),
        "split_manifest": str(manifest_path),
        "split_counts": manifest.counts(),
        "total": total,
        "stats": stats,
        "elapsed_s": round(elapsed, 3),
        "in_dim": in_dim,
        "weights": str(weights_out),
        "summary": str(summary_path),
    }


def process_single_uma_file(
    uma_pt: Path,
    *,
    out_pt: Path | None = None,
    compressor_weights: Path | None = None,
    init_seed: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    torch_device = torch.device(device)
    x = load_uma_node_matrix(uma_pt)
    in_dim = int(x.shape[-1])
    if compressor_weights is None:
        torch.manual_seed(init_seed)
    model = build_compressor(in_dim)
    if compressor_weights is not None:
        load_compressor_weights(model, compressor_weights)
    model = model.to(torch_device).eval()
    with torch.no_grad():
        crystal, attn = compress_uma_matrix(x.to(torch_device), model)
    print(f"source:  {uma_pt}")
    print(f"input:   {tuple(x.shape)}")
    print(f"attn:    {tuple(attn.shape)}")
    print(f"output:  {tuple(crystal.shape)}")
    if out_pt is not None:
        out_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "crystal_emb": crystal.squeeze(0).cpu(),
                "attention_weights": attn.squeeze(0).cpu(),
                "source_uma_pt": str(uma_pt),
                "pool_mode": "atom_attention_compressor",
                "in_dim": in_dim,
                "out_dim": in_dim,
                "N": int(x.shape[0]),
            },
            out_pt,
        )
        print(f"saved:   {out_pt}")
    return {"input_shape": tuple(x.shape), "output_shape": tuple(crystal.shape)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=f"UMA attention compress — {DEFAULT_BATCH_LABEL} only, 70/15/15 split"
    )
    p.add_argument(
        "--batch-label",
        default=os.environ.get("UMA_BATCH_LABEL", DEFAULT_BATCH_LABEL),
        help=f"Must be {DEFAULT_BATCH_LABEL}",
    )
    p.add_argument(
        "--in-dir",
        type=Path,
        default=Path(os.environ.get("UMA_NODE_OUT_DIR", str(DEFAULT_UMA_IN_DIR))),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.environ.get("UMA_ATTN_COMPRESS_OUT_DIR", str(DEFAULT_OUT_DIR))),
    )
    p.add_argument("--uma-pt", type=Path, default=None, help="Single-file smoke test")
    p.add_argument("--out-pt", type=Path, default=None)
    p.add_argument("--compressor-weights", type=Path, default=None)
    p.add_argument("--init-seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--device", default=os.environ.get("UMA_ATTN_COMPRESS_DEVICE", "cpu"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--resplit",
        action="store_true",
        help="Recompute train/val/test split (deletes consistency with old outputs)",
    )
    return p.parse_args()


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.uma_pt is not None:
        return process_single_uma_file(
            args.uma_pt.expanduser().resolve(),
            out_pt=args.out_pt,
            compressor_weights=args.compressor_weights,
            init_seed=args.init_seed,
            device=args.device,
        )

    if args.batch_label != DEFAULT_BATCH_LABEL:
        raise SystemExit(
            f"ERROR: only {DEFAULT_BATCH_LABEL} is supported (0 ≤ N < 50).\n"
            f"Got: {args.batch_label}"
        )

    in_dir = args.in_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    batch_in = in_dir / f"batch_{args.batch_label}"
    if not batch_in.is_dir():
        raise FileNotFoundError(f"UMA batch missing: {batch_in}")

    print(f"=== UMA attention compress: {args.batch_label} ===")
    print(f"input : {batch_in}")
    print(f"output: {out_dir / f'batch_{args.batch_label}'}/{{train,val,test}}/")
    print(f"split : train={args.train_frac} val={args.val_frac} "
          f"test={1.0 - args.train_frac - args.val_frac:.2f} seed={args.split_seed}")
    print(f"device: {args.device}")

    result = process_uma_attn_compress_batch(
        args.batch_label,
        in_dir,
        out_dir,
        compressor_weights=args.compressor_weights,
        init_seed=args.init_seed,
        split_seed=args.split_seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        device=args.device,
        limit=args.limit,
        overwrite=args.overwrite,
        resplit=args.resplit,
    )

    print("\n=== Done ===")
    print(f"split_counts: {result['split_counts']}")
    for split_name, s in result["stats"].items():
        print(
            f"  {split_name}: processed={s['processed']}  "
            f"skipped={s['skipped']}  errors={s['errors']}"
        )
    print(f"elapsed: {result['elapsed_s']:.1f}s")
    print(f"manifest: {result['split_manifest']}")
    print(f"summary:  {result['summary']}")
    return result


def run_batch(batch_label: str, extra_argv: list[str] | None = None) -> dict[str, Any]:
    argv = ["--batch-label", batch_label] + (extra_argv or [])
    old = sys.argv
    try:
        sys.argv = [old[0]] + argv
        return _run(parse_args())
    finally:
        sys.argv = old


if __name__ == "__main__":
    _run(parse_args())
