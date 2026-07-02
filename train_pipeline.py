#!/usr/bin/env python3
"""
VAE-Attention-Atom-Predictor training pipeline (Task 08).

Composes:
  - attention compression (Task 05 AtomAttentionCompressor)
  - VAE reparameterization (vae_reparameterize.py)
  - MLP heads for N (MSE+Poisson) and L (MSE) (mlp_head_n.py, mlp_head_l.py)

Reads Task 05 UMA outputs (*_uma_node.pt), Task 02 normalized lattice (L_norm),
trains end-to-end with backprop for 100 epochs by default.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Tuple

import torch
import torch.nn as nn

from mlp_head_l import LatticeDecoder, lattice_loss
from mlp_head_n import AtomCountDecoder, atom_count_loss
from uma_train_dataset import (
    DEFAULT_BATCH_LABEL,
    DEFAULT_LATTICE_ROOT,
    DEFAULT_UMA_IN_DIR,
    LATTICE_DIM,
    build_uma_splits,
    load_uma_sample,
)
from vae_atom_compressor import VAEAtomCompressor
from vae_reparameterize import kl_divergence

# Re-exports for compare_compression.py, inspect_val_predictions.py, etc.
__all__ = [
    "AtomCountDecoder",
    "LatticeDecoder",
    "VAEAtomCompressor",
    "compute_multitask_loss",
    "train_pipeline",
]

D_MODEL = 128
LATENT_DIM = 64
DEFAULT_OUT_ROOT = "outputs/vae_attn_atom_predictor_uma_nl"
HIDDEN_DIM = 128
N_MIN = 20
N_MAX = 50
SPLIT_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
EPOCHS = 100
CHECKPOINT_EVERY = 25
W_VAE = 0.01
W_ATTN = 0.1
W_ATOM = 1.0
W_LATTICE = 10.0


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def attention_entropy_loss(attn_weights: torch.Tensor) -> torch.Tensor:
    """Negative entropy of attention distribution over atoms."""
    p = attn_weights.clamp(min=1e-8)
    return -torch.sum(p * torch.log(p))


@dataclass
class LossBreakdown:
    total: torch.Tensor
    vae_kld: torch.Tensor
    attn_entropy: torch.Tensor
    atom_mse: torch.Tensor
    atom_poisson: torch.Tensor
    atom_total: torch.Tensor
    lattice_mse: torch.Tensor

    def as_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().cpu()),
            "vae_kld": float(self.vae_kld.detach().cpu()),
            "attn_entropy": float(self.attn_entropy.detach().cpu()),
            "atom_mse": float(self.atom_mse.detach().cpu()),
            "atom_poisson": float(self.atom_poisson.detach().cpu()),
            "atom_total": float(self.atom_total.detach().cpu()),
            "lattice_mse": float(self.lattice_mse.detach().cpu()),
        }


def compute_multitask_loss(
    pred_n: torch.Tensor,
    true_n: torch.Tensor,
    pred_L: torch.Tensor,
    true_L: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    attn_weights: torch.Tensor,
    *,
    w_vae: float = W_VAE,
    w_attn: float = W_ATTN,
    w_atom: float = W_ATOM,
    w_lattice: float = W_LATTICE,
    mse_fn: nn.MSELoss | None = None,
    poisson_fn: nn.PoissonNLLLoss | None = None,
) -> LossBreakdown:
    if mse_fn is None:
        mse_fn = nn.MSELoss()
    if poisson_fn is None:
        poisson_fn = nn.PoissonNLLLoss(log_input=False, full=False)

    vae_kld = kl_divergence(mu, logvar)
    attn_entropy = attention_entropy_loss(attn_weights)
    atom_bd = atom_count_loss(pred_n, true_n, mse_fn=mse_fn, poisson_fn=poisson_fn)
    lat_bd = lattice_loss(pred_L, true_L, mse_fn=mse_fn)

    total = (
        w_vae * vae_kld
        + w_attn * attn_entropy
        + w_atom * atom_bd.total
        + w_lattice * lat_bd.mse
    )

    return LossBreakdown(
        total=total,
        vae_kld=vae_kld,
        attn_entropy=attn_entropy,
        atom_mse=atom_bd.mse,
        atom_poisson=atom_bd.poisson,
        atom_total=atom_bd.total,
        lattice_mse=lat_bd.mse,
    )


# ---------------------------------------------------------------------------
# Dataset — UMA (default) or synthetic smoke test
# ---------------------------------------------------------------------------


Sample = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def generate_synthetic_sample(rng: random.Random, d_model: int = D_MODEL) -> Sample:
    n_atoms = rng.randint(N_MIN, N_MAX)
    true_n = torch.tensor(float(n_atoms), dtype=torch.float32)
    true_l = torch.randn(LATTICE_DIM, dtype=torch.float32) * 0.5
    crystal_bias = torch.randn(1, d_model)
    atom_matrix = torch.randn(n_atoms, d_model) * 0.25 + crystal_bias
    return atom_matrix.float(), true_n, true_l


def generate_synthetic_dataset(n_samples: int = 100, seed: int = SPLIT_SEED) -> list[Sample]:
    rng = random.Random(seed)
    return [generate_synthetic_sample(rng) for _ in range(n_samples)]


def split_samples(
    samples: list[Sample],
    *,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    seed: int = SPLIT_SEED,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    n = len(samples)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()

    n_train = int(math.floor(n * train_frac))
    n_val = int(math.floor(n * val_frac))
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    pick = lambda idxs: [samples[i] for i in idxs]  # noqa: E731
    return pick(train_idx), pick(val_idx), pick(test_idx)


def iter_samples(
    samples: list[Sample] | list[str],
    *,
    lattice_root: Path | None = DEFAULT_LATTICE_ROOT,
    batch_label: str = DEFAULT_BATCH_LABEL,
) -> Iterator[Sample]:
    for item in samples:
        if isinstance(item, (str, Path)):
            yield load_uma_sample(
                Path(item),
                lattice_root=lattice_root,
                batch_label=batch_label,
            )
        else:
            yield item


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------


@dataclass
class EpochMetrics:
    total: float = 0.0
    vae_kld: float = 0.0
    attn_entropy: float = 0.0
    atom_total: float = 0.0
    lattice_mse: float = 0.0
    n_samples: int = 0

    def update(self, breakdown: LossBreakdown) -> None:
        d = breakdown.as_dict()
        self.total += d["total"]
        self.vae_kld += d["vae_kld"]
        self.attn_entropy += d["attn_entropy"]
        self.atom_total += d["atom_total"]
        self.lattice_mse += d["lattice_mse"]
        self.n_samples += 1

    def average(self) -> dict[str, float]:
        n = max(self.n_samples, 1)
        return {
            "total": self.total / n,
            "vae_kld": self.vae_kld / n,
            "attn_entropy": self.attn_entropy / n,
            "atom_total": self.atom_total / n,
            "lattice_mse": self.lattice_mse / n,
        }


def run_epoch(
    compressor: VAEAtomCompressor,
    atom_decoder: AtomCountDecoder,
    lattice_decoder: LatticeDecoder,
    samples: list[Sample] | list[str],
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
    lattice_root: Path | None = DEFAULT_LATTICE_ROOT,
    batch_label: str = DEFAULT_BATCH_LABEL,
) -> dict[str, float]:
    if train:
        compressor.train()
        atom_decoder.train()
        lattice_decoder.train()
    else:
        compressor.eval()
        atom_decoder.eval()
        lattice_decoder.eval()

    metrics = EpochMetrics()
    mse_fn = nn.MSELoss()
    poisson_fn = nn.PoissonNLLLoss(log_input=False, full=False)

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for atom_matrix, true_n, true_l in iter_samples(
            samples,
            lattice_root=lattice_root,
            batch_label=batch_label,
        ):
            atom_matrix = atom_matrix.to(device)
            true_n = true_n.to(device)
            true_l = true_l.to(device)

            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            z, mu, logvar, attn = compressor(atom_matrix)
            z_vec = z.squeeze(0)
            pred_n = atom_decoder(z_vec)
            pred_l = lattice_decoder(z_vec)

            breakdown = compute_multitask_loss(
                pred_n,
                true_n,
                pred_l,
                true_l,
                mu,
                logvar,
                attn,
                mse_fn=mse_fn,
                poisson_fn=poisson_fn,
            )
            metrics.update(breakdown)

            if train and optimizer is not None:
                breakdown.total.backward()
                optimizer.step()

    return metrics.average()


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def checkpoint_path(checkpoint_dir: Path, epoch: int) -> Path:
    return checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.is_dir():
        return None
    paths = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    return paths[-1] if paths else None


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    compressor: VAEAtomCompressor,
    atom_decoder: AtomCountDecoder,
    lattice_decoder: LatticeDecoder,
    optimizer: torch.optim.Optimizer,
    train_history: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "compressor_state_dict": compressor.state_dict(),
            "atom_decoder_state_dict": atom_decoder.state_dict(),
            "lattice_decoder_state_dict": lattice_decoder.state_dict(),
            "decoder_state_dict": atom_decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_history": train_history,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    compressor: VAEAtomCompressor,
    atom_decoder: AtomCountDecoder,
    lattice_decoder: LatticeDecoder,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, list[dict[str, Any]]]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    compressor.load_state_dict(ckpt["compressor_state_dict"])
    if "atom_decoder_state_dict" in ckpt:
        atom_decoder.load_state_dict(ckpt["atom_decoder_state_dict"])
    else:
        atom_decoder.load_state_dict(ckpt["decoder_state_dict"])
    if "lattice_decoder_state_dict" in ckpt:
        lattice_decoder.load_state_dict(ckpt["lattice_decoder_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch = int(ckpt["epoch"])
    history = list(ckpt.get("train_history", []))
    return epoch, history


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    task_dir = Path(__file__).resolve().parent
    default_uma_in = Path(
        os.environ.get("VAE_ATTN_UMA_IN_DIR", str(DEFAULT_UMA_IN_DIR))
    )
    default_ckpt = Path(
        os.environ.get(
            "VAE_ATTN_CHECKPOINT_DIR",
            f"{DEFAULT_OUT_ROOT}/checkpoints",
        )
    )
    if not default_ckpt.is_absolute():
        default_ckpt = task_dir / default_ckpt
    p = argparse.ArgumentParser(description="Train VAE-Attention-Atom-Predictor on UMA")
    p.add_argument("--checkpoint-dir", type=Path, default=default_ckpt)
    p.add_argument(
        "--uma-in-dir",
        type=Path,
        default=default_uma_in,
        help="Parent dir containing batch_N_0000_0050/ with *_uma_node.pt",
    )
    p.add_argument(
        "--batch-label",
        default=os.environ.get("VAE_ATTN_BATCH_LABEL", DEFAULT_BATCH_LABEL),
    )
    p.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Optional path for split_manifest.json (default: under uma batch dir)",
    )
    p.add_argument("--resplit", action="store_true", help="Regenerate 70/15/15 split")
    p.add_argument(
        "--limit",
        type=int,
        default=int(os.environ.get("VAE_ATTN_LIMIT", "0")),
        help="Process only first N UMA files (0 = all)",
    )
    p.add_argument(
        "--lattice-root",
        type=Path,
        default=Path(os.environ.get("VAE_ATTN_LATTICE_ROOT", str(DEFAULT_LATTICE_ROOT))),
        help="Task 02 normalized_lattice_by_N parent directory",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint in --checkpoint-dir (default: train from scratch)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=int(os.environ.get("VAE_ATTN_TRAIN_EPOCHS", str(EPOCHS))),
    )
    p.add_argument(
        "--lr",
        type=float,
        default=float(os.environ.get("VAE_ATTN_TRAIN_LR", "1e-3")),
    )
    p.add_argument(
        "--device",
        default=os.environ.get(
            "VAE_ATTN_TRAIN_DEVICE",
            os.environ.get("CDVAE_TRAIN_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("VAE_ATTN_SPLIT_SEED", str(SPLIT_SEED))),
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Use random demo data (local smoke test only — not for production)",
    )
    p.add_argument(
        "--w-lattice",
        type=float,
        default=float(os.environ.get("VAE_ATTN_W_LATTICE", str(W_LATTICE))),
        help="Weight for lattice MSE loss (CDVAE default cost_lattice=10)",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=int(os.environ.get("VAE_ATTN_N_SAMPLES", "100")),
        help="Only used with --synthetic",
    )
    return p.parse_args()


def _run_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    lattice_root = None if args.synthetic else args.lattice_root.expanduser().resolve()
    return {
        "lattice_root": lattice_root,
        "batch_label": args.batch_label,
    }


def train_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    global W_LATTICE
    W_LATTICE = float(args.w_lattice)
    run_kw = _run_kwargs(args)

    print("=== VAE-Attention-Atom-Predictor pipeline ===")
    print(f"device: {device}  epochs: {args.epochs}  lr: {args.lr}")
    print(
        f"loss weights: w_vae={W_VAE}  w_attn={W_ATTN}  w_atom={W_ATOM}  w_lattice={W_LATTICE}"
    )

    data_source = "synthetic" if args.synthetic else "uma"
    d_model = D_MODEL
    split_manifest_info: dict[str, Any] | None = None

    if args.synthetic:
        print("WARNING: --synthetic enabled — not using real UMA outputs.")
        samples = generate_synthetic_dataset(n_samples=args.n_samples, seed=args.seed)
        train_samples, val_samples, test_samples = split_samples(samples, seed=args.seed)
        print(
            f"dataset: source=synthetic  total={len(samples)}  "
            f"train={len(train_samples)}  val={len(val_samples)}  test={len(test_samples)}"
        )
    else:
        uma_in = args.uma_in_dir.expanduser().resolve()
        lattice_root = run_kw["lattice_root"]
        print(f"dataset: source=uma  batch={args.batch_label}")
        print(f"uma_in_dir: {uma_in}")
        print(f"lattice_root: {lattice_root}")
        train_paths, val_paths, test_paths, split_manifest, d_model = build_uma_splits(
            uma_in_dir=uma_in,
            batch_label=args.batch_label,
            split_seed=args.seed,
            train_frac=TRAIN_FRAC,
            val_frac=VAL_FRAC,
            manifest_path=args.split_manifest,
            resplit=args.resplit,
            limit=args.limit,
            lattice_root=lattice_root,
        )
        counts = split_manifest.counts()
        print(
            f"dataset: total={counts['train'] + counts['val'] + counts['test']}  "
            f"train={counts['train']}  val={counts['val']}  test={counts['test']}  "
            f"d_model={d_model}  (lazy load per epoch)"
        )
        split_manifest_info = split_manifest.to_dict()
        train_samples, val_samples, test_samples = train_paths, val_paths, test_paths

    compressor = VAEAtomCompressor(d_model=d_model, latent_dim=LATENT_DIM).to(device)
    atom_decoder = AtomCountDecoder(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM).to(device)
    lattice_decoder = LatticeDecoder(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(
        list(compressor.parameters())
        + list(atom_decoder.parameters())
        + list(lattice_decoder.parameters()),
        lr=args.lr,
    )

    start_epoch = 0
    train_history: list[dict[str, Any]] = []
    ckpt_dir = args.checkpoint_dir.expanduser().resolve()

    if args.resume:
        latest = find_latest_checkpoint(ckpt_dir)
        if latest is not None:
            start_epoch, train_history = load_checkpoint(
                latest, compressor, atom_decoder, lattice_decoder, optimizer, device
            )
            print(f"Resumed from {latest.name} at epoch {start_epoch}")
        else:
            print("No checkpoint found — training from scratch.")
    else:
        print("Training from scratch (--resume not set).")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_metrics = run_epoch(
            compressor,
            atom_decoder,
            lattice_decoder,
            train_samples,
            optimizer=optimizer,
            device=device,
            train=True,
            **run_kw,
        )
        val_metrics = run_epoch(
            compressor,
            atom_decoder,
            lattice_decoder,
            val_samples,
            optimizer=None,
            device=device,
            train=False,
            **run_kw,
        )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        train_history.append(record)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"epoch {epoch:03d}/{args.epochs}  "
                f"train_loss={train_metrics['total']:.4f}  "
                f"val_loss={val_metrics['total']:.4f}  "
                f"train_atom={train_metrics['atom_total']:.4f}  "
                f"val_atom={val_metrics['atom_total']:.4f}  "
                f"train_lat={train_metrics['lattice_mse']:.4f}  "
                f"val_lat={val_metrics['lattice_mse']:.4f}",
                flush=True,
            )

        if epoch % CHECKPOINT_EVERY == 0 or epoch == args.epochs:
            ckpt_path = checkpoint_path(ckpt_dir, epoch)
            save_checkpoint(
                ckpt_path,
                epoch=epoch,
                compressor=compressor,
                atom_decoder=atom_decoder,
                lattice_decoder=lattice_decoder,
                optimizer=optimizer,
                train_history=train_history,
            )
            print(f"  saved checkpoint: {ckpt_path}", flush=True)

    test_metrics = run_epoch(
        compressor,
        atom_decoder,
        lattice_decoder,
        test_samples,
        optimizer=None,
        device=device,
        train=False,
        **run_kw,
    )
    print(
        f"\nTest: loss={test_metrics['total']:.4f}  "
        f"atom={test_metrics['atom_total']:.4f}  "
        f"lat={test_metrics['lattice_mse']:.4f}"
    )

    manifest = {
        "data_source": data_source,
        "batch_label": args.batch_label if not args.synthetic else None,
        "uma_in_dir": str(args.uma_in_dir.resolve()) if not args.synthetic else None,
        "lattice_root": str(run_kw["lattice_root"]) if run_kw["lattice_root"] else None,
        "d_model": d_model,
        "epochs": args.epochs,
        "split_seed": args.seed,
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "loss_weights": {
            "w_vae": W_VAE,
            "w_attn": W_ATTN,
            "w_atom": W_ATOM,
            "w_lattice": W_LATTICE,
        },
        "split_manifest": split_manifest_info,
        "test_metrics": test_metrics,
        "checkpoint_dir": str(ckpt_dir),
    }
    manifest_path = ckpt_dir.parent / "train_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {manifest_path}")

    return manifest


def main() -> None:
    args = parse_args()
    train_pipeline(args)


if __name__ == "__main__":
    main()
