#!/usr/bin/env python3
"""
CDVAE-style VAE-Attention training (Task 08).

Same attention + VAE backbone as train_pipeline.py, but:
  - N prediction: softmax classification (CrossEntropy), matching aggregator_cdvae.num_atom_loss
  - Lattice: MSE on L_norm (w_lattice=10)

Reads Task 05 *_uma_node.pt and Task 02 normalized lattice labels via uma_train_dataset.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TASK08_DIR = Path(__file__).resolve().parent
TASK07_DIR = TASK08_DIR.parent / "07_aggregator_heads"
if str(TASK07_DIR) not in sys.path:
    sys.path.insert(0, str(TASK07_DIR))

from aggregator_cdvae import LatticeScalerModule  # noqa: E402
from aggregator_data import lattice_scaler_json  # noqa: E402

from uma_train_dataset import (
    DEFAULT_BATCH_LABEL,
    DEFAULT_LATTICE_ROOT,
    DEFAULT_UMA_IN_DIR,
    LATTICE_DIM,
    Sample,
    build_uma_splits,
    load_uma_sample,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **kwargs):  # type: ignore[misc]
        return it


D_MODEL = 128
LATENT_DIM = 64
HIDDEN_DIM = 128

# CDVAE aggregator: fc_num_atoms outputs max_atoms+1 logits; class index = N
N_MIN = 0
N_MAX = 50
NUM_N_CLASSES = N_MAX + 1

W_VAE = 0.01
W_ATTN = 0.1
W_ATOM = 1.0
W_LATTICE = 10.0

EPOCHS = 100
CHECKPOINT_EVERY = 25
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
SPLIT_SEED = 42
DEFAULT_OUT_ROOT = "outputs/cdvae_style_train"
EARLY_STOP_PATIENCE = 10
ATTN_VIZ_SAMPLES = 5
LATTICE_NAMES = ("a", "b", "c", "alpha", "beta", "gamma")


try:
    from torch.utils.tensorboard import SummaryWriter
except (ImportError, AttributeError, ModuleNotFoundError):  # pragma: no cover
    # PyTorch TB can fail if setuptools removed distutils (common on HPC conda envs)
    SummaryWriter = None  # type: ignore[misc, assignment]


def _make_tensorboard_writer(log_dir: Path) -> Any:
    """Return SummaryWriter or None if TensorBoard is unavailable."""
    if SummaryWriter is None:
        return None
    try:
        return SummaryWriter(log_dir=str(log_dir))
    except Exception as exc:  # pragma: no cover
        print(f"warning: TensorBoard disabled ({exc})", flush=True)
        return None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class VAEAtomCompressor(nn.Module):
    """Attention pool + VAE (train_pipeline.py VAEAtomCompressor + compressed output)."""

    def __init__(self, d_model: int = D_MODEL, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.scale = math.sqrt(d_model)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_update = nn.Linear(2 * d_model, d_model, bias=True)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, atom_matrix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if atom_matrix.dim() != 2:
            raise ValueError(f"expected [N, d], got {tuple(atom_matrix.shape)}")

        target = atom_matrix.mean(dim=0, keepdim=True)
        query = self.W_q(target)
        keys = self.W_k(atom_matrix)
        values = self.W_v(atom_matrix)

        scores = (query @ keys.transpose(0, 1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        message = attn_weights @ values

        combined = torch.cat([target, message], dim=-1)
        delta = self.W_update(combined)
        compressed = self.layer_norm(target + delta)

        mu = self.fc_mu(compressed)
        logvar = self.fc_logvar(compressed)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar, attn_weights, compressed


class AtomCountClassifier(nn.Module):
    """
    CDVAE-style atom-count head: logits over N in [0, N_MAX].
    Matches aggregator_cdvae.fc_num_atoms (max_atoms+1 classes, no Softplus).
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_classes: int = NUM_N_CLASSES,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.net(z)

    @torch.no_grad()
    def predict_n(self, z: torch.Tensor) -> int:
        logits = self.forward(z)
        return int(logits.argmax(dim=-1).item())


class LatticeDecoder(nn.Module):
    """Normalized lattice regression head (same as train_pipeline.LatticeDecoder)."""

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        out_dim: int = LATTICE_DIM,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.net(z).squeeze(0)


# ---------------------------------------------------------------------------
# Losses (CDVAE aggregator_cdvae.py)
# ---------------------------------------------------------------------------


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def attention_entropy_loss(attn_weights: torch.Tensor) -> torch.Tensor:
    p = attn_weights.clamp(min=1e-8)
    return -torch.sum(p * torch.log(p))


def atom_count_classification_loss(
    logits: torch.Tensor,
    true_n: torch.Tensor,
    *,
    n_max: int = N_MAX,
) -> torch.Tensor:
    """
    CDVAE num_atom_loss: F.cross_entropy(pred, target_N).
    Class index equals atom count N (aggregator_cdvae.py L162-163).
    """
    # [batch, num_classes] — guard against accidental [1, 1, C] from double unsqueeze
    if logits.dim() > 2:
        logits = logits.reshape(-1, logits.shape[-1])
    elif logits.dim() == 1:
        logits = logits.unsqueeze(0)
    target = true_n.reshape(-1).long().clamp(min=0, max=n_max)
    return F.cross_entropy(logits, target)


def lattice_mse_loss(pred_lattice: torch.Tensor, true_lattice: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_lattice.reshape(-1), true_lattice.reshape(-1))


@dataclass
class LossBreakdown:
    total: torch.Tensor
    vae_kld: torch.Tensor
    attn_entropy: torch.Tensor
    atom_ce: torch.Tensor
    lattice_mse: torch.Tensor

    def as_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().cpu()),
            "vae_kld": float(self.vae_kld.detach().cpu()),
            "attn_entropy": float(self.attn_entropy.detach().cpu()),
            "atom_ce": float(self.atom_ce.detach().cpu()),
            "lattice_mse": float(self.lattice_mse.detach().cpu()),
        }


def compute_multitask_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    attn_weights: torch.Tensor,
    n_logits: torch.Tensor,
    true_n: torch.Tensor,
    pred_lattice: torch.Tensor,
    true_lattice: torch.Tensor,
    *,
    w_vae: float = W_VAE,
    w_attn: float = W_ATTN,
    w_atom: float = W_ATOM,
    w_lattice: float = W_LATTICE,
    ablate_lattice: bool = False,
) -> LossBreakdown:
    vae_kld = kl_divergence(mu, logvar)
    attn_entropy = attention_entropy_loss(attn_weights)
    atom_ce = atom_count_classification_loss(n_logits, true_n)
    lattice_mse = (
        torch.zeros((), device=n_logits.device)
        if ablate_lattice
        else lattice_mse_loss(pred_lattice, true_lattice)
    )

    total = (
        w_vae * vae_kld
        + w_attn * attn_entropy
        + w_atom * atom_ce
        + w_lattice * lattice_mse
    )
    return LossBreakdown(
        total=total,
        vae_kld=vae_kld,
        attn_entropy=attn_entropy,
        atom_ce=atom_ce,
        lattice_mse=lattice_mse,
    )


# ---------------------------------------------------------------------------
# Data iteration
# ---------------------------------------------------------------------------


def iter_samples(
    paths: list[str],
    *,
    lattice_root: Path | None,
    batch_label: str,
) -> Iterator[Sample]:
    for p in paths:
        yield load_uma_sample(Path(p), lattice_root=lattice_root, batch_label=batch_label)


def load_lattice_scaler(lattice_root: Path, batch_label: str) -> LatticeScalerModule | None:
    """Task 02 batch StandardScaler for L_norm <-> L_scaled."""
    path = lattice_scaler_json(lattice_root, batch_label)
    if not path.is_file():
        print(f"warning: lattice scaler not found ({path}); physical lattice metrics skipped")
        return None
    return LatticeScalerModule.from_json(path)


def norm_to_scaled_lattice(
    L_norm: torch.Tensor,
    scaler: LatticeScalerModule,
) -> torch.Tensor:
    x = L_norm.reshape(1, -1) if L_norm.dim() == 1 else L_norm
    x = x.to(device=scaler.means.device, dtype=scaler.means.dtype)
    return scaler.inverse_transform(x).reshape(-1)


def scaled_to_physical_lattice(L_scaled: torch.Tensor, n_atoms: int) -> torch.Tensor:
    """CDVAE scale_length: lengths in Angstrom, angles in degrees."""
    out = L_scaled.reshape(-1).clone()
    scale = float(n_atoms) ** (1.0 / 3.0)
    out[:3] = out[:3] * scale
    return out


def lattice_physical_mae(
    pred_norm: torch.Tensor,
    true_norm: torch.Tensor,
    n_atoms: int,
    scaler: LatticeScalerModule,
) -> dict[str, float]:
    """MAE in physical units (A for lengths, degrees for angles)."""
    pred_phys = scaled_to_physical_lattice(norm_to_scaled_lattice(pred_norm, scaler), n_atoms)
    true_phys = scaled_to_physical_lattice(norm_to_scaled_lattice(true_norm, scaler), n_atoms)
    err = (pred_phys - true_phys).abs()
    return {
        "mean_all": float(err.mean().item()),
        "mean_lengths_A": float(err[:3].mean().item()),
        "mean_angles_deg": float(err[3:].mean().item()),
        **{f"mae_{name}": float(err[i].item()) for i, name in enumerate(LATTICE_NAMES)},
    }


@dataclass
class PredCollector:
    """Accumulate per-sample predictions for detailed validation/test reports."""

    true_ns: list[int] = field(default_factory=list)
    pred_ns: list[int] = field(default_factory=list)
    lattice_norm_mae: list[float] = field(default_factory=list)
    lattice_phys: list[dict[str, float]] = field(default_factory=list)

    def add(
        self,
        true_n: int,
        pred_n: int,
        lattice_err_norm: float,
        lattice_phys: dict[str, float] | None = None,
    ) -> None:
        self.true_ns.append(true_n)
        self.pred_ns.append(pred_n)
        self.lattice_norm_mae.append(lattice_err_norm)
        if lattice_phys is not None:
            self.lattice_phys.append(lattice_phys)

    def per_class_accuracy(self) -> dict[str, dict[str, float]]:
        by_n: dict[int, list[int]] = defaultdict(list)
        for t, p in zip(self.true_ns, self.pred_ns):
            by_n[t].append(1 if t == p else 0)
        out: dict[str, dict[str, float]] = {}
        for n in sorted(by_n):
            hits = by_n[n]
            out[str(n)] = {
                "accuracy": sum(hits) / len(hits),
                "correct": float(sum(hits)),
                "total": float(len(hits)),
            }
        return out

    def summary(self) -> dict[str, Any]:
        n = max(len(self.true_ns), 1)
        errs = [abs(t - p) for t, p in zip(self.true_ns, self.pred_ns)]
        return {
            "n_samples": len(self.true_ns),
            "n_accuracy": sum(1 for e in errs if e == 0) / n,
            "n_within_1": sum(1 for e in errs if e <= 1) / n,
            "n_within_2": sum(1 for e in errs if e <= 2) / n,
            "mean_lattice_norm_mae": float(np.mean(self.lattice_norm_mae)) if self.lattice_norm_mae else 0.0,
            "per_class_accuracy": self.per_class_accuracy(),
            "lattice_physical": _aggregate_lattice_phys(self.lattice_phys),
        }


def _aggregate_lattice_phys(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def confusion_matrix(true_ns: list[int], pred_ns: list[int], n_max: int = N_MAX) -> np.ndarray:
    cm = np.zeros((n_max + 1, n_max + 1), dtype=np.int64)
    for t, p in zip(true_ns, pred_ns):
        if 0 <= t <= n_max and 0 <= p <= n_max:
            cm[t, p] += 1
    return cm


def save_confusion_heatmap(
    cm: np.ndarray,
    path: Path,
    *,
    title: str = "N prediction confusion matrix",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if plt is None:
        np.save(path.with_suffix(".npy"), cm)
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues", aspect="auto", origin="lower")
    ax.set_xlabel("Predicted N")
    ax.set_ylabel("True N")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_attention_examples(
    examples: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ex in examples:
        stem = ex["stem"]
        payload = {
            "stem": stem,
            "true_n": ex["true_n"],
            "pred_n": ex["pred_n"],
            "attn_weights": ex["attn_weights"],
        }
        with (out_dir / f"{stem}_attention.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        if plt is not None and ex["attn_weights"]:
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.bar(range(len(ex["attn_weights"])), ex["attn_weights"])
            ax.set_xlabel("Atom index")
            ax.set_ylabel("Attention weight")
            ax.set_title(f"{stem}  true_N={ex['true_n']} pred_N={ex['pred_n']}")
            fig.tight_layout()
            fig.savefig(out_dir / f"{stem}_attention.png", dpi=120)
            plt.close(fig)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------


@dataclass
class EpochMetrics:
    total: float = 0.0
    vae_kld: float = 0.0
    attn_entropy: float = 0.0
    atom_ce: float = 0.0
    lattice_mse: float = 0.0
    n_samples: int = 0
    n_correct: int = 0
    n_within_1: int = 0
    n_within_2: int = 0
    lattice_abs_sum: float = 0.0

    def update(self, breakdown: LossBreakdown) -> None:
        d = breakdown.as_dict()
        self.total += d["total"]
        self.vae_kld += d["vae_kld"]
        self.attn_entropy += d["attn_entropy"]
        self.atom_ce += d["atom_ce"]
        self.lattice_mse += d["lattice_mse"]
        self.n_samples += 1

    def update_pred(self, pred_n: int, true_n: int, lattice_err: float) -> None:
        err = abs(pred_n - true_n)
        if err == 0:
            self.n_correct += 1
        if err <= 1:
            self.n_within_1 += 1
        if err <= 2:
            self.n_within_2 += 1
        self.lattice_abs_sum += lattice_err

    def average(self) -> dict[str, float]:
        n = max(self.n_samples, 1)
        out = {
            "total": self.total / n,
            "vae_kld": self.vae_kld / n,
            "attn_entropy": self.attn_entropy / n,
            "atom_ce": self.atom_ce / n,
            "lattice_mse": self.lattice_mse / n,
            "n_samples": float(self.n_samples),
        }
        if self.n_samples > 0:
            out["n_accuracy"] = self.n_correct / self.n_samples
            out["n_within_1"] = self.n_within_1 / self.n_samples
            out["n_within_2"] = self.n_within_2 / self.n_samples
            out["mean_lattice_error"] = self.lattice_abs_sum / self.n_samples
        return out


def run_epoch(
    compressor: VAEAtomCompressor,
    n_classifier: AtomCountClassifier,
    lattice_decoder: LatticeDecoder | None,
    paths: list[str],
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
    lattice_root: Path | None,
    batch_label: str,
    ablate_lattice: bool = False,
    lattice_scaler: LatticeScalerModule | None = None,
    collector: PredCollector | None = None,
    attn_examples: list[dict[str, Any]] | None = None,
    attn_limit: int = 0,
) -> dict[str, float]:
    if train:
        compressor.train()
        n_classifier.train()
        if lattice_decoder is not None:
            lattice_decoder.train()
    else:
        compressor.eval()
        n_classifier.eval()
        if lattice_decoder is not None:
            lattice_decoder.eval()

    metrics = EpochMetrics()
    context = torch.enable_grad() if train else torch.no_grad()

    iterator: Any = tqdm(paths, desc="train" if train else "val", leave=False)

    with context:
        for pt_path in iterator:
            try:
                atom_matrix, true_n, true_l = load_uma_sample(
                    Path(pt_path),
                    lattice_root=lattice_root,
                    batch_label=batch_label,
                )
            except Exception as exc:
                print(f"skip {pt_path}: {exc}", flush=True)
                continue

            n_int = int(true_n.item())
            if n_int < N_MIN or n_int > N_MAX:
                continue

            atom_matrix = atom_matrix.to(device)
            true_n = true_n.to(device)
            true_l = true_l.to(device)

            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            z, mu, logvar, attn, _compressed = compressor(atom_matrix)
            z_vec = z.squeeze(0)
            n_logits = n_classifier(z_vec)  # [1, num_classes]
            pred_l = (
                torch.zeros(LATTICE_DIM, device=device)
                if ablate_lattice or lattice_decoder is None
                else lattice_decoder(z_vec)
            )

            breakdown = compute_multitask_loss(
                mu,
                logvar,
                attn,
                n_logits,
                true_n.unsqueeze(0) if true_n.dim() == 0 else true_n.reshape(1),
                pred_l,
                true_l,
                ablate_lattice=ablate_lattice,
            )
            metrics.update(breakdown)

            if not train:
                pred_n = int(n_logits.argmax(dim=-1).item())
                lat_err = (
                    0.0
                    if ablate_lattice
                    else float((pred_l - true_l).abs().mean().item())
                )
                metrics.update_pred(pred_n, n_int, lat_err)

                if collector is not None:
                    phys: dict[str, float] | None = None
                    if (
                        not ablate_lattice
                        and lattice_scaler is not None
                        and lattice_decoder is not None
                    ):
                        phys = lattice_physical_mae(pred_l, true_l, n_int, lattice_scaler)
                    collector.add(n_int, pred_n, lat_err, phys)

                if (
                    attn_examples is not None
                    and attn_limit > 0
                    and len(attn_examples) < attn_limit
                ):
                    stem = Path(pt_path).name.replace("_uma_node.pt", "")
                    attn_examples.append(
                        {
                            "stem": stem,
                            "true_n": n_int,
                            "pred_n": pred_n,
                            "attn_weights": attn.detach().cpu().reshape(-1).tolist(),
                        }
                    )

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
    n_classifier: AtomCountClassifier,
    lattice_decoder: LatticeDecoder | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    history: list[dict[str, Any]],
    best_val_loss: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "compressor_state_dict": compressor.state_dict(),
        "n_classifier_state_dict": n_classifier.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "best_val_loss": best_val_loss,
        "config": {
            "num_n_classes": NUM_N_CLASSES,
            "n_min": N_MIN,
            "n_max": N_MAX,
            "w_vae": W_VAE,
            "w_attn": W_ATTN,
            "w_atom": W_ATOM,
            "w_lattice": W_LATTICE,
            "ablate_lattice": args.ablate_lattice,
        },
    }
    if lattice_decoder is not None:
        payload["lattice_decoder_state_dict"] = lattice_decoder.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    compressor: VAEAtomCompressor,
    n_classifier: AtomCountClassifier,
    lattice_decoder: LatticeDecoder | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    device: torch.device,
) -> tuple[int, list[dict[str, Any]], float]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    compressor.load_state_dict(ckpt["compressor_state_dict"])
    n_classifier.load_state_dict(ckpt["n_classifier_state_dict"])
    if lattice_decoder is not None and "lattice_decoder_state_dict" in ckpt:
        lattice_decoder.load_state_dict(ckpt["lattice_decoder_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    epoch = int(ckpt["epoch"])
    history = list(ckpt.get("history", []))
    best_val = float(ckpt.get("best_val_loss", float("inf")))
    return epoch, history, best_val


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    task_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="CDVAE-style VAE-Attention training (N CE + lattice)")
    p.add_argument(
        "--uma-in-dir",
        type=Path,
        default=Path(os.environ.get("CDVAE_STYLE_UMA_IN_DIR", str(DEFAULT_UMA_IN_DIR))),
    )
    p.add_argument(
        "--lattice-root",
        type=Path,
        default=Path(os.environ.get("CDVAE_STYLE_LATTICE_ROOT", str(DEFAULT_LATTICE_ROOT))),
    )
    p.add_argument(
        "--batch-label",
        default=os.environ.get("CDVAE_STYLE_BATCH_LABEL", DEFAULT_BATCH_LABEL),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("CDVAE_STYLE_OUT_ROOT", DEFAULT_OUT_ROOT)),
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Defaults to <output-dir>/checkpoints",
    )
    p.add_argument("--epochs", type=int, default=int(os.environ.get("CDVAE_STYLE_EPOCHS", str(EPOCHS))))
    p.add_argument("--lr", type=float, default=float(os.environ.get("CDVAE_STYLE_LR", "1e-3")))
    p.add_argument("--seed", type=int, default=int(os.environ.get("CDVAE_STYLE_SEED", str(SPLIT_SEED))))
    p.add_argument(
        "--device",
        default=os.environ.get(
            "CDVAE_STYLE_DEVICE",
            "cuda" if torch.cuda.is_available() else "cpu",
        ),
    )
    p.add_argument("--limit", type=int, default=int(os.environ.get("CDVAE_STYLE_LIMIT", "0")))
    p.add_argument("--resplit", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--resume-path",
        type=Path,
        default=None,
        help="Explicit checkpoint path (overrides --resume latest search)",
    )
    p.add_argument(
        "--ablate-lattice",
        action="store_true",
        help="Train N classifier only (no lattice head loss)",
    )
    p.add_argument(
        "--w-lattice",
        type=float,
        default=float(os.environ.get("CDVAE_STYLE_W_LATTICE", str(W_LATTICE))),
    )
    p.add_argument(
        "--lr-schedule",
        choices=("none", "cosine", "plateau"),
        default=os.environ.get("CDVAE_STYLE_LR_SCHEDULE", "plateau"),
        help="Learning rate schedule (default: ReduceLROnPlateau)",
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=int(os.environ.get("CDVAE_STYLE_EARLY_STOP", str(EARLY_STOP_PATIENCE))),
        help="Stop if val loss does not improve for N epochs (0=disabled)",
    )
    p.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=None,
        help="TensorBoard log directory (default: <output-dir>/tb)",
    )
    p.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging",
    )
    p.add_argument(
        "--regression-ckpt",
        type=Path,
        default=None,
        help="Optional train_pipeline checkpoint for baseline comparison in final report",
    )
    return p.parse_args()


def compare_regression_baseline(
    regression_ckpt: Path,
    compressor: VAEAtomCompressor,
    n_classifier: AtomCountClassifier,
    val_paths: list[str],
    *,
    lattice_root: Path,
    batch_label: str,
    device: torch.device,
) -> dict[str, Any]:
    """Compare classification head vs regression train_pipeline on validation split."""
    from train_pipeline import AtomCountDecoder, VAEAtomCompressor as RegCompressor

    try:
        ckpt = torch.load(regression_ckpt, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(regression_ckpt, map_location=device)

    reg_comp = RegCompressor().to(device)
    reg_dec = AtomCountDecoder().to(device)
    reg_comp.load_state_dict(ckpt["compressor_state_dict"])
    key = "atom_decoder_state_dict" if "atom_decoder_state_dict" in ckpt else "decoder_state_dict"
    reg_dec.load_state_dict(ckpt[key])
    reg_comp.eval()
    reg_dec.eval()
    compressor.eval()
    n_classifier.eval()

    reg_err: list[float] = []
    cls_err: list[float] = []
    cls_hit: list[int] = []

    with torch.no_grad():
        for pt in val_paths:
            atom_matrix, true_n, _ = load_uma_sample(
                Path(pt), lattice_root=lattice_root, batch_label=batch_label
            )
            atom_matrix = atom_matrix.to(device)
            true_i = int(true_n.item())

            z_r, _, _, _ = reg_comp(atom_matrix)
            pred_reg = float(reg_dec(z_r.squeeze(0)).item())
            reg_err.append(abs(pred_reg - true_i))

            z_c, _, _, _, _ = compressor(atom_matrix)
            pred_cls = n_classifier.predict_n(z_c.squeeze(0))
            cls_err.append(abs(pred_cls - true_i))
            cls_hit.append(1 if pred_cls == true_i else 0)

    return {
        "regression_ckpt": str(regression_ckpt),
        "n_val": len(val_paths),
        "regression_mae": float(np.mean(reg_err)) if reg_err else 0.0,
        "regression_exact": sum(1 for e in reg_err if e < 0.5),
        "classification_mae": float(np.mean(cls_err)) if cls_err else 0.0,
        "classification_exact_accuracy": float(np.mean(cls_hit)) if cls_hit else 0.0,
        "classification_within_1": sum(1 for e in cls_err if e <= 1) / max(len(cls_err), 1),
        "classification_within_2": sum(1 for e in cls_err if e <= 2) / max(len(cls_err), 1),
    }


def main() -> None:
    args = parse_args()
    global W_LATTICE
    W_LATTICE = float(args.w_lattice)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = args.output_dir.expanduser().resolve()
    ckpt_dir = (args.checkpoint_dir or out_dir / "checkpoints").expanduser().resolve()
    viz_dir = out_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tb_dir: Path | None = None
    writer: Any = None
    if not args.no_tensorboard:
        tb_dir = (args.tensorboard_dir or out_dir / "tb").expanduser().resolve()
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = _make_tensorboard_writer(tb_dir)
        if writer is None:
            print("TensorBoard logging skipped (package unavailable)", flush=True)

    uma_in = args.uma_in_dir.expanduser().resolve()
    lattice_root = args.lattice_root.expanduser().resolve()
    lattice_scaler = load_lattice_scaler(lattice_root, args.batch_label)

    print("=== CDVAE-style VAE-Attention training ===")
    print(f"device={device}  epochs={args.epochs}  lr={args.lr}  lr_schedule={args.lr_schedule}")
    print(
        f"weights: w_vae={W_VAE} w_attn={W_ATTN} w_atom={W_ATOM} w_lattice={W_LATTICE} "
        f"ablate_lattice={args.ablate_lattice}"
    )
    print(f"N classes: {NUM_N_CLASSES} (N in [{N_MIN}, {N_MAX}])")
    print(f"early_stop_patience={args.early_stop_patience}")
    print(f"uma_in: {uma_in}")
    print(f"lattice_root: {lattice_root}")
    if tb_dir:
        print(f"tensorboard: {tb_dir}")

    train_paths, val_paths, test_paths, split_manifest, d_model = build_uma_splits(
        uma_in_dir=uma_in,
        batch_label=args.batch_label,
        split_seed=args.seed,
        train_frac=TRAIN_FRAC,
        val_frac=VAL_FRAC,
        resplit=args.resplit,
        limit=args.limit,
        lattice_root=lattice_root,
    )
    counts = split_manifest.counts()
    print(
        f"split: train={counts['train']} val={counts['val']} test={counts['test']} d_model={d_model}"
    )

    compressor = VAEAtomCompressor(d_model=d_model, latent_dim=LATENT_DIM).to(device)
    n_classifier = AtomCountClassifier(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM).to(device)
    lattice_decoder: LatticeDecoder | None = None
    if not args.ablate_lattice:
        lattice_decoder = LatticeDecoder(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM).to(device)

    params = list(compressor.parameters()) + list(n_classifier.parameters())
    if lattice_decoder is not None:
        params += list(lattice_decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.01
        )
    elif args.lr_schedule == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=args.lr * 1e-3
        )

    start_epoch = 0
    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improve = 0

    resume_path = args.resume_path
    if resume_path is None and args.resume:
        resume_path = find_latest_checkpoint(ckpt_dir)
    if resume_path is not None and Path(resume_path).is_file():
        start_epoch, history, best_val_loss = load_checkpoint(
            Path(resume_path),
            compressor,
            n_classifier,
            lattice_decoder,
            optimizer,
            scheduler,
            device,
        )
        if history:
            best_epoch = max(
                (int(h["epoch"]) for h in history if h.get("val", {}).get("total", float("inf")) <= best_val_loss),
                default=start_epoch,
            )
        print(f"Resumed from {resume_path} at epoch {start_epoch}, best_val={best_val_loss:.4f}")
    else:
        print("Training from scratch.")

    run_kw: dict[str, Any] = {
        "lattice_root": lattice_root,
        "batch_label": args.batch_label,
        "ablate_lattice": args.ablate_lattice,
        "lattice_scaler": lattice_scaler,
    }

    stopped_early = False
    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_m = run_epoch(
            compressor,
            n_classifier,
            lattice_decoder,
            train_paths,
            optimizer=optimizer,
            device=device,
            train=True,
            **run_kw,
        )
        val_m = run_epoch(
            compressor,
            n_classifier,
            lattice_decoder,
            val_paths,
            optimizer=None,
            device=device,
            train=False,
            **run_kw,
        )

        if scheduler is not None:
            if args.lr_schedule == "plateau":
                scheduler.step(val_m["total"])
            else:
                scheduler.step()

        record = {"epoch": epoch, "train": train_m, "val": val_m, "lr": optimizer.param_groups[0]["lr"]}
        history.append(record)

        if writer is not None:
            for split_name, m in (("train", train_m), ("val", val_m)):
                for k, v in m.items():
                    writer.add_scalar(f"{split_name}/{k}", v, epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"epoch {epoch:03d}/{args.epochs}  lr={optimizer.param_groups[0]['lr']:.2e}  "
                f"train_loss={train_m['total']:.4f}  val_loss={val_m['total']:.4f}  "
                f"train_atom_ce={train_m.get('atom_ce', 0):.4f}  "
                f"val_atom_ce={val_m.get('atom_ce', 0):.4f}  "
                f"val_N_acc={val_m.get('n_accuracy', 0):.4f}  "
                f"val_lat={val_m.get('lattice_mse', 0):.4f}",
                flush=True,
            )

        if epoch % CHECKPOINT_EVERY == 0 or epoch == args.epochs:
            ckpt_payload_path = checkpoint_path(ckpt_dir, epoch)
            save_checkpoint(
                ckpt_payload_path,
                epoch=epoch,
                compressor=compressor,
                n_classifier=n_classifier,
                lattice_decoder=lattice_decoder,
                optimizer=optimizer,
                scheduler=scheduler,
                history=history,
                best_val_loss=best_val_loss,
                args=args,
            )
            print(f"  saved {ckpt_payload_path.name}", flush=True)

        if val_m["total"] < best_val_loss:
            best_val_loss = val_m["total"]
            best_epoch = epoch
            epochs_without_improve = 0
            best_path = ckpt_dir / "best_model.pt"
            save_checkpoint(
                best_path,
                epoch=epoch,
                compressor=compressor,
                n_classifier=n_classifier,
                lattice_decoder=lattice_decoder,
                optimizer=optimizer,
                scheduler=scheduler,
                history=history,
                best_val_loss=best_val_loss,
                args=args,
            )
            print(f"  new best val_loss={best_val_loss:.4f} -> {best_path.name}", flush=True)
        else:
            epochs_without_improve += 1

        if args.early_stop_patience > 0 and epochs_without_improve >= args.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch} "
                f"(no val improvement for {args.early_stop_patience} epochs)",
                flush=True,
            )
            stopped_early = True
            break

    # Reload best weights for final evaluation
    best_path = ckpt_dir / "best_model.pt"
    if best_path.is_file():
        load_checkpoint(
            best_path,
            compressor,
            n_classifier,
            lattice_decoder,
            optimizer,
            scheduler,
            device,
        )
        print(f"Loaded best checkpoint (epoch {best_epoch}) for final evaluation")

    val_collector = PredCollector()
    attn_examples: list[dict[str, Any]] = []
    val_m_detailed = run_epoch(
        compressor,
        n_classifier,
        lattice_decoder,
        val_paths,
        optimizer=None,
        device=device,
        train=False,
        collector=val_collector,
        attn_examples=attn_examples,
        attn_limit=ATTN_VIZ_SAMPLES,
        **run_kw,
    )

    test_collector = PredCollector()
    test_m = run_epoch(
        compressor,
        n_classifier,
        lattice_decoder,
        test_paths,
        optimizer=None,
        device=device,
        train=False,
        collector=test_collector,
        **run_kw,
    )

    val_summary = val_collector.summary()
    test_summary = test_collector.summary()
    cm_val = confusion_matrix(val_collector.true_ns, val_collector.pred_ns)
    cm_test = confusion_matrix(test_collector.true_ns, test_collector.pred_ns)

    save_confusion_heatmap(cm_val, viz_dir / "confusion_matrix_val.png", title="Validation N confusion")
    save_confusion_heatmap(cm_test, viz_dir / "confusion_matrix_test.png", title="Test N confusion")
    np.save(viz_dir / "confusion_matrix_val.npy", cm_val)
    np.save(viz_dir / "confusion_matrix_test.npy", cm_test)
    save_attention_examples(attn_examples, viz_dir / "attention")

    baseline_cmp: dict[str, Any] | None = None
    if args.regression_ckpt is not None and args.regression_ckpt.is_file():
        print("Comparing with regression baseline...")
        baseline_cmp = compare_regression_baseline(
            args.regression_ckpt,
            compressor,
            n_classifier,
            val_paths,
            lattice_root=lattice_root,
            batch_label=args.batch_label,
            device=device,
        )

    report: dict[str, Any] = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "stopped_early": stopped_early,
        "val_metrics": val_m_detailed,
        "val_detailed": val_summary,
        "test_metrics": test_m,
        "test_detailed": test_summary,
        "split": split_manifest.to_dict(),
        "config": {
            "num_n_classes": NUM_N_CLASSES,
            "w_lattice": W_LATTICE,
            "ablate_lattice": args.ablate_lattice,
            "lr_schedule": args.lr_schedule,
            "early_stop_patience": args.early_stop_patience,
        },
        "baseline_comparison": baseline_cmp,
        "viz": {
            "confusion_val": str(viz_dir / "confusion_matrix_val.png"),
            "confusion_test": str(viz_dir / "confusion_matrix_test.png"),
            "attention_dir": str(viz_dir / "attention"),
        },
    }

    history_path = out_dir / "history.json"
    report_path = out_dir / "final_report.json"
    per_class_path = out_dir / "per_class_accuracy.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with per_class_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "validation": val_summary.get("per_class_accuracy", {}),
                "test": test_summary.get("per_class_accuracy", {}),
            },
            f,
            indent=2,
        )

    print("\n=== Final results (best model) ===")
    print(f"Best epoch: {best_epoch}  val_loss={best_val_loss:.4f}")
    print(
        f"Val  N_acc={val_summary.get('n_accuracy', 0):.4f}  "
        f"within±1={val_summary.get('n_within_1', 0):.4f}  "
        f"within±2={val_summary.get('n_within_2', 0):.4f}"
    )
    print(f"Test N_acc={test_summary.get('n_accuracy', 0):.4f}")
    if val_summary.get("lattice_physical"):
        lp = val_summary["lattice_physical"]
        print(
            f"Val lattice physical MAE: lengths={lp.get('mean_lengths_A', 0):.4f} A  "
            f"angles={lp.get('mean_angles_deg', 0):.4f} deg"
        )
    if baseline_cmp:
        print(
            f"Baseline regression MAE={baseline_cmp['regression_mae']:.3f} vs "
            f"classification MAE={baseline_cmp['classification_mae']:.3f}"
        )
    print(f"history: {history_path}")
    print(f"report:  {report_path}")
    print(f"per_class: {per_class_path}")
    if tb_dir:
        print(f"tensorboard: tensorboard --logdir {tb_dir}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
