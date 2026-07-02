#!/usr/bin/env python3
"""
MLP head: latent z -> normalized lattice L_norm in R^6.

Loss = MSE(pred_L, true_L).

Extracted from train_pipeline.py (LatticeDecoder + lattice_loss).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn

from uma_train_dataset import LATTICE_DIM

DEFAULT_LATENT_DIM = 128
DEFAULT_HIDDEN_DIM = 128


class LatticeDecoder(nn.Module):
    """MLP: z [latent_dim] -> L_norm [6] (no output activation)."""

    def __init__(
        self,
        latent_dim: int = DEFAULT_LATENT_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
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


@dataclass
class LatticeLossBreakdown:
    mse: torch.Tensor

    def as_dict(self) -> dict[str, float]:
        return {"lattice_mse": float(self.mse.detach().cpu())}


def lattice_loss(
    pred_L_norm: torch.Tensor,
    true_L_norm: torch.Tensor,
    *,
    mse_fn: nn.MSELoss | None = None,
) -> LatticeLossBreakdown:
    """MSE in normalized lattice space (same as aggregator_cdvae.lattice_loss)."""
    if mse_fn is None:
        mse_fn = nn.MSELoss()
    pred = pred_L_norm.reshape(-1).float()
    target = true_L_norm.reshape(-1).float()
    mse = mse_fn(pred, target)
    return LatticeLossBreakdown(mse=mse)


def _smoke_test(device: torch.device) -> None:
    decoder = LatticeDecoder().to(device)
    z = torch.randn(128, device=device)
    true_l = torch.randn(LATTICE_DIM, device=device) * 0.5
    pred_l = decoder(z)
    breakdown = lattice_loss(pred_l, true_l)
    print("=== mlp_head_l smoke test ===")
    print(f"z:       {tuple(z.shape)}")
    print(f"pred L:  {tuple(pred_l.shape)}")
    print(f"true L:  {tuple(true_l.shape)}")
    print(f"loss:    {breakdown.as_dict()}")


def main() -> None:
    p = argparse.ArgumentParser(description="Lattice MLP head smoke test")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    _smoke_test(torch.device(args.device))


if __name__ == "__main__":
    main()
