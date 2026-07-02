#!/usr/bin/env python3
"""
MLP head: latent z -> positive atom count N (Poisson rate).

Loss = MSE(rate, N) + PoissonNLL(rate, N)  (sum, not either/or).

Extracted from train_pipeline.py (AtomCountDecoder + atom loss terms).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn

DEFAULT_LATENT_DIM = 128
DEFAULT_HIDDEN_DIM = 128


class AtomCountDecoder(nn.Module):
    """MLP: z [latent_dim] -> positive scalar rate (Softplus)."""

    def __init__(
        self,
        latent_dim: int = DEFAULT_LATENT_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.net(z).squeeze(-1)


@dataclass
class AtomCountLossBreakdown:
    total: torch.Tensor
    mse: torch.Tensor
    poisson: torch.Tensor

    def as_dict(self) -> dict[str, float]:
        return {
            "atom_total": float(self.total.detach().cpu()),
            "atom_mse": float(self.mse.detach().cpu()),
            "atom_poisson": float(self.poisson.detach().cpu()),
        }


def atom_count_loss(
    pred_n: torch.Tensor,
    true_n: torch.Tensor,
    *,
    mse_fn: nn.MSELoss | None = None,
    poisson_fn: nn.PoissonNLLLoss | None = None,
) -> AtomCountLossBreakdown:
    """
    Atom-count loss: MSE + PoissonNLL on rate vs true N.

    pred_n is the Softplus rate lambda (not log-rate).
    """
    if mse_fn is None:
        mse_fn = nn.MSELoss()
    if poisson_fn is None:
        poisson_fn = nn.PoissonNLLLoss(log_input=False, full=False)

    target = true_n.reshape_as(pred_n).float()
    rate = pred_n.reshape_as(target)
    mse = mse_fn(rate, target)
    poisson = poisson_fn(rate, target)
    total = mse + poisson
    return AtomCountLossBreakdown(total=total, mse=mse, poisson=poisson)


def _smoke_test(device: torch.device) -> None:
    decoder = AtomCountDecoder().to(device)
    z = torch.randn(128, device=device)
    true_n = torch.tensor(32.0, device=device)
    rate = decoder(z)
    breakdown = atom_count_loss(rate, true_n)
    print("=== mlp_head_n smoke test ===")
    print(f"z:      {tuple(z.shape)}")
    print(f"rate:   {rate.item():.4f}")
    print(f"true N: {true_n.item():.1f}")
    print(f"loss:   {breakdown.as_dict()}")


def main() -> None:
    p = argparse.ArgumentParser(description="Atom count MLP head smoke test")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    _smoke_test(torch.device(args.device))


if __name__ == "__main__":
    main()
