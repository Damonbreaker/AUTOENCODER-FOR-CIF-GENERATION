#!/usr/bin/env python3
"""
Gaussian VAE head: compressed crystal vector -> mu, logvar, z (reparameterization).

Extracted from train_pipeline.py (VAEAtomCompressor VAE layers).
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Reparameterization trick: z = mu + eps * exp(0.5 * logvar)."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Standard Gaussian KL, summed over latent dimensions."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class VAEReparameterizeHead(nn.Module):
    """
    Map a compressed representation [B, d_in] to stochastic latent z [B, latent_dim].

    Input is typically the attention-compressed vector from AtomAttentionCompressor.
    """

    def __init__(self, d_in: int, latent_dim: int) -> None:
        super().__init__()
        self.d_in = d_in
        self.latent_dim = latent_dim
        self.fc_mu = nn.Linear(d_in, latent_dim)
        self.fc_logvar = nn.Linear(d_in, latent_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.fc_mu(x), self.fc_logvar(x)

    def forward(
        self,
        x: torch.Tensor,
        *,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            z:      [B, latent_dim]
            mu:     [B, latent_dim]
            logvar: [B, latent_dim]
        """
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar) if sample else mu
        return z, mu, logvar


def _smoke_test(device: torch.device) -> None:
    d_in, latent_dim = 128, 64
    head = VAEReparameterizeHead(d_in, latent_dim).to(device)
    compressed = torch.randn(1, d_in, device=device)
    z, mu, logvar = head(compressed)
    kld = kl_divergence(mu, logvar)
    print("=== vae_reparameterize smoke test ===")
    print(f"compressed: {tuple(compressed.shape)}")
    print(f"z:        {tuple(z.shape)}")
    print(f"mu:       {tuple(mu.shape)}")
    print(f"logvar:   {tuple(logvar.shape)}")
    print(f"KL:       {kld.item():.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description="VAE reparameterization head smoke test")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    _smoke_test(torch.device(args.device))


if __name__ == "__main__":
    main()
