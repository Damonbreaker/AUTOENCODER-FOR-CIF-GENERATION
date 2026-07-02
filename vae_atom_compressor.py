#!/usr/bin/env python3
"""
Trainable attention compression + VAE reparameterization.

AtomAttentionCompressor: (N, d) -> (1, d) via single-head attention (pure PyTorch).
VAEAtomCompressor chains it with VAEReparameterizeHead.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vae_reparameterize import VAEReparameterizeHead


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


class VAEAtomCompressor(nn.Module):
    """
    Attention-based matrix compression with VAE reparameterization.

    Input:  atom_matrix [N, d_model]
    Output: z [1, latent_dim], mu, logvar, attn_weights [1, N]
    """

    def __init__(self, d_model: int, latent_dim: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.attn_compressor = AtomAttentionCompressor(d_model)
        self.vae_head = VAEReparameterizeHead(d_model, latent_dim)

    def forward(
        self,
        atom_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if atom_matrix.dim() != 2:
            raise ValueError(f"expected [N, d], got {tuple(atom_matrix.shape)}")
        if atom_matrix.size(-1) != self.d_model:
            raise ValueError(
                f"d_model={self.d_model}, got feature dim {atom_matrix.size(-1)}"
            )

        compressed, attn_weights = self.attn_compressor(
            atom_matrix,
            return_attention=True,
        )
        z, mu, logvar = self.vae_head(compressed)
        return z, mu, logvar, attn_weights
