#!/usr/bin/env python3
"""Task 05 — Atom attention compress UMA nodes, N in [0, 50) with 70/15/15 split."""
from __future__ import annotations

import sys

from atom_attention_compressor import run_batch

BATCH_LABEL = "N_0000_0050"
N_RANGE = "0 ≤ N < 50"

if __name__ == "__main__":
    print(f"=== UMA atom-attention compress {BATCH_LABEL} ({N_RANGE}) ===")
    run_batch(BATCH_LABEL, sys.argv[1:])
