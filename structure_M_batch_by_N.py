#!/usr/bin/env python3
"""
Group Task 02 all_npz outputs into batches by atom count N (bin width 50).

Default bins (aligned to multiples of 50):
  N in [50, 100)  -> batch N_0050_0100
  N in [100, 150) -> batch N_0100_0150
  ...

Reads only the N field from each .npz (fast scan). Writes:
  outputs/batches_by_N/manifest.json
  outputs/batches_by_N/batch_N_XXXX_YYYY.jsonl  (one path per line)
  outputs/batches_by_N/batch_N_XXXX_YYYY/         (symlinks to npz files)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

TASK_DIR = Path(__file__).resolve().parent
DEFAULT_NPZ_DIR = TASK_DIR / "outputs" / "all_npz"
DEFAULT_OUT_DIR = TASK_DIR / "outputs" / "batches_by_N"
DEFAULT_BIN_WIDTH = 50
PROGRESS_EVERY = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distribute structure_M .npz files into N-atom bins (width 50)."
    )
    parser.add_argument(
        "--npz-dir",
        type=Path,
        default=Path(os.environ.get("STRUCTURE_M_NPZ_DIR", str(DEFAULT_NPZ_DIR))),
        help=f"Directory of .npz files (default: {DEFAULT_NPZ_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.environ.get("STRUCTURE_M_BATCH_DIR", str(DEFAULT_OUT_DIR))),
        help=f"Output directory for manifests and batch folders (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--bin-width",
        type=int,
        default=int(os.environ.get("STRUCTURE_M_BIN_WIDTH", DEFAULT_BIN_WIDTH)),
        help=f"Atoms per bin (default: {DEFAULT_BIN_WIDTH})",
    )
    parser.add_argument(
        "--start-at-min",
        action="store_true",
        help="First bin starts at min(N) instead of aligning to bin-width (e.g. 53-103, 103-153).",
    )
    parser.add_argument(
        "--no-symlinks",
        action="store_true",
        help="Only write jsonl lists; do not create per-batch symlink folders.",
    )
    return parser.parse_args()


def iter_npz_files(npz_dir: Path) -> Iterable[Path]:
    if not npz_dir.is_dir():
        raise FileNotFoundError(f"NPZ directory not found: {npz_dir}")
    yield from sorted(npz_dir.glob("*.npz"))


def read_N(npz_path: Path) -> int:
    with np.load(npz_path, allow_pickle=False) as data:
        if "N" not in data:
            raise KeyError(f"missing 'N' in {npz_path}")
        return int(data["N"])


def bin_range(N: int, width: int, start_at_min: bool, min_N: int) -> tuple[int, int]:
    """Return [low, high) bin for atom count N."""
    if start_at_min:
        if N < min_N:
            raise ValueError(f"N={N} < min_N={min_N}")
        offset = N - min_N
        low = min_N + (offset // width) * width
    else:
        low = (N // width) * width
    return low, low + width


def bin_label(low: int, high: int) -> str:
    return f"N_{low:04d}_{high:04d}"


def scan_npz_dir(npz_dir: Path) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    errors: list[dict] = []
    files = list(iter_npz_files(npz_dir))
    total = len(files)
    if total == 0:
        raise FileNotFoundError(f"No .npz files in {npz_dir}")

    t0 = time.perf_counter()
    for i, npz_path in enumerate(files, start=1):
        try:
            n_atoms = read_N(npz_path)
            records.append({"npz": str(npz_path.resolve()), "N": n_atoms})
        except Exception as exc:
            errors.append(
                {"npz": str(npz_path), "error": f"{type(exc).__name__}: {exc}"}
            )
        if i % PROGRESS_EVERY == 0 or i == total:
            elapsed = time.perf_counter() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            print(f"Scanned [{i}/{total}]  ({rate:.0f} files/s)")

    return records, errors


def assign_batches(
    records: list[dict], width: int, start_at_min: bool
) -> dict[tuple[int, int], list[dict]]:
    min_N = min(r["N"] for r in records)
    max_N = max(r["N"] for r in records)
    batches: dict[tuple[int, int], list[dict]] = defaultdict(list)

    for rec in records:
        low, high = bin_range(rec["N"], width, start_at_min, min_N)
        batches[(low, high)].append(rec)

    # Stable order: increasing N bins, then sorted paths within bin
    ordered: dict[tuple[int, int], list[dict]] = {}
    for key in sorted(batches.keys()):
        ordered[key] = sorted(batches[key], key=lambda r: (r["N"], r["npz"]))
    return ordered


def write_batch_jsonl(path: Path, items: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item) + "\n")


def write_symlinks(batch_dir: Path, items: list[dict]) -> None:
    batch_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        src = Path(item["npz"])
        dst = batch_dir / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)


def main() -> None:
    args = parse_args()
    width = args.bin_width
    if width <= 0:
        raise ValueError(f"--bin-width must be > 0, got {width}")

    npz_dir = args.npz_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"NPZ dir     : {npz_dir}")
    print(f"Output dir  : {out_dir}")
    print(f"Bin width   : {width}")
    print(f"Align bins  : {'start at min(N)' if args.start_at_min else f'multiples of {width}'}")
    print(f"Symlinks    : {not args.no_symlinks}")

    t0 = time.perf_counter()
    records, errors = scan_npz_dir(npz_dir)
    if not records:
        raise RuntimeError("No valid .npz records found.")

    min_N = min(r["N"] for r in records)
    max_N = max(r["N"] for r in records)
    batches = assign_batches(records, width, args.start_at_min)

    batch_summaries: list[dict] = []
    for (low, high), items in batches.items():
        label = bin_label(low, high)
        jsonl_path = out_dir / f"batch_{label}.jsonl"
        write_batch_jsonl(jsonl_path, items)

        batch_folder = out_dir / f"batch_{label}"
        if not args.no_symlinks:
            write_symlinks(batch_folder, items)

        batch_summaries.append(
            {
                "label": label,
                "N_low_inclusive": low,
                "N_high_exclusive": high,
                "count": len(items),
                "jsonl": str(jsonl_path),
                "folder": str(batch_folder) if not args.no_symlinks else None,
                "N_min_in_batch": min(r["N"] for r in items),
                "N_max_in_batch": max(r["N"] for r in items),
            }
        )
        print(
            f"  batch {label}: {len(items):6d} files  "
            f"(N {low}-{high}, actual {batch_summaries[-1]['N_min_in_batch']}-{batch_summaries[-1]['N_max_in_batch']})"
        )

    elapsed = time.perf_counter() - t0
    manifest = {
        "npz_dir": str(npz_dir),
        "out_dir": str(out_dir),
        "bin_width": width,
        "start_at_min": args.start_at_min,
        "total_npz_scanned": len(records) + len(errors),
        "ok": len(records),
        "failed": len(errors),
        "N_min": min_N,
        "N_max": max_N,
        "num_batches": len(batch_summaries),
        "batches": batch_summaries,
        "errors": errors,
        "elapsed_s": elapsed,
    }
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print("=" * 72)
    print(f"Done. {len(records)} npz -> {len(batch_summaries)} batches  ({elapsed:.1f}s)")
    print(f"N range: {min_N} .. {max_N}")
    print(f"Manifest: {manifest_path}")
    if errors:
        err_path = out_dir / "scan_errors.json"
        with err_path.open("w", encoding="utf-8") as fh:
            json.dump(errors, fh, indent=2)
        print(f"Scan errors ({len(errors)}): {err_path}")


if __name__ == "__main__":
    main()
