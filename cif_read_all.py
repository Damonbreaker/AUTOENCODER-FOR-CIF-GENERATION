#!/usr/bin/env python3
"""
Read every .cif in MOF_database with GEMMI and log summaries.

Replica of tasks/01_cif_read/cif_read_all.py — organized under pipelines/cif_read_gemmi/.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List

from gemmi_cif_utils import (
    DEFAULT_CIF_DIR,
    find_cif_files,
    read_cif,
    structure_summary_dict,
)

PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PIPELINE_DIR / "outputs"
PROGRESS_EVERY = 1000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Read all CIF files with GEMMI and write manifest + JSONL records."
    )
    p.add_argument(
        "--cif-dir",
        type=Path,
        default=Path(os.environ.get("CIF_DIR", str(DEFAULT_CIF_DIR))),
        help="Root directory to search recursively for .cif files",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))),
        help="Directory for manifest, records, and error log",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=int(os.environ.get("CIF_READ_PROGRESS_EVERY", PROGRESS_EVERY)),
        help="Print progress every N files",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cif_dir = args.cif_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "cif_read_all_manifest.json"
    records_path = output_dir / "cif_read_all_records.jsonl"
    error_log_path = output_dir / "cif_read_all_errors.log"

    print(f"CIF directory: {cif_dir}")
    all_cifs = find_cif_files(cif_dir)
    total = len(all_cifs)
    print(f"Reading all {total} CIF files (sorted order, no random sample).")
    print(f"Manifest : {manifest_path}")
    print(f"Records  : {records_path}")
    print(f"Error log: {error_log_path}")

    t0 = time.perf_counter()
    n_ok = 0
    errors: List[dict] = []

    with error_log_path.open("w", encoding="utf-8") as err_fh, records_path.open(
        "w", encoding="utf-8"
    ) as rec_fh:
        for i, cif_path in enumerate(all_cifs, start=1):
            try:
                structure = read_cif(cif_path)
                record = structure_summary_dict(cif_path, structure)
                rec_fh.write(json.dumps(record) + "\n")
                n_ok += 1
                if i % args.progress_every == 0 or i == total:
                    elapsed = time.perf_counter() - t0
                    rate = i / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[{i}/{total}] OK  N={record['N']:4d}  "
                        f"{record['formula']:<20s}  {cif_path.name}  "
                        f"({rate:.1f} files/s)"
                    )
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                errors.append({"path": str(cif_path), "error": msg})
                err_fh.write(f"{cif_path}\t{msg}\n")
                if i % args.progress_every == 0 or i == total:
                    print(f"[{i}/{total}] ERR {cif_path.name}  {msg}")

    elapsed = time.perf_counter() - t0
    manifest = {
        "cif_dir": str(cif_dir),
        "total_files": total,
        "ok": n_ok,
        "failed": len(errors),
        "elapsed_s": elapsed,
        "records_jsonl": str(records_path),
        "errors": errors,
    }
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print("=" * 72)
    print(f"Done. OK={n_ok}  failed={len(errors)}  elapsed={elapsed:.1f}s")
    print(f"Manifest written: {manifest_path}")
    print("cif_read_all done.")


if __name__ == "__main__":
    main()
