#!/usr/bin/env python3
"""
Extract M = (N, A, F, L, composition) for every CIF in MOF_database.

Replica of tasks/02_structure_M/structure_M_all.py — organized under pipelines/structure_M_gemmi/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

PIPELINE_DIR = Path(__file__).resolve().parent
CIF_READ_DIR = PIPELINE_DIR.parent / "cif_read_gemmi"
for _p in (str(CIF_READ_DIR), str(PIPELINE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gemmi_cif_utils import DEFAULT_CIF_DIR, find_cif_files, read_cif  # noqa: E402
from structure_M_utils import (  # noqa: E402
    check_M,
    load_M,
    save_M,
    structure_to_M,
)

DEFAULT_OUTPUT_DIR = PIPELINE_DIR / "outputs"
DEFAULT_NPZ_DIR = DEFAULT_OUTPUT_DIR / "all_npz"
PROGRESS_EVERY = 1000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract structure M for all CIF files and write .npz + manifest."
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
        help="Root output directory (manifest + jsonl + all_npz/)",
    )
    p.add_argument(
        "--npz-dir",
        type=Path,
        default=None,
        help="Directory for per-structure .npz files (default: <output-dir>/all_npz)",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=int(os.environ.get("STRUCTURE_M_PROGRESS_EVERY", PROGRESS_EVERY)),
        help="Print progress every N files",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cif_dir = args.cif_dir.resolve()
    output_dir = args.output_dir.resolve()
    npz_dir = (args.npz_dir or (output_dir / "all_npz")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "structure_M_all_manifest.json"
    records_path = output_dir / "structure_M_all_records.jsonl"
    error_log_path = output_dir / "structure_M_all_errors.log"

    print(f"CIF directory: {cif_dir}")
    all_cifs = find_cif_files(cif_dir)
    total = len(all_cifs)
    print(f"Extracting M for all {total} CIF files (sorted order, no random sample).")
    print(f"NPZ dir  : {npz_dir}")
    print(f"Manifest : {manifest_path}")
    print(f"Records  : {records_path}")
    print(f"Error log: {error_log_path}")

    t0 = time.perf_counter()
    n_ok = 0
    errors: List[dict] = []
    first_npz: Path | None = None

    with error_log_path.open("w", encoding="utf-8") as err_fh, records_path.open(
        "w", encoding="utf-8"
    ) as rec_fh:
        for i, cif_path in enumerate(all_cifs, start=1):
            try:
                structure = read_cif(cif_path)
                m = structure_to_M(structure)
                check_M(m)
                out_path = save_M(cif_path, m, npz_dir)
                if first_npz is None:
                    first_npz = out_path
                record = {
                    "path": str(cif_path),
                    "npz": str(out_path),
                    "N": m.N,
                    "composition": m.composition,
                }
                rec_fh.write(json.dumps(record) + "\n")
                n_ok += 1
                if i % args.progress_every == 0 or i == total:
                    elapsed = time.perf_counter() - t0
                    rate = i / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[{i}/{total}] OK  N={m.N:4d}  {m.composition:<20s}  "
                        f"{cif_path.name}  ({rate:.1f} files/s)"
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
        "npz_dir": str(npz_dir),
        "total_files": total,
        "ok": n_ok,
        "failed": len(errors),
        "elapsed_s": elapsed,
        "records_jsonl": str(records_path),
        "errors": errors,
    }
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    if first_npz is not None:
        m_reload = load_M(first_npz)
        check_M(m_reload)
        print(
            f"Reload check: {first_npz.name} OK — "
            f"N={m_reload.N}, composition={m_reload.composition}"
        )

    print("=" * 72)
    print(f"Done. OK={n_ok}  failed={len(errors)}  elapsed={elapsed:.1f}s")
    print(f"Manifest written: {manifest_path}")
    print("structure_M_all done.")


if __name__ == "__main__":
    main()
