"""
Normalize lattice parameters for all Task 02 .npz structure files.

Reads:  N, A, F, L, composition from structure_M .npz
        (original Task 02 OR pipelines/structure_M_gemmi — same format)

Writes: per-structure .npz with L_scaled_N and L_scaled_V + manifest/jsonl

Methods (lengths only; angles unchanged):
  scale_N: a,b,c / N^(1/3)           (CDVAE scale_length)
  scale_V: a,b,c / V^(1/3)           V = triclinic cell volume

Windows — original Task 02 output on PRAGYA or copied locally:
  py -3 normalize_lattice_npz.py --task2
  py -3 normalize_lattice_npz.py ..\\..\\tasks\\02_structure_M\\outputs\\all_npz

Other:
  py -3 normalize_lattice_npz.py --search
  py -3 normalize_lattice_npz.py path\\to\\file.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parent
PIPELINES_ROOT = PIPELINE_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from lattice_norm_utils import (  # noqa: E402
    L_to_dict,
    normalize_lattice_both,
    triclinic_volume,
)

STEPWISE_ROOT = PIPELINES_ROOT.parent
TASK02_NPZ_DIR = STEPWISE_ROOT / "tasks" / "02_structure_M" / "outputs" / "all_npz"

DEFAULT_NPZ_DIRS = [
    TASK02_NPZ_DIR,
    PIPELINES_ROOT / "structure_M_gemmi" / "outputs" / "all_npz",
    PIPELINES_ROOT,
]
DEFAULT_OUT_DIR = PIPELINE_DIR / "outputs" / "normalized_lattice"
PROGRESS_EVERY = 100
SKIP_NPZ_SUFFIXES = ("_lattice_norm.npz",)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Normalize lattice L in structure-M .npz files (Task 02 output)."
    )
    p.add_argument("path", nargs="?", help=".npz file or folder of .npz files")
    p.add_argument(
        "--task2",
        action="store_true",
        help="Use original Task 02 output: tasks/02_structure_M/outputs/all_npz/",
    )
    p.add_argument(
        "--search",
        action="store_true",
        help="Auto-find structure .npz (Task 02 all_npz, then pipelines)",
    )
    p.add_argument(
        "--npz-dir",
        type=Path,
        default=None,
        help="Explicit input folder (overrides --task2). Env: STRUCTURE_M_NPZ_DIR",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.environ.get("LATTICE_NORM_OUT_DIR", str(DEFAULT_OUT_DIR))),
        help="Output directory for normalized .npz + manifest",
    )
    p.add_argument("--limit", type=int, default=0, help="Max files (0 = all)")
    p.add_argument(
        "--progress-every",
        type=int,
        default=int(os.environ.get("LATTICE_NORM_PROGRESS_EVERY", PROGRESS_EVERY)),
    )
    p.add_argument(
        "--no-write-npz",
        action="store_true",
        help="Only write manifest/jsonl/csv (no per-structure output .npz)",
    )
    return p.parse_args()


def _is_structure_npz(path: Path) -> bool:
    name = path.name
    return name.endswith(".npz") and not any(name.endswith(s) for s in SKIP_NPZ_SUFFIXES)


def _collect_npz_under(root: Path, *, recursive: bool) -> list[Path]:
    if not root.is_dir():
        return []
    it = root.rglob("*.npz") if recursive else root.glob("*.npz")
    return sorted(p for p in it if _is_structure_npz(p))


def iter_npz_files(target: Path, *, recursive: bool = True) -> list[Path]:
    if target.is_file():
        if not _is_structure_npz(target):
            raise ValueError(f"Not a structure .npz: {target}")
        return [target]
    if target.is_dir():
        files = _collect_npz_under(target, recursive=recursive)
        if not files:
            raise FileNotFoundError(f"No structure .npz under {target}")
        return files
    raise FileNotFoundError(f"Not found: {target}")


def resolve_input_dir(args: argparse.Namespace) -> Path | None:
    env_dir = os.environ.get("STRUCTURE_M_NPZ_DIR")
    if args.npz_dir is not None:
        return args.npz_dir.resolve()
    if env_dir:
        return Path(env_dir).resolve()
    if args.task2:
        return TASK02_NPZ_DIR.resolve()
    return None


def search_npz_files() -> list[Path]:
    found: list[Path] = []
    seen: set[str] = set()
    for root in DEFAULT_NPZ_DIRS:
        for p in _collect_npz_under(root, recursive=False):
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                found.append(p)
    return found


def load_structure_npz(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    if "L" not in data.files or "N" not in data.files:
        raise KeyError(f"{path.name}: need keys N and L, got {list(data.files)}")
    N = int(data["N"])
    L = np.asarray(data["L"], dtype=np.float64)
    composition = str(data["composition"]) if "composition" in data.files else ""
    source_cif = str(data["source_cif"]) if "source_cif" in data.files else ""
    out: dict[str, Any] = {"N": N, "L": L, "composition": composition, "source_cif": source_cif}
    for k in ("A", "F"):
        if k in data.files:
            out[k] = np.asarray(data[k])
    return out


def save_normalized_npz(
    out_path: Path,
    rec: dict[str, Any],
    norm: dict[str, Any],
    source_npz: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "N": np.int64(rec["N"]),
        "L": norm["L"],
        "L_scaled_N": norm["L_scaled_N"],
        "L_scaled_V": norm["L_scaled_V"],
        "volume": np.float64(norm["volume"]),
        "composition": np.array(rec["composition"]),
        "source_npz": np.array(str(source_npz)),
    }
    if "source_cif" in rec and rec["source_cif"]:
        payload["source_cif"] = np.array(rec["source_cif"])
    if "A" in rec:
        payload["A"] = rec["A"]
    if "F" in rec:
        payload["F"] = rec["F"]
    np.savez_compressed(out_path, **payload)


def record_row(source: Path, out_npz: Path | None, rec: dict[str, Any], norm: dict[str, Any]) -> dict:
    row = {
        "source_npz": str(source),
        "out_npz": str(out_npz) if out_npz else "",
        "source_cif": rec.get("source_cif", ""),
        "N": int(rec["N"]),
        "composition": rec.get("composition", ""),
        "volume_A3": float(norm["volume"]),
        "scale_N_factor": float(rec["N"]) ** (1.0 / 3.0),
        "scale_V_factor": float(norm["volume"]) ** (1.0 / 3.0),
    }
    row.update(L_to_dict(norm["L"]))
    row.update(L_to_dict(norm["L_scaled_N"], "scaled_N"))
    row.update(L_to_dict(norm["L_scaled_V"], "scaled_V"))
    return row


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_npz_dir = out_dir / "npz"
    out_dir.mkdir(parents=True, exist_ok=True)

    input_dir = resolve_input_dir(args)
    if args.path:
        targets = iter_npz_files(Path(args.path))
        print(f"Input: {args.path}")
    elif input_dir is not None:
        if not input_dir.is_dir():
            print(f"ERROR: input dir not found: {input_dir}")
            print("Run Task 02 first (structure_M_all.py) or copy all_npz/ locally.")
            sys.exit(1)
        targets = iter_npz_files(input_dir)
        print(f"Input (Task 02 structure .npz): {input_dir}")
        print(f"Found {len(targets)} file(s)")
    else:
        targets = search_npz_files()
        if not targets:
            print("No structure .npz found. Try:")
            print("  py -3 normalize_lattice_npz.py --task2")
            print("  py -3 normalize_lattice_npz.py ..\\..\\tasks\\02_structure_M\\outputs\\all_npz")
            print("  py -3 normalize_lattice_npz.py path\\to\\file.npz")
            sys.exit(1)
        print(f"Auto-found {len(targets)} structure .npz file(s)")

    if args.limit > 0:
        targets = targets[: args.limit]

    manifest_path = out_dir / "lattice_normalize_manifest.json"
    records_path = out_dir / "lattice_normalize_records.jsonl"
    errors_path = out_dir / "lattice_normalize_errors.log"
    csv_path = out_dir / "lattice_normalize_summary.csv"

    print(f"Output dir : {out_dir}")
    print(f"NPZ out    : {out_npz_dir}")
    print(f"Methods    : scale_N (a,b,c / N^(1/3)), scale_V (a,b,c / V^(1/3))")
    print(f"Files      : {len(targets)}")

    t0 = time.perf_counter()
    n_ok = 0
    errors: list[dict] = []
    csv_rows: list[dict] = []

    csv_fields = [
        "source_npz", "out_npz", "source_cif", "N", "composition", "volume_A3",
        "scale_N_factor", "scale_V_factor",
        "a", "b", "c", "alpha", "beta", "gamma",
        "a_scaled_N", "b_scaled_N", "c_scaled_N", "alpha_scaled_N", "beta_scaled_N", "gamma_scaled_N",
        "a_scaled_V", "b_scaled_V", "c_scaled_V", "alpha_scaled_V", "beta_scaled_V", "gamma_scaled_V",
    ]

    with errors_path.open("w", encoding="utf-8") as err_fh, records_path.open(
        "w", encoding="utf-8"
    ) as rec_fh:
        for i, npz_path in enumerate(targets, start=1):
            try:
                rec = load_structure_npz(npz_path)
                norm = normalize_lattice_both(rec["L"], rec["N"])
                out_path = None
                if not args.no_write_npz:
                    out_path = out_npz_dir / f"{npz_path.stem}_lattice_norm.npz"
                    save_normalized_npz(out_path, rec, norm, npz_path)

                row = record_row(npz_path, out_path, rec, norm)
                rec_fh.write(json.dumps(row) + "\n")
                csv_rows.append(row)
                n_ok += 1

                if i % args.progress_every == 0 or i == len(targets):
                    Ln = norm["L_scaled_N"]
                    print(
                        f"[{i}/{len(targets)}] OK  N={rec['N']:4d}  V={norm['volume']:.1f}  "
                        f"a'={Ln[0]:.4f} (N-scale)  {npz_path.name}"
                    )
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                errors.append({"path": str(npz_path), "error": msg})
                err_fh.write(f"{npz_path}\t{msg}\n")
                if i % args.progress_every == 0 or i == len(targets):
                    print(f"[{i}/{len(targets)}] ERR {npz_path.name}  {msg}")

    elapsed = time.perf_counter() - t0

    if csv_rows:
        import csv

        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=csv_fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(csv_rows)

    manifest = {
        "methods": {
            "scale_N": "a,b,c divided by N^(1/3); angles unchanged (CDVAE scale_length)",
            "scale_V": "a,b,c divided by V^(1/3); V = abc*sqrt(1-cos²α-cos²β-cos²γ+2cosαcosβcosγ)",
        },
        "total_files": len(targets),
        "ok": n_ok,
        "failed": len(errors),
        "elapsed_s": elapsed,
        "out_dir": str(out_dir),
        "records_jsonl": str(records_path),
        "summary_csv": str(csv_path),
        "errors": errors,
    }
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print("=" * 72)
    print(f"Done. OK={n_ok}  failed={len(errors)}  elapsed={elapsed:.1f}s")
    print(f"Manifest : {manifest_path}")
    print(f"CSV      : {csv_path}")
    print(f"Records  : {records_path}")


if __name__ == "__main__":
    main()
