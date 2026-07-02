"""
Standard-scale lattice lengths for structures with N <= n_max (default 50).

Produces 5 length-normalization variants (angles unchanged):

  1. L_var_V      — a,b,c / V^(1/3)              (from scale_V)
  2. L_var_N      — a,b,c / N^(1/3)              (from scale_N)
  3. L_var_std    — StandardScaler on raw a,b,c
  4. L_var_V_std  — StandardScaler on V-scaled a,b,c
  5. L_var_N_std  — StandardScaler on N-scaled a,b,c

Input: lattice_normalize_summary.csv from normalize_lattice_npz.py
       (or recompute from structure .npz via --task2).

Windows:
  py -3 standard_scale_lattice_npz.py --summary-csv ..\\..\\tasks\\02_structure_M\\lattice_normalize_summary.csv
  py -3 standard_scale_lattice_npz.py --task2 --n-max 50
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from lattice_norm_utils import normalize_lattice_both  # noqa: E402
from lattice_std_scale_utils import ABCScaler, L_to_dict, abc_to_L  # noqa: E402
from normalize_lattice_npz import (  # noqa: E402
    TASK02_NPZ_DIR,
    iter_npz_files,
    load_structure_npz,
    resolve_input_dir,
)

DEFAULT_OUT_DIR = PIPELINE_DIR / "outputs" / "lattice_std_scaled_Nmax50"
DEFAULT_N_MAX = 50
PROGRESS_EVERY = 500

VARIANTS = ("var_V", "var_N", "var_std", "var_V_std", "var_N_std")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="5 lattice length variants + StandardScaler (N <= n_max)."
    )
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="lattice_normalize_summary.csv from normalize_lattice_npz.py",
    )
    p.add_argument("path", nargs="?", help="Optional .npz file or folder (like normalize_lattice_npz.py)")
    p.add_argument("--task2", action="store_true", help="Read Task 02 all_npz/ instead of CSV")
    p.add_argument("--npz-dir", type=Path, default=None)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.environ.get("LATTICE_STD_OUT_DIR", str(DEFAULT_OUT_DIR))),
    )
    p.add_argument(
        "--n-max",
        type=int,
        default=int(os.environ.get("LATTICE_STD_N_MAX", str(DEFAULT_N_MAX))),
        help="Keep structures with N <= n_max (default 50)",
    )
    p.add_argument("--limit", type=int, default=0, help="Max rows/files after filter (0 = all)")
    p.add_argument(
        "--progress-every",
        type=int,
        default=int(os.environ.get("LATTICE_STD_PROGRESS_EVERY", PROGRESS_EVERY)),
    )
    p.add_argument(
        "--no-write-npz",
        action="store_true",
        help="Only CSV + scaler JSON (skip per-structure .npz)",
    )
    return p.parse_args()


def _default_summary_csv() -> Path | None:
    candidates = [
        PIPELINE_DIR / "outputs" / "normalized_lattice" / "lattice_normalize_summary.csv",
        PIPELINE_DIR.parent.parent / "tasks" / "02_structure_M" / "lattice_normalize_summary.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def load_rows_from_csv(csv_path: Path, *, n_max: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    float_cols = (
        "volume_A3",
        "a", "b", "c", "alpha", "beta", "gamma",
        "a_scaled_N", "b_scaled_N", "c_scaled_N",
        "a_scaled_V", "b_scaled_V", "c_scaled_V",
    )
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            n = int(float(row["N"]))
            if n > n_max:
                continue
            row["N"] = n
            for col in float_cols:
                if col in row and row[col] != "":
                    row[col] = float(row[col])
            rows.append(row)
    return rows


def row_from_npz(npz_path: Path, norm: dict[str, Any], rec: dict[str, Any]) -> dict[str, Any]:
    L = norm["L"]
    Ln = norm["L_scaled_N"]
    Lv = norm["L_scaled_V"]
    return {
        "source_npz": str(npz_path),
        "source_cif": rec.get("source_cif", ""),
        "N": int(rec["N"]),
        "composition": rec.get("composition", ""),
        "volume_A3": float(norm["volume"]),
        "a": float(L[0]),
        "b": float(L[1]),
        "c": float(L[2]),
        "alpha": float(L[3]),
        "beta": float(L[4]),
        "gamma": float(L[5]),
        "a_scaled_N": float(Ln[0]),
        "b_scaled_N": float(Ln[1]),
        "c_scaled_N": float(Ln[2]),
        "a_scaled_V": float(Lv[0]),
        "b_scaled_V": float(Lv[1]),
        "c_scaled_V": float(Lv[2]),
    }


def load_rows_from_npz(targets: list[Path], *, n_max: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for npz_path in targets:
        rec = load_structure_npz(npz_path)
        if int(rec["N"]) > n_max:
            continue
        norm = normalize_lattice_both(rec["L"], rec["N"])
        rows.append(row_from_npz(npz_path, norm, rec))
    return rows


def fit_scalers(rows: list[dict[str, Any]]) -> dict[str, ABCScaler]:
    raw = np.array([[r["a"], r["b"], r["c"]] for r in rows], dtype=np.float64)
    n_abc = np.array(
        [[r["a_scaled_N"], r["b_scaled_N"], r["c_scaled_N"]] for r in rows],
        dtype=np.float64,
    )
    v_abc = np.array(
        [[r["a_scaled_V"], r["b_scaled_V"], r["c_scaled_V"]] for r in rows],
        dtype=np.float64,
    )
    scaler_raw = ABCScaler().fit(raw)
    scaler_n = ABCScaler().fit(n_abc)
    scaler_v = ABCScaler().fit(v_abc)
    return {"raw": scaler_raw, "N": scaler_n, "V": scaler_v}


def compute_variants(
    row: dict[str, Any],
    scalers: dict[str, ABCScaler],
) -> dict[str, np.ndarray]:
    angles = np.array(
        [row["alpha"], row["beta"], row["gamma"]],
        dtype=np.float64,
    )
    abc_raw = np.array([row["a"], row["b"], row["c"]], dtype=np.float64)
    abc_n = np.array(
        [row["a_scaled_N"], row["b_scaled_N"], row["c_scaled_N"]],
        dtype=np.float64,
    )
    abc_v = np.array(
        [row["a_scaled_V"], row["b_scaled_V"], row["c_scaled_V"]],
        dtype=np.float64,
    )

    return {
        "L_var_V": abc_to_L(abc_v, angles),
        "L_var_N": abc_to_L(abc_n, angles),
        "L_var_std": abc_to_L(scalers["raw"].transform(abc_raw), angles),
        "L_var_V_std": abc_to_L(scalers["V"].transform(abc_v), angles),
        "L_var_N_std": abc_to_L(scalers["N"].transform(abc_n), angles),
    }


def stem_from_row(row: dict[str, Any]) -> str:
    src = Path(row["source_npz"])
    return src.stem


def save_npz(
    out_path: Path,
    row: dict[str, Any],
    variants: dict[str, np.ndarray],
    *,
    n_max: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    L_raw = abc_to_L(
        [row["a"], row["b"], row["c"]],
        [row["alpha"], row["beta"], row["gamma"]],
    )
    payload: dict[str, Any] = {
        "N": np.int64(row["N"]),
        "L": L_raw,
        "volume": np.float64(row.get("volume_A3", 0.0)),
        "composition": np.array(row.get("composition", "")),
        "source_npz": np.array(row["source_npz"]),
        "n_max_filter": np.int64(n_max),
    }
    if row.get("source_cif"):
        payload["source_cif"] = np.array(row["source_cif"])
    for key, L in variants.items():
        payload[key] = L
    np.savez_compressed(out_path, **payload)


def summary_row(
    row: dict[str, Any],
    variants: dict[str, np.ndarray],
    out_npz: Path | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "source_npz": row["source_npz"],
        "out_npz": str(out_npz) if out_npz else "",
        "source_cif": row.get("source_cif", ""),
        "N": int(row["N"]),
        "composition": row.get("composition", ""),
        "volume_A3": row.get("volume_A3", ""),
    }
    out.update(L_to_dict(abc_to_L([row["a"], row["b"], row["c"]], [row["alpha"], row["beta"], row["gamma"]])))
    for var_key, L in variants.items():
        suffix = var_key.removeprefix("L_")
        out.update(L_to_dict(L, suffix))
    return out


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_npz_dir = out_dir / "npz"
    scaler_dir = out_dir / "scalers"
    out_dir.mkdir(parents=True, exist_ok=True)
    scaler_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.summary_csv
    if csv_path is None and not args.task2 and args.path is None:
        csv_path = _default_summary_csv()

    if csv_path is not None and csv_path.is_file():
        print(f"Input CSV : {csv_path.resolve()}")
        rows = load_rows_from_csv(csv_path.resolve(), n_max=args.n_max)
    elif args.path or args.task2 or args.npz_dir or os.environ.get("STRUCTURE_M_NPZ_DIR"):
        input_dir = resolve_input_dir(args)
        if args.path:
            targets = iter_npz_files(Path(args.path))
            print(f"Input npz: {args.path}")
        elif input_dir is not None:
            targets = iter_npz_files(input_dir)
            print(f"Input npz: {input_dir}")
        else:
            print("ERROR: provide --summary-csv, --task2, or a .npz path")
            sys.exit(1)
        rows = load_rows_from_npz(targets, n_max=args.n_max)
    else:
        print("ERROR: no input found. Try:")
        print("  py -3 standard_scale_lattice_npz.py --summary-csv path\\to\\lattice_normalize_summary.csv")
        print("  py -3 standard_scale_lattice_npz.py --task2 --n-max 50")
        sys.exit(1)

    if args.limit > 0:
        rows = rows[: args.limit]

    if not rows:
        print(f"No structures with N <= {args.n_max}")
        sys.exit(1)

    print(f"Filter     : N <= {args.n_max}")
    print(f"Structures : {len(rows)}")
    print(f"Output     : {out_dir}")

    t0 = time.perf_counter()
    scalers = fit_scalers(rows)

    for name, scaler in scalers.items():
        path = scaler_dir / f"abc_scaler_{name}.json"
        path.write_text(json.dumps(scaler.to_dict(), indent=2), encoding="utf-8")

    csv_out = out_dir / "lattice_std_scaled_summary.csv"
    manifest_path = out_dir / "lattice_std_scaled_manifest.json"
    records_path = out_dir / "lattice_std_scaled_records.jsonl"

    csv_fields = [
        "source_npz", "out_npz", "source_cif", "N", "composition", "volume_A3",
        "a", "b", "c", "alpha", "beta", "gamma",
    ]
    for suffix in VARIANTS:
        csv_fields.extend(
            [f"a_{suffix}", f"b_{suffix}", f"c_{suffix}", f"alpha_{suffix}", f"beta_{suffix}", f"gamma_{suffix}"]
        )

    csv_rows: list[dict[str, Any]] = []
    with records_path.open("w", encoding="utf-8") as rec_fh:
        for i, row in enumerate(rows, start=1):
            variants = compute_variants(row, scalers)
            out_path = None
            if not args.no_write_npz:
                out_path = out_npz_dir / f"{stem_from_row(row)}_lattice_std.npz"
                save_npz(out_path, row, variants, n_max=args.n_max)

            summary = summary_row(row, variants, out_path)
            rec_fh.write(json.dumps(summary) + "\n")
            csv_rows.append(summary)

            if i % args.progress_every == 0 or i == len(rows):
                print(f"[{i}/{len(rows)}] N={row['N']:3d}  {stem_from_row(row)}")

    with csv_out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(csv_rows)

    elapsed = time.perf_counter() - t0
    manifest = {
        "n_max": args.n_max,
        "n_structures": len(rows),
        "variants": {
            "L_var_V": "a,b,c / V^(1/3); angles unchanged",
            "L_var_N": "a,b,c / N^(1/3); angles unchanged",
            "L_var_std": "StandardScaler on raw a,b,c; angles unchanged",
            "L_var_V_std": "StandardScaler on V-scaled a,b,c; angles unchanged",
            "L_var_N_std": "StandardScaler on N-scaled a,b,c; angles unchanged",
        },
        "scaler_fit_on": f"all {len(rows)} structures with N <= {args.n_max}",
        "scalers": {
            "raw": str(scaler_dir / "abc_scaler_raw.json"),
            "N": str(scaler_dir / "abc_scaler_N.json"),
            "V": str(scaler_dir / "abc_scaler_V.json"),
        },
        "elapsed_s": elapsed,
        "summary_csv": str(csv_out),
        "records_jsonl": str(records_path),
        "out_dir": str(out_dir),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"Done. {len(rows)} structures  elapsed={elapsed:.1f}s")
    print(f"CSV      : {csv_out}")
    print(f"Scalers  : {scaler_dir}")
    print(f"Manifest : {manifest_path}")
    print("Variants : L_var_V, L_var_N, L_var_std, L_var_V_std, L_var_N_std")


if __name__ == "__main__":
    main()
