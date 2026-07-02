#!/usr/bin/env python3
"""
Shared GEMMI CIF helpers (extracted from tasks/01_cif_read/cif_read.py).

Used by cif_read_sample.py and cif_read_all.py in this pipeline.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, List, Tuple

import gemmi

DEFAULT_CIF_DIR = Path(
    "/home/chemical/phd/chz258281/~scratch/Jatin/Proj1/Newpipeline/MOF_database"
)


def find_cif_files(cif_dir: Path) -> list[Path]:
    """Return all .cif files under cif_dir (including subfolders), sorted."""
    if not cif_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {cif_dir}")
    cifs = sorted(cif_dir.rglob("*.cif"))
    if not cifs:
        raise FileNotFoundError(f"No .cif files found in: {cif_dir}")
    return cifs


def pick_n_cifs(all_cifs: list[Path], n: int, seed: int) -> list[Path]:
    """Pick n CIF paths reproducibly (same seed → same files)."""
    if n > len(all_cifs):
        raise ValueError(f"Asked for n={n} but only {len(all_cifs)} files exist")
    import random

    rng = random.Random(seed)
    return sorted(rng.sample(all_cifs, n))


def preprocess_cif_for_gemmi(text: str) -> str:
    """Quote unquoted CIF values with spaces (needed for fapswitch MOF CIFs)."""
    new_lines: List[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if not stripped.startswith("_") or stripped.startswith("loop_"):
            new_lines.append(line)
            continue
        match = re.match(r"^(_[^\s]+)([\t ]+)(.+)$", stripped)
        if not match:
            new_lines.append(line)
            continue
        tag, sep, value = match.group(1), match.group(2), match.group(3).strip()
        if value.startswith(("'", '"', ";")) or " " not in value:
            new_lines.append(line)
            continue
        indent = line[: len(line) - len(stripped)]
        new_lines.append(f"{indent}{tag}{sep}'{value}'")
    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(new_lines) + suffix


def formula_from_sites(structure: gemmi.SmallStructure) -> str:
    """Hill-style formula from gemmi sites."""
    counts: Counter = Counter(site.element.name for site in structure.sites)
    order: List[str] = []
    for el in ("C", "H"):
        if el in counts:
            order.append(el)
    for el in sorted(counts):
        if el not in order:
            order.append(el)
    parts: List[str] = []
    for el in order:
        n = counts[el]
        parts.append(f"{el}{n}" if n != 1 else el)
    return "".join(parts)


def site_frac(site: Any) -> Tuple[float, float, float]:
    """Fractional coords (gemmi 0.7.x uses .fract)."""
    fract = getattr(site, "fract", None)
    if fract is not None:
        return (float(fract.x), float(fract.y), float(fract.z))
    frac = getattr(site, "frac", None)
    if frac is not None:
        return (float(frac.x), float(frac.y), float(frac.z))
    raise AttributeError("gemmi site has neither .fract nor .frac")


def read_cif(path: Path) -> gemmi.SmallStructure:
    """Load one MOF CIF into a gemmi SmallStructure."""
    raw_text = path.read_text(errors="replace")
    cif_text = preprocess_cif_for_gemmi(raw_text)

    doc = gemmi.cif.read_string(cif_text)
    if len(doc) == 0:
        raise ValueError(f"No data blocks in CIF: {path}")

    try:
        block = doc.sole_block()
    except Exception:
        block = doc[0]

    structure = gemmi.make_small_structure_from_block(block)
    if len(structure.sites) == 0:
        raise ValueError(f"GEMMI read 0 sites from: {path}")
    return structure


def structure_summary_dict(path: Path, structure: gemmi.SmallStructure) -> dict:
    """Compact record for manifest (one row per CIF)."""
    cell = structure.cell
    return {
        "path": str(path),
        "formula": formula_from_sites(structure),
        "N": len(structure.sites),
        "a": float(cell.a),
        "b": float(cell.b),
        "c": float(cell.c),
        "alpha": float(cell.alpha),
        "beta": float(cell.beta),
        "gamma": float(cell.gamma),
        "volume": float(cell.volume),
    }


def print_structure_summary(path: Path, structure: gemmi.SmallStructure) -> None:
    """Print basic crystal info for one gemmi structure."""
    cell = structure.cell
    formula = formula_from_sites(structure)
    n_sites = len(structure.sites)

    print("=" * 72)
    print(f"file    : {path}")
    print(f"formula : {formula}")
    print(f"N atoms : {n_sites}")
    print(f"a,b,c   : {cell.a:.4f}, {cell.b:.4f}, {cell.c:.4f} Angstrom")
    print(f"alpha,beta,gamma : {cell.alpha:.2f}, {cell.beta:.2f}, {cell.gamma:.2f} deg")
    print(f"volume  : {cell.volume:.2f} Angstrom^3")
    print("first sites (element, frac_x, frac_y, frac_z):")
    for i, site in enumerate(structure.sites[:3]):
        x, y, z = site_frac(site)
        symbol = site.element.name
        print(f"  site {i}: {symbol:>2s}  {x:.6f}  {y:.6f}  {z:.6f}")
    if n_sites > 3:
        print(f"  ... ({n_sites - 3} more sites)")
