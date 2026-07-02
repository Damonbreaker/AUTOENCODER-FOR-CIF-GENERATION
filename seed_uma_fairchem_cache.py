#!/usr/bin/env python3
"""Populate fairchem's HuggingFace cache for UMA (works offline on PRAGYA)."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ID = "facebook/UMA"
# facebook/UMA main branch commit (must match refs/main for hf_hub_download offline)
OFFLINE_REVISION = os.environ.get(
    "UMA_HF_REVISION", "7210de6fe86ad94854b21b881fefbcfdfeab373b"
)

REF_FILES = [
    ("references", "iso_atom_elem_refs.yaml"),
    ("references", "form_elem_refs.yaml"),
]

MODEL_CHECKPOINTS = {
    "uma-m-1p1": ("checkpoints", "uma-m-1p1.pt"),
    "uma-s-1p1": ("checkpoints", "uma-s-1p1.pt"),
    "uma-s-1p2": ("checkpoints", "uma-s-1p2.pt"),
}


def clean_path(path: str) -> str:
    return path.replace("\r", "").strip()


def default_cache_dir() -> str:
    raw = os.environ.get(
        "FAIRCHEM_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "fairchem"),
    )
    return clean_path(raw)


def discover_snapshot_dir(explicit: str | None = None) -> Path:
    if explicit:
        p = Path(clean_path(explicit))
        if p.is_dir():
            return p
        raise FileNotFoundError(f"Snapshot dir not found: {p}")

    home = Path(os.path.expanduser("~"))
    candidates = [
        Path(clean_path(os.environ["HF_HOME"])) / "models" / "facebook" / "UMA"
        if os.environ.get("HF_HOME")
        else None,
        Path(clean_path(os.environ.get("STEPWISE", ""))) / "wheels" / "uma" / "hf_cache" / "models" / "facebook" / "UMA"
        if os.environ.get("STEPWISE")
        else None,
        home / "~scratch" / "Jatin" / "Proj1" / "Newpipeline" / "cdvae_crystal_diffusion_vae" / "cdvae_stepwise" / "wheels" / "uma" / "hf_cache" / "models" / "facebook" / "UMA",
        home / "~scratch" / "huggingface" / "models" / "facebook" / "UMA",
        home / ".cache" / "huggingface" / "models" / "facebook" / "UMA",
    ]
    for cand in candidates:
        if cand is not None and cand.is_dir():
            return cand

    # Last resort: search under ~/~scratch for uma-m-1p1.pt parent dirs
    scratch = home / "~scratch"
    if scratch.is_dir():
        for pt in scratch.rglob("uma-m-1p1.pt"):
            if pt.parent.name == "checkpoints" and pt.parent.parent.name == "UMA":
                return pt.parent.parent

    tried = [str(c) for c in candidates if c is not None]
    raise FileNotFoundError(
        "Could not find UMA snapshot directory. Tried:\n  "
        + "\n  ".join(tried)
        + f"\n  and rglob under {scratch}"
    )


def resolve_files(model: str) -> list[tuple[str, str]]:
    if model not in MODEL_CHECKPOINTS:
        known = ", ".join(sorted(MODEL_CHECKPOINTS))
        raise SystemExit(f"Unknown model {model!r}. Known: {known}")
    sub, fn = MODEL_CHECKPOINTS[model]
    return [(sub, fn), *REF_FILES]


def repo_cache_folder(cache_dir: str) -> Path:
    return Path(cache_dir) / "models--facebook--UMA"


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def migrate_snapshot_to_fairchem_cache(
    snapshot_dir: Path,
    cache_dir: str,
    model: str,
) -> None:
    repo_folder = repo_cache_folder(cache_dir)
    refs_main = repo_folder / "refs" / "main"
    refs_main.parent.mkdir(parents=True, exist_ok=True)
    refs_main.write_text(OFFLINE_REVISION, encoding="utf-8")

    snapshot_out = repo_folder / "snapshots" / OFFLINE_REVISION

    for subfolder, filename in resolve_files(model):
        label = f"{subfolder}/{filename}"
        src = snapshot_dir / subfolder / filename
        if not src.is_file():
            raise FileNotFoundError(f"Missing in snapshot: {src}")
        dst = snapshot_out / subfolder / filename
        link_or_copy(src, dst)
        print(f"[migrated] {label}")
        print(f"           {src} -> {dst}")


def main() -> int:
    p = argparse.ArgumentParser(description="Seed or verify fairchem UMA weight cache")
    p.add_argument("--model", default=clean_path(os.environ.get("UMA_MODEL", "uma-m-1p1")))
    p.add_argument("--cache-dir", default=default_cache_dir())
    p.add_argument(
        "--snapshot-dir",
        default=None,
        help="Old huggingface-cli --local-dir tree (auto-detected if omitted)",
    )
    p.add_argument(
        "--migrate-from-snapshot",
        action="store_true",
        help="Seed fairchem cache from existing snapshot files (offline PRAGYA fix)",
    )
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()

    args.cache_dir = clean_path(args.cache_dir)
    os.makedirs(args.cache_dir, exist_ok=True)

    snapshot_path: Path | None = None
    if args.migrate_from_snapshot or args.snapshot_dir:
        try:
            snapshot_path = discover_snapshot_dir(args.snapshot_dir)
        except FileNotFoundError as exc:
            print(f"[FAIL] {exc}", file=sys.stderr)
            print(
                "\nRun the bash setup instead:\n"
                "  bash pragya/uma_env/setup_fairchem_hf_offline.sh",
                file=sys.stderr,
            )
            return 1

    if args.migrate_from_snapshot and snapshot_path is not None:
        print(f"snapshot  : {snapshot_path}")
        print(f"cache_dir : {args.cache_dir}")
        print(f"model     : {args.model}")
        try:
            migrate_snapshot_to_fairchem_cache(snapshot_path, args.cache_dir, args.model)
        except FileNotFoundError as exc:
            print(f"[FAIL] {exc}", file=sys.stderr)
            return 1
        print("\n[OK] migration complete")

    offline = args.check_only or args.migrate_from_snapshot or os.environ.get(
        "HF_HUB_OFFLINE", ""
    ).lower() in {"1", "true", "yes"}

    from huggingface_hub import hf_hub_download

    print(f"cache_dir : {args.cache_dir}")
    print(f"model     : {args.model}")
    print(f"offline   : {offline}")

    missing: list[str] = []
    for subfolder, filename in resolve_files(args.model):
        label = f"{subfolder}/{filename}"
        try:
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                subfolder=subfolder,
                cache_dir=args.cache_dir,
                local_files_only=offline,
            )
            size_gb = os.path.getsize(path) / (1024**3)
            print(f"[OK] {label}  ({size_gb:.2f} GB)")
            print(f"     -> {path}")
        except Exception as exc:
            missing.append(label)
            print(f"[MISSING] {label}: {exc}", file=sys.stderr)

    if missing:
        if snapshot_path is None:
            try:
                snapshot_path = discover_snapshot_dir()
            except FileNotFoundError:
                snapshot_path = None
        if snapshot_path is not None:
            print(
                "\nSnapshot found but fairchem cache empty. Run:\n"
                "  bash pragya/uma_env/setup_fairchem_hf_offline.sh",
                file=sys.stderr,
            )
        return 1

    print("\n[OK] fairchem cache ready for offline use")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
