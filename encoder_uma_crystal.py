#!/usr/bin/env python3
"""
Frozen UMA backbone encoder — node embeddings only (no energy / force / stress heads).

Reads ASE Atoms built from Task 02 npz, runs fairchem eSCNMD backbone.forward(),
returns per-atom node_embedding before output_heads.

Compatible with fairchem-core 2.7.x (uma-m-1p1 / uma-s-1p1) and newer checkpoints.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Must run before fairchem hub lookups (PBS may pass empty HF_HUB_OFFLINE via qsub -v).
from uma_offline_env import configure_uma_offline_env, resolve_stepwise_root

_STEPWISE = resolve_stepwise_root(__file__)

configure_uma_offline_env(stepwise_root=_STEPWISE)

import torch

try:
    from ase import Atoms
    from fairchem.core import pretrained_mlip
    from fairchem.core.datasets import data_list_collater
    from fairchem.core.datasets.atomic_data import AtomicData
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "UMA encoder requires fairchem-core. Example:\n"
        "  pip install fairchem-core\n"
        "  huggingface-cli login   # for UMA weights"
    ) from exc

try:
    from fairchem.core.units.mlip_unit.api.inference import guess_inference_settings
except ImportError:  # fairchem 2.7.x
    guess_inference_settings = None  # type: ignore[assignment,misc]

try:
    from fairchem.core.units.mlip_unit.api.inference import validate_uma_atoms_data
except ImportError:  # fairchem 2.7.x
    validate_uma_atoms_data = None  # type: ignore[assignment,misc]


@dataclass
class UMAEncoderConfig:
    model_name: str = "uma-m-1p1"
    task_name: str = "odac"
    device: str = "cuda"
    inference_settings: str = "default"


@dataclass
class UMANodeEmbedding:
    node_emb: torch.Tensor
    node_emb_l0: torch.Tensor
    emb_shape: tuple[int, ...]
    emb_dim_l0: int


def _prepare_atoms_for_task(atoms: Atoms, task_name: str) -> None:
    if validate_uma_atoms_data is not None:
        validate_uma_atoms_data(atoms, task_name)
        return
    if "charge" not in atoms.info:
        atoms.info["charge"] = 0
    if "spin" not in atoms.info:
        atoms.info["spin"] = 1 if task_name == "omol" else 0


def _batch_tensor_keys(batch: Any) -> list[str]:
    keys_attr = getattr(batch, "keys", None)
    if callable(keys_attr):
        return list(keys_attr())
    if keys_attr is not None:
        return list(keys_attr)
    return []


def _ensure_batch_on_device(batch: Any, device: torch.device) -> Any:
    """Move every tensor in fairchem AtomicData — .to() alone can miss fields."""
    if hasattr(batch, "to"):
        batch = batch.to(device)
    for key in _batch_tensor_keys(batch):
        try:
            val = batch[key]
        except (KeyError, TypeError):
            continue
        if torch.is_tensor(val) and val.device != device:
            batch[key] = val.to(device)
    batch_attr = getattr(batch, "batch", None)
    if torch.is_tensor(batch_attr) and batch_attr.device != device:
        batch.batch = batch_attr.to(device)
    return batch


def _ensure_fairchem_cache_dir() -> str:
    """fairchem reads FAIRCHEM_CACHE_DIR verbatim — expand ~ to an absolute path."""
    raw = os.environ.get(
        "FAIRCHEM_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "fairchem"),
    )
    path = os.path.abspath(os.path.expanduser(raw))
    os.environ["FAIRCHEM_CACHE_DIR"] = path
    os.makedirs(path, exist_ok=True)
    return path


def _default_snapshot_dir() -> Path:
    raw = os.environ.get("UMA_SNAPSHOT_DIR")
    if raw:
        return Path(os.path.abspath(os.path.expanduser(raw)))
    stepwise = resolve_stepwise_root(__file__)
    candidates = [
        stepwise / "wheels" / "uma" / "hf_cache" / "models" / "facebook" / "UMA",
        Path.home() / "~scratch" / "huggingface" / "models" / "facebook" / "UMA",
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand
    return candidates[0]


def _load_predict_unit(cfg: UMAEncoderConfig) -> Any:
    """Load MLIPPredictUnit across fairchem 2.7.x and newer APIs."""
    stepwise = resolve_stepwise_root(__file__)
    configure_uma_offline_env(stepwise_root=stepwise)

    snap = _default_snapshot_dir()
    ckpt = snap / "checkpoints" / f"{cfg.model_name}.pt"
    if ckpt.is_file():
        from fairchem.core.units.mlip_unit import load_predict_unit

        last_err: Exception | None = None
        for call in (
            lambda: load_predict_unit(str(ckpt), device=str(cfg.device)),
            lambda: load_predict_unit(str(ckpt)),
        ):
            try:
                return call()
            except TypeError as exc:
                last_err = exc
                continue
            except Exception as exc:
                last_err = exc
                continue
        raise RuntimeError(
            f"Failed to load UMA checkpoint from {ckpt} (offline). "
            "Do not fall back to HuggingFace on PRAGYA."
        ) from last_err

    cache_dir = _ensure_fairchem_cache_dir()
    os.environ["HF_HUB_OFFLINE"] = "1"
    kwargs: dict[str, Any] = {"device": str(cfg.device), "cache_dir": cache_dir}

    if guess_inference_settings is not None:
        try:
            settings = guess_inference_settings(cfg.inference_settings)
            return pretrained_mlip.get_predict_unit(
                cfg.model_name,
                inference_settings=settings,
                **kwargs,
            )
        except TypeError:
            kwargs.pop("cache_dir", None)
    try:
        return pretrained_mlip.get_predict_unit(cfg.model_name, **kwargs)
    except TypeError:
        kwargs.pop("cache_dir", None)
        return pretrained_mlip.get_predict_unit(cfg.model_name, device=str(cfg.device))


class UMABackboneEncoder:
    """Load pretrained UMA and run backbone only (stop before property predictor heads)."""

    def __init__(self, cfg: UMAEncoderConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.predictor = _load_predict_unit(cfg)
        self.model = self._resolve_model()
        self.model.to(self.device)
        self.model.eval()
        if hasattr(self.predictor, "to"):
            self.predictor.to(self.device)
        self.backbone = self.model.backbone
        self.backbone.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.cutoff = float(getattr(self.backbone, "cutoff", 6.0))
        self.max_neighbors = int(getattr(self.backbone, "max_neighbors", 300))

    def _resolve_model(self) -> torch.nn.Module:
        model = self.predictor.model
        if hasattr(model, "module"):
            return model.module
        return model

    def _atomic_data_with_cpu_edges(self, atoms: Atoms) -> Any:
        base = dict(
            task_name=self.cfg.task_name,
            r_edges=True,
            radius=self.cutoff,
            max_neigh=self.max_neighbors,
        )
        errors: list[Exception] = []
        try:
            from fairchem.core.datasets.atomic_data import ExternalGraphMethod

            for method in (ExternalGraphMethod.NVIDIA, ExternalGraphMethod.PYMATGEN):
                try:
                    return AtomicData.from_ase(
                        atoms,
                        external_graph_method=method,
                        **base,
                    )
                except Exception as exc:
                    errors.append(exc)
        except ImportError:
            pass
        try:
            return AtomicData.from_ase(atoms, **base)
        except Exception as exc:
            errors.append(exc)
        hint = (
            "Install pymatgen into .venv_uma: "
            "bash pragya/uma_env/install_pymatgen_offline.sh"
        )
        raise RuntimeError(
            f"CPU PBC graph build failed ({errors[-1] if errors else 'unknown'}). {hint}"
        ) from (errors[-1] if errors else None)

    def _batch_otf_cpu(self, atoms: Atoms) -> Any:
        atomic_data = AtomicData.from_ase(
            atoms,
            task_name=self.cfg.task_name,
            r_edges=False,
        )
        return data_list_collater([atomic_data], otf_graph=True)

    def _pack_embedding(self, out: dict[str, Any]) -> UMANodeEmbedding:
        if "node_embedding" not in out:
            raise KeyError(
                f"backbone.forward() missing node_embedding; keys={list(out.keys())}"
            )
        node_emb = out["node_embedding"].detach().cpu()
        if node_emb.ndim == 3:
            node_emb_l0 = node_emb[:, 0, :].contiguous()
        elif node_emb.ndim == 2:
            node_emb_l0 = node_emb
        else:
            raise RuntimeError(f"unexpected node_embedding shape {tuple(node_emb.shape)}")
        return UMANodeEmbedding(
            node_emb=node_emb,
            node_emb_l0=node_emb_l0,
            emb_shape=tuple(node_emb.shape),
            emb_dim_l0=int(node_emb_l0.shape[-1]),
        )

    def _forward_backbone(self, atoms: Atoms) -> UMANodeEmbedding:
        """CUDA: prefer CPU pymatgen graph; fallback uses predictor.predict (no pymatgen)."""
        if self.device.type == "cuda":
            try:
                atomic_data = self._atomic_data_with_cpu_edges(atoms)
                batch = data_list_collater([atomic_data], otf_graph=False)
                batch = _ensure_batch_on_device(batch, self.device)
                return self._pack_embedding(self.backbone.forward(batch))
            except Exception:
                pass

            # fairchem otf_graph + direct backbone.forward mixes CPU/CUDA tensors.
            # predictor.predict() moves batch to GPU and builds the graph (PNNL pattern).
            batch = self._batch_otf_cpu(atoms)
            if hasattr(self.predictor, "predict"):
                self.predictor.predict(batch)
            batch = _ensure_batch_on_device(batch, self.device)
            return self._pack_embedding(self.backbone.forward(batch))

        atomic_data = AtomicData.from_ase(
            atoms,
            task_name=self.cfg.task_name,
            r_edges=False,
        )
        batch = data_list_collater([atomic_data], otf_graph=True)
        batch = _ensure_batch_on_device(batch, self.device)
        return self._pack_embedding(self.backbone.forward(batch))

    def _atoms_to_batch(self, atoms: Atoms) -> Any:
        _prepare_atoms_for_task(atoms, self.cfg.task_name)
        use_cpu_edges = self.device.type == "cuda"
        last_err: Exception | None = None
        strategies: tuple[tuple[bool, bool], ...] = (
            (True, False),
            (False, True),
        ) if not use_cpu_edges else ((True, False),)

        for r_edges, otf_graph in strategies:
            try:
                if r_edges:
                    atomic_data = self._atomic_data_with_cpu_edges(atoms)
                else:
                    atomic_data = AtomicData.from_ase(
                        atoms,
                        task_name=self.cfg.task_name,
                        r_edges=False,
                    )
                batch = data_list_collater([atomic_data], otf_graph=otf_graph)
                return _ensure_batch_on_device(batch, self.device)
            except Exception as exc:
                last_err = exc
                continue
        raise RuntimeError(
            "failed to build UMA graph batch. "
            "If pymatgen is missing on CUDA, encoder uses predictor.predict fallback — "
            "sync latest encoder_uma_crystal.py. "
            "Optional speedup: bash pragya/uma_env/install_pymatgen_offline.sh (needs cp311 wheel)."
        ) from last_err

    @torch.no_grad()
    def encode_atoms(self, atoms: Atoms) -> UMANodeEmbedding:
        _prepare_atoms_for_task(atoms, self.cfg.task_name)
        return self._forward_backbone(atoms)

    @torch.no_grad()
    def encode_atoms_list(self, atoms_list: list[Atoms]) -> list[UMANodeEmbedding]:
        return [self.encode_atoms(atoms) for atoms in atoms_list]
