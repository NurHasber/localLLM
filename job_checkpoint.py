"""Persist diffusion checkpoints for pause/resume."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
CHECKPOINT_ROOT = ROOT / "logs" / "checkpoints"


def job_checkpoint_dir(job_id: str) -> Path:
    path = CHECKPOINT_ROOT / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def has_checkpoint(job_id: str) -> bool:
    work = CHECKPOINT_ROOT / job_id
    return (work / "meta.json").is_file() and (work / "latents.pt").is_file()


def checkpoint_completed_step(job_id: str) -> int | None:
    loaded = load_checkpoint_meta(job_id)
    if not loaded:
        return None
    try:
        return int(loaded.get("completed_step") or 0) or None
    except (TypeError, ValueError):
        return None


def load_checkpoint_meta(job_id: str) -> dict[str, Any] | None:
    meta_path = CHECKPOINT_ROOT / job_id / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def delete_checkpoint(job_id: str) -> None:
    shutil.rmtree(CHECKPOINT_ROOT / job_id, ignore_errors=True)


def save_checkpoint(
    job_id: str,
    *,
    completed_step: int,
    latents,
    meta: dict[str, Any],
) -> None:
    """Atomically save latents + meta so a crash mid-write cannot corrupt the previous checkpoint."""
    import torch

    work = job_checkpoint_dir(job_id)
    meta = dict(meta)
    meta["completed_step"] = int(completed_step)

    tmp_latents = work / "latents.pt.tmp"
    tmp_meta = work / "meta.json.tmp"
    final_latents = work / "latents.pt"
    final_meta = work / "meta.json"

    torch.save(latents.detach().cpu(), tmp_latents)
    tmp_meta.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    # Replace meta last so a partial write never points at missing latents.
    tmp_latents.replace(final_latents)
    tmp_meta.replace(final_meta)


def load_checkpoint(job_id: str) -> tuple[dict[str, Any], Any] | None:
    import torch

    work = CHECKPOINT_ROOT / job_id
    meta_path = work / "meta.json"
    latent_path = work / "latents.pt"
    if not meta_path.is_file() or not latent_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        latents = torch.load(latent_path, map_location="cpu", weights_only=True)
    except Exception:
        return None
    if not isinstance(meta, dict) or "completed_step" not in meta:
        return None
    return meta, latents
