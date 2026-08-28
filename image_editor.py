"""Multi-model local image editing (lazy load, no startup download)."""

from __future__ import annotations

import gc
import io
import json
import os
import logging
import random
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING, Callable

from PIL import Image

if TYPE_CHECKING:
    from pipeline_progress import PipelineProgress

CancelFn = Callable[[], bool]
PauseFn = Callable[[], bool]

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

_log = logging.getLogger(__name__)

_torch = None
_torch_lock = threading.Lock()
_global_load_lock = threading.Lock()
_cache_complete: dict[str, bool] = {}
_cache_lock = threading.Lock()
_active_pipes: dict[str, object] = {}
_active_pipes_lock = threading.Lock()

# JoyCaption analysis only — never feed full camera megapixels into the vision model.
_MAX_SIDE_JOYCAPTION_ANALYSIS = 768

# Qwen image-edit works best at moderate resolutions; large inputs cause
# the model to "see" only a crop of the image, cutting off people/objects.
_MAX_SIDE_QWEN_GENERATION = 1024


def _snapshot_dir(repo_id: str) -> Path | None:
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id, local_files_only=True))
    except Exception:
        return None


def _sharded_folder_complete(folder: Path) -> bool:
    index_files = list(folder.glob("*.safetensors.index.json"))
    if not index_files:
        return any(folder.glob("*.safetensors")) or any(folder.glob("*.bin"))
    for index_path in index_files:
        data = json.loads(index_path.read_text(encoding="utf-8"))
        for filename in set(data.get("weight_map", {}).values()):
            if not (folder / filename).is_file():
                return False
    return True


def _repo_is_complete(repo_id: str) -> bool:
    with _cache_lock:
        if repo_id in _cache_complete:
            return _cache_complete[repo_id]

    root = _snapshot_dir(repo_id)
    if not root:
        complete = False
    else:
        index_files = list(root.rglob("*.safetensors.index.json"))
        if not index_files:
            complete = any(root.rglob("*.safetensors"))
        else:
            complete = all(_sharded_folder_complete(index_path.parent) for index_path in index_files)

    with _cache_lock:
        _cache_complete[repo_id] = complete
    return complete


def invalidate_repo_cache(repo_id: str | None = None) -> None:
    with _cache_lock:
        if repo_id is None:
            _cache_complete.clear()
        else:
            _cache_complete.pop(repo_id, None)


def _missing_shard_paths(repo_id: str) -> list[Path]:
    root = _snapshot_dir(repo_id)
    if not root:
        return []
    missing: list[Path] = []
    for index_path in root.rglob("*.safetensors.index.json"):
        folder = index_path.parent
        data = json.loads(index_path.read_text(encoding="utf-8"))
        for filename in set(data.get("weight_map", {}).values()):
            path = folder / filename
            if not path.is_file():
                missing.append(path)
    return missing


def _download_repo(repo_id: str, pipeline: PipelineProgress | None = None) -> None:
    from huggingface_hub import snapshot_download

    missing = _missing_shard_paths(repo_id)
    total = len(missing)

    print(f"\n{'=' * 60}", flush=True)
    print(f"[download] {repo_id}", flush=True)
    if total:
        print(f"  Докачать {total} файл(ов) — прогресс ниже (это может занять 10–40 мин):", flush=True)
        for path in missing[:8]:
            print(f"    · {path.parent.name}/{path.name}", flush=True)
        if total > 8:
            print(f"    · ... и ещё {total - 8}", flush=True)
    else:
        print("  Проверка и загрузка файлов с HuggingFace...", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    prev_bar = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

    stop = threading.Event()

    def monitor() -> None:
        pulse = 0.08
        tracked = list(missing)
        while not stop.wait(4):
            if tracked:
                left = sum(1 for path in tracked if not path.is_file())
                done = len(tracked) - left
                fraction = 0.05 + 0.45 * (done / len(tracked))
                detail = f"скачано {done}/{len(tracked)} файлов"
            else:
                pulse = min(0.42, pulse + 0.02)
                fraction = pulse
                detail = "скачивание с HuggingFace..."
            if pipeline:
                pipeline.report("load_model", fraction, detail=detail)

    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    try:
        snapshot_download(repo_id, max_workers=1)
        print(f"\n[download] {repo_id} — все файлы на диске\n", flush=True)
        if pipeline:
            pipeline.report("load_model", 0.5, detail="скачивание завершено")
    finally:
        stop.set()
        worker.join(timeout=0.5)
        if prev_bar is None:
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        else:
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = prev_bar


def _ensure_repo_files(repo_id: str, pipeline: PipelineProgress | None) -> None:
    if _repo_is_complete(repo_id):
        if pipeline:
            pipeline.report("load_model", 0.2, detail="все файлы модели на диске")
        return

    if pipeline:
        pipeline.report("load_model", 0.05, detail="докачивание недостающих частей модели")

    try:
        _download_repo(repo_id, pipeline)
        invalidate_repo_cache(repo_id)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        raise RuntimeError(f"Не удалось докачать модель {repo_id}: {exc}") from exc


def _abort_if(cancel_checker: CancelFn | None) -> None:
    if cancel_checker and cancel_checker():
        from job_errors import JobCancelled

        raise JobCancelled()


def _abort_if_paused(pause_checker: PauseFn | None) -> None:
    if pause_checker and pause_checker():
        from job_errors import JobPaused

        raise JobPaused()


def _check_control(cancel_checker: CancelFn | None, pause_checker: PauseFn | None) -> None:
    _abort_if(cancel_checker)
    _abort_if_paused(pause_checker)


def _run_with_load_pulse(
    pipeline: PipelineProgress | None,
    action,
    *,
    phase_id: str = "load_model",
    cancel_checker: CancelFn | None = None,
    details: list[str] | None = None,
):
    stop = threading.Event()
    pulse_details = details or [
        "чтение весов с диска...",
        "сборка модели...",
        "подготовка GPU...",
        "ещё немного...",
    ]

    def pulse() -> None:
        step = 0
        fraction = 0.25
        while not stop.wait(2):
            if cancel_checker and cancel_checker():
                return
            if pipeline:
                fraction = min(0.92, fraction + 0.04)
                pipeline.report(
                    phase_id,
                    fraction,
                    detail=pulse_details[step % len(pulse_details)],
                    journal_replace=True,
                )
                step += 1

    worker = threading.Thread(target=pulse, daemon=True)
    worker.start()
    try:
        _abort_if(cancel_checker)
        return action()
    finally:
        stop.set()
        worker.join(timeout=0.2)


def _gpu_vram_gb() -> float:
    torch = _import_torch()
    if not torch.cuda.is_available():
        return 0.0
    try:
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception:
        return 0.0


def get_gpu_info() -> dict:
    torch = _import_torch()
    if not torch.cuda.is_available():
        return {
            "cuda": False,
            "device_name": None,
            "vram_total_gb": 0.0,
            "preferred_mode": "cpu",
        }
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024**3)
    return {
        "cuda": True,
        "device_name": props.name,
        "vram_total_gb": round(total_gb, 2),
        "preferred_mode": _preferred_gpu_mode(),
    }


def resolve_gpu_chain_model(model_id: str) -> str:
    if uses_joycaption_flow(model_id) and not is_test_model(model_id):
        return MODELS["joycaption"].chain_to or "flux-kontext"
    return model_id


def inference_steps_for_model(model_id: str, steps: int | None = None) -> int:
    chain_id = resolve_gpu_chain_model(model_id)
    allowed = MODEL_STEP_PRESETS.get(chain_id) or MODEL_STEP_PRESETS.get(model_id, [28])
    default = MODEL_DEFAULT_STEPS.get(chain_id, MODEL_DEFAULT_STEPS.get(model_id, 28))
    if steps is None or steps <= 0:
        return default
    if steps not in allowed:
        allowed_txt = ", ".join(str(s) for s in allowed)
        raise ValueError(f"Для модели допустимы шаги: {allowed_txt}")
    return steps


def _preferred_gpu_mode() -> str:
    """cuda = full VRAM (fast), offload = sequential CPU/GPU (slow, low GPU %)."""
    if _gpu_vram_gb() >= 18:
        return "cuda"
    if _import_torch().cuda.is_available():
        return "offload"
    return "cpu"


# Backward alias
_flux_load_device = _preferred_gpu_mode


def _place_diffusion_pipe(
    pipe,
    *,
    pipeline: PipelineProgress | None,
    label: str,
    cancel_checker: CancelFn | None = None,
) -> str:
    """Move diffusers pipeline to GPU or CPU offload; returns mode id."""
    torch = _import_torch()
    if not torch.cuda.is_available():
        pipe.to("cpu")
        return "cpu"

    mode = _preferred_gpu_mode()

    def _move_to_cuda() -> None:
        _abort_if(cancel_checker)
        pipe.to("cuda")

    def _enable_offload() -> None:
        _abort_if(cancel_checker)
        pipe.enable_model_cpu_offload()

    if mode == "cuda":
        if pipeline:
            pipeline.report(
                "load_model",
                0.92,
                detail=f"{label}: перенос весов на GPU (1–3 мин, GPU будет загружен)",
            )
        try:
            _run_with_load_pulse(
                pipeline,
                _move_to_cuda,
                cancel_checker=cancel_checker,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            _free_gpu_memory()
            if pipeline:
                pipeline.report(
                    "load_model",
                    0.94,
                    detail=f"{label}: не хватило VRAM — CPU offload",
                )
            _run_with_load_pulse(
                pipeline,
                _enable_offload,
                cancel_checker=cancel_checker,
            )
            if _gpu_vram_gb() >= 18:
                _log.warning(
                    "%s: OOM on full CUDA — fallback to CPU offload (VRAM %.1f GB)",
                    label,
                    _gpu_vram_gb(),
                )
            return "offload"
        if pipeline:
            pipeline.report("load_model", 0.98, detail=f"{label} на GPU")
        return "cuda"

    if _gpu_vram_gb() >= 18:
        _log.warning(
            "%s: VRAM %.1f GB but preferred mode is offload (<18 GB threshold not met in worker?)",
            label,
            _gpu_vram_gb(),
        )
    if pipeline:
        pipeline.report("load_model", 0.95, detail=f"{label}: CPU offload (медленнее)")
    _run_with_load_pulse(pipeline, _enable_offload, cancel_checker=cancel_checker)
    return "offload"


def _invalidate_gpu_models_if_mode_changed() -> None:
    want = _preferred_gpu_mode()
    for state in MODELS.values():
        if state.kind not in ("flux", "qwen_edit", "qwen_t2i") or not _model_ready(state):
            continue
        if state.device_mode == want:
            continue
        state.pipe = None
        state.processor = None
        state.device_mode = ""
    _free_gpu_memory()


def _generate_start_detail(model_id: str) -> str:
    state = MODELS.get(model_id)
    name = state.name if state else model_id
    return f"запуск «{name}» на GPU"


def _diffusion_generator(torch, seed: int | None = None):
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    if torch.cuda.is_available():
        return torch.Generator(device="cuda").manual_seed(seed), seed
    return torch.Generator(device="cpu").manual_seed(seed), seed


def _free_gpu_memory() -> None:
    gc.collect()
    if _torch is None:
        return
    torch = _torch
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass


def register_job_pipe(job_id: str | None, pipe) -> None:
    if not job_id or pipe is None:
        return
    with _active_pipes_lock:
        _active_pipes[job_id] = pipe


def clear_job_pipe(job_id: str | None) -> None:
    if not job_id:
        return
    with _active_pipes_lock:
        _active_pipes.pop(job_id, None)


def interrupt_job_pipe(job_id: str) -> None:
    with _active_pipes_lock:
        pipe = _active_pipes.get(job_id)
    if pipe is None:
        return
    try:
        pipe._interrupt = True
    except Exception:
        pass


def abort_generation(job_id: str | None) -> None:
    """Signal an in-flight diffusion pipeline to stop as soon as possible."""
    if not job_id:
        return
    interrupt_job_pipe(job_id)
    with _active_pipes_lock:
        pipe = _active_pipes.get(job_id)
    if pipe is None:
        return
    try:
        pipe._interrupt = True
    except Exception:
        pass


def release_gpu_vram() -> None:
    """Unload diffusion models and free CUDA memory (e.g. after cancel)."""
    global _active_model
    with _active_pipes_lock:
        _active_pipes.clear()
    for state in MODELS.values():
        state.pipe = None
        state.processor = None
        state.device_mode = ""
        state.loading = False
    _active_model = None
    try:
        _free_gpu_memory()
    except Exception as exc:  # noqa: BLE001
        _log.warning("release_gpu_vram: %s", exc)


def _load_pipeline(
    pipe_cls,
    repo_id: str,
    pipeline: PipelineProgress | None,
    cancel_checker: CancelFn | None = None,
    **kwargs,
):
    _ensure_repo_files(repo_id, pipeline)
    _abort_if(cancel_checker)

    def _load():
        _abort_if(cancel_checker)
        return pipe_cls.from_pretrained(
            repo_id,
            local_files_only=True,
            **kwargs,
        )

    with _global_load_lock:
        try:
            return _run_with_load_pulse(
                pipeline, _load, cancel_checker=cancel_checker
            )
        except (OSError, FileNotFoundError, ValueError):
            if pipeline:
                pipeline.report("load_model", 0.08, detail="повторная загрузка файлов модели")
            _download_repo(repo_id, pipeline)
            _abort_if(cancel_checker)
            return _run_with_load_pulse(
                pipeline,
                lambda: pipe_cls.from_pretrained(repo_id, local_files_only=True, **kwargs),
                cancel_checker=cancel_checker,
            )


def _import_torch():
    global _torch
    with _torch_lock:
        if _torch is not None:
            return _torch
        if sys.platform == "win32":
            import importlib.util

            spec = importlib.util.find_spec("torch")
            if spec and spec.origin:
                lib = Path(spec.origin).parent / "lib"
                if lib.is_dir():
                    os.add_dll_directory(str(lib))
        import torch as th

        _torch = th
        return _torch


def _prepare_image(image_bytes: bytes, *, max_side: int | None = None) -> Image.Image:
    """Load RGB image at source resolution (optional downscale); align to 8px for diffusion."""
    from PIL import ImageOps
    from output_resolution import align_dimensions, fit_long_edge

    img = Image.open(io.BytesIO(image_bytes))
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    img = img.convert("RGB")
    if max_side is not None and max(img.size) > max_side:
        w, h = fit_long_edge(img.size[0], img.size[1], max_side)
        img = img.resize((w, h), Image.Resampling.LANCZOS)
    w, h = align_dimensions(img.size[0], img.size[1], 8)
    if (w, h) != img.size:
        img = img.resize((w, h), Image.Resampling.LANCZOS)
    return img


def _prepare_image_for_analysis(
    image_bytes: bytes,
    *,
    max_side: int = _MAX_SIDE_JOYCAPTION_ANALYSIS,
) -> Image.Image:
    """Downscale for JoyCaption only — source megapixels must not affect analysis speed."""
    from PIL import ImageOps

    img = Image.open(io.BytesIO(image_bytes))
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    # JPEG draft decode avoids materializing full multi-megapixel buffers when possible.
    try:
        if getattr(img, "format", None) == "JPEG":
            img.draft("RGB", (max_side, max_side))
    except Exception:
        pass
    img = img.convert("RGB")
    if max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return img


def _latent_dims(pipe, image: Image.Image) -> tuple[int, int]:
    """Pixel height/width used by the pipeline for unpack/decode (matches diffusers)."""
    return _pipeline_pixel_dims(pipe, image)


def _pipeline_pixel_dims(pipe, image: Image.Image) -> tuple[int, int]:
    from output_resolution import pipeline_dimensions

    factor = getattr(pipe, "vae_scale_factor", 8)
    w, h = image.size
    return pipeline_dimensions(
        w,
        h,
        pipe_class_name=pipe.__class__.__name__,
        vae_scale_factor=factor,
    )


def _pipeline_is_qwen(pipe) -> bool:
    return "Qwen" in type(pipe).__name__


def _scheduler_sigma_list(pipe) -> list[float]:
    sigmas = getattr(pipe.scheduler, "sigmas", None)
    if sigmas is None:
        return []
    vals = [float(x) for x in sigmas.detach().cpu().reshape(-1).tolist()]
    timesteps = getattr(pipe.scheduler, "timesteps", None)
    n_steps = int(timesteps.numel()) if timesteps is not None else 0
    if n_steps and len(vals) == n_steps + 1:
        vals = vals[:n_steps]
    return vals


def _resume_sigmas(pipe, meta: dict, completed: int, remaining_ts: list, *, qwen_edit: bool) -> list[float]:
    total_steps = int(meta.get("total_steps") or 0)
    if qwen_edit and total_steps > 0:
        # Qwen-Image-Edit-2509 rebuilds the schedule as linspace(1, 1/N, N) then applies mu.
        # Pass the remaining PRE-shift sigmas so mu is applied once, not a fresh 1.0 start.
        import numpy as np

        full = np.linspace(1.0, 1.0 / total_steps, total_steps)
        return [float(x) for x in full[completed:]]
    saved = meta.get("sigmas") or []
    if saved:
        return [float(x) for x in saved[completed:]]
    num_train = int(getattr(pipe.scheduler.config, "num_train_timesteps", 1000) or 1000)
    return [float(t) / num_train for t in remaining_ts]


def _flux_mu(pipe, image_seq_len: int) -> float:
    from diffusers.pipelines.flux.pipeline_flux_kontext import calculate_shift

    sched = pipe.scheduler
    return float(
        calculate_shift(
            image_seq_len,
            sched.config.get("base_image_seq_len", 256),
            sched.config.get("max_image_seq_len", 4096),
            sched.config.get("base_shift", 0.5),
            sched.config.get("max_shift", 1.15),
        )
    )


def _step_callback(
    pipeline: PipelineProgress | None,
    total_steps: int,
    cancel_checker: CancelFn | None,
    timing: "DiffusionStepTracker",
    job_id: str | None = None,
    *,
    pause_checker: PauseFn | None = None,
    checkpoint_meta: dict | None = None,
    flux_kontext: bool = False,
    qwen_edit: bool = False,
    step_base: int = 0,
):
    from step_timing import format_duration

    def _cb(_pipe, step_index: int, _timestep, callback_kwargs):
        _check_control(cancel_checker, pause_checker)
        # step_index is 0-based within the current pipe() call (remaining steps on resume).
        step = int(step_base) + step_index + 1
        duration = timing.complete_step(step)
        latents = callback_kwargs.get("latents")
        checkpoint_ok = False
        if latents is not None and job_id and checkpoint_meta is not None:
            if "timesteps" not in checkpoint_meta and hasattr(_pipe, "scheduler"):
                try:
                    checkpoint_meta["timesteps"] = [
                        float(t) for t in _pipe.scheduler.timesteps.cpu().tolist()
                    ]
                    sigmas = _scheduler_sigma_list(_pipe)
                    if sigmas:
                        checkpoint_meta["sigmas"] = sigmas
                except Exception as exc:  # noqa: BLE001
                    if job_id:
                        from job_logger import log_generation_info

                        log_generation_info(
                            job_id,
                            "checkpoint_timesteps_capture_failed",
                            error=str(exc),
                        )
            if (flux_kontext or qwen_edit) and "mu" not in checkpoint_meta:
                try:
                    checkpoint_meta["mu"] = _flux_mu(_pipe, int(latents.shape[1]))
                except Exception as exc:  # noqa: BLE001
                    if job_id:
                        from job_logger import log_generation_info

                        log_generation_info(job_id, "checkpoint_mu_capture_failed", error=str(exc))
            from job_checkpoint import save_checkpoint

            save_checkpoint(
                job_id,
                completed_step=step,
                latents=latents,
                meta=checkpoint_meta,
            )
            checkpoint_ok = True
            if job_id:
                from job_logger import log_generation_info

                log_generation_info(
                    job_id,
                    "checkpoint_saved",
                    step=step,
                    total_steps=total_steps,
                    has_timesteps=bool(checkpoint_meta.get("timesteps")),
                    has_mu=bool(checkpoint_meta.get("mu")),
                    has_sigmas=bool(checkpoint_meta.get("sigmas")),
                )
        if job_id:
            from job_logger import log_job

            log_job(
                job_id,
                f"diffusion step {step}/{total_steps} took {format_duration(duration)}",
            )
        if pipeline:
            detail = timing.detail_step_completed(step, duration)
            if checkpoint_ok:
                detail = f"{detail} · чекпоинт сохранён (откат при паузе)"
            pipeline.report(
                "generate",
                1.0,
                detail=detail,
                sub_step=step,
                sub_total=total_steps,
                log_journal=True,
                journal_replace=True,
                checkpoint=checkpoint_ok,
            )
        return callback_kwargs

    return _cb


def _run_pipe(
    pipe,
    *,
    pipeline: PipelineProgress | None,
    total_steps: int,
    cancel_checker: CancelFn | None = None,
    pause_checker: PauseFn | None = None,
    preview_image: Image.Image | None = None,
    job_id: str | None = None,
    model_id: str = "unknown",
    flux_kontext: bool = False,
    resume_job_id: str | None = None,
    **kwargs,
):
    from step_timing import DiffusionStepTracker

    pipe.set_progress_bar_config(disable=True)
    register_job_pipe(job_id, pipe)
    torch = _import_torch()
    device = getattr(pipe, "_execution_device", None) or getattr(pipe, "device", None)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qwen_edit = _pipeline_is_qwen(pipe)

    checkpoint_meta: dict | None = None
    resumed = False
    step_base = 0
    if resume_job_id:
        from job_checkpoint import load_checkpoint

        loaded = load_checkpoint(resume_job_id)
        if loaded:
            meta, latents_cpu = loaded
            completed = int(meta["completed_step"])
            total_steps = int(meta["total_steps"])
            remaining_ts = meta.get("timesteps", [])[completed:]
            if remaining_ts:
                dtype = getattr(pipe, "dtype", torch.float16)
                kwargs["latents"] = latents_cpu.to(device=device, dtype=dtype)
                kwargs["num_inference_steps"] = len(remaining_ts)
                kwargs.pop("timesteps", None)
                remaining_sigmas = _resume_sigmas(
                    pipe, meta, completed, remaining_ts, qwen_edit=qwen_edit
                )
                if flux_kontext or qwen_edit:
                    kwargs["sigmas"] = remaining_sigmas
                resumed = True
                step_base = completed
                # Keep saving checkpoints after resume — previously this was None and
                # progress was lost on the next crash/pause.
                checkpoint_meta = dict(meta)
                checkpoint_meta["job_id"] = job_id or resume_job_id
                checkpoint_meta["model_id"] = model_id
                checkpoint_meta["total_steps"] = total_steps
                checkpoint_meta["flux_kontext"] = flux_kontext
                checkpoint_meta["qwen_edit"] = qwen_edit
                if remaining_sigmas and not checkpoint_meta.get("sigmas"):
                    checkpoint_meta["sigmas"] = [0.0] * completed + remaining_sigmas
                if job_id:
                    from job_logger import log_generation_info, log_job

                    log_job(
                        job_id,
                        f"RESUME from step {completed}/{total_steps} ({len(remaining_ts)} left)",
                    )
                    log_generation_info(
                        job_id,
                        "resume_start",
                        model_id=model_id,
                        completed_step=completed,
                        total_steps=total_steps,
                        remaining_steps=len(remaining_ts),
                        flux_kontext=flux_kontext,
                        qwen_edit=qwen_edit,
                        has_mu=bool(meta.get("mu")),
                        has_sigmas=bool(remaining_sigmas),
                    )
            elif job_id:
                from job_logger import log_generation_info

                log_generation_info(
                    job_id,
                    "resume_skipped_no_timesteps",
                    model_id=model_id,
                    completed_step=completed,
                    total_steps=total_steps,
                )

    timing = DiffusionStepTracker(total_steps, model_id=model_id)
    if resumed:
        timing.begin_step(step_base + 1)
    else:
        timing.begin_step(1)

    kwargs.setdefault("num_inference_steps", total_steps)
    dim_label = ""
    gen_h = gen_w = 0
    if preview_image is not None:
        gen_h, gen_w = _pipeline_pixel_dims(pipe, preview_image)
        kwargs.setdefault("height", gen_h)
        kwargs.setdefault("width", gen_w)
        if flux_kontext:
            kwargs.setdefault("max_area", gen_w * gen_h)
            kwargs.setdefault("_auto_resize", False)
            img = kwargs.get("image")
            if img is not None and img.size != (gen_w, gen_h):
                kwargs["image"] = img.resize((gen_w, gen_h), Image.Resampling.LANCZOS)
        from output_resolution import format_dimensions_label

        dim_label = format_dimensions_label(gen_w, gen_h)

    if not resumed and job_id:
        checkpoint_meta = {
            "job_id": job_id,
            "model_id": model_id,
            "total_steps": total_steps,
            "flux_kontext": flux_kontext,
            "height": gen_h,
            "width": gen_w,
            "prompt": kwargs.get("prompt"),
            "guidance_scale": kwargs.get("guidance_scale"),
            "negative_prompt": kwargs.get("negative_prompt"),
            "true_cfg_scale": kwargs.get("true_cfg_scale"),
        }
    elif checkpoint_meta is not None and preview_image is not None:
        checkpoint_meta.setdefault("height", gen_h)
        checkpoint_meta.setdefault("width", gen_w)

    kwargs["callback_on_step_end"] = _step_callback(
        pipeline,
        total_steps,
        cancel_checker,
        timing,
        job_id=job_id,
        pause_checker=pause_checker,
        checkpoint_meta=checkpoint_meta,
        flux_kontext=flux_kontext,
        qwen_edit=qwen_edit,
        step_base=step_base,
    )

    start_step = timing.working_step()
    start_detail = (
        f"генерация {dim_label} — шаг {start_step}/{total_steps}, "
        "первая итерация может занять несколько минут"
        if dim_label
        else f"шаг {start_step}/{total_steps}, первая итерация может занять несколько минут"
    )
    if resumed:
        start_detail = f"возобновление — шаг {start_step}/{total_steps}"
    if pipeline:
        pipeline.report(
            "generate",
            0.0,
            detail=start_detail,
            sub_step=start_step,
            sub_total=total_steps,
            log_journal=True,
            journal_replace=False,
        )

    stop = threading.Event()

    def wait_pulse() -> None:
        tick = 0
        while not stop.wait(2):
            if cancel_checker and cancel_checker():
                interrupt_job_pipe(job_id or "")
                return
            if pause_checker and pause_checker():
                interrupt_job_pipe(job_id or "")
                return
            tick += 1
            if pipeline:
                working = timing.working_step()
                elapsed = timing.elapsed_on_current_step()
                est = timing.estimate_step_sec()
                if est and est > 0:
                    in_step = min(0.95, elapsed / est)
                else:
                    in_step = min(0.12, tick * 0.02)
                pipeline.report(
                    "generate",
                    in_step,
                    detail=timing.detail_waiting(),
                    sub_step=working,
                    sub_total=total_steps,
                    log_journal=True,
                    journal_replace=True,
                )

    waiter = threading.Thread(target=wait_pulse, daemon=True)
    waiter.start()
    if job_id:
        from job_logger import log_job

        log_job(job_id, f"pipe() start model={model_id} {dim_label or 'no-dims'} steps={total_steps} resumed={resumed}")
    try:
        with torch.inference_mode():
            result = pipe(**kwargs)
            _check_control(cancel_checker, pause_checker)
            return result.images[0]
    except Exception as exc:  # noqa: BLE001
        if job_id:
            from job_logger import log_generation_error

            log_generation_error(
                job_id,
                exc,
                model_id=model_id,
                phase="generate",
                resumed=resumed,
                step=timing.working_step(),
                total_steps=total_steps,
                flux_kontext=flux_kontext,
                dim_label=dim_label,
                has_latents=bool(kwargs.get("latents") is not None),
                has_sigmas=bool(kwargs.get("sigmas") is not None),
            )
        raise
    finally:
        stop.set()
        waiter.join(timeout=0.2)
        clear_job_pipe(job_id)


def _image_to_png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@dataclass
class ModelState:
    model_id: str
    name: str
    description: str
    size_gb: str
    gated: bool
    repo_id: str
    kind: str = "flux"
    chain_to: str | None = None
    usage_hint: str = ""
    pipe: Any = None
    processor: Any = None
    device_mode: str = ""
    loading: bool = False
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


QWEN_IMAGE_EDIT_REPO = "Qwen/Qwen-Image-Edit-2509"
QWEN_IMAGE_EDIT_2511_REPO = "Qwen/Qwen-Image-Edit-2511"
QWEN_IMAGE_T2I_REPO = "Qwen/Qwen-Image-2512"
QWEN_EDIT_2509_IDS = frozenset({"qwen35-hauhau", "qwen36-hauhau"})
QWEN_EDIT_2511_IDS = frozenset({"qwen-edit-2511"})
QWEN_EDIT_IDS = QWEN_EDIT_2509_IDS | QWEN_EDIT_2511_IDS
QWEN_T2I_IDS = frozenset({"qwen-image-2512"})

_REFERENCE_JOYCAPTION_INSTRUCTION = (
    "Describe the visible physical features in this reference photo: "
    "hairstyle, hair color, body shape, skin marks, moles, tattoos, and anatomy. "
    "One concise English paragraph for an image editor."
)

MODELS: dict[str, ModelState] = {
    "flux-kontext": ModelState(
        model_id="flux-kontext",
        name="FLUX.1 Kontext",
        description="Редактирование людей, одежды и сцены по текстовому промпту.",
        size_gb="~12",
        gated=True,
        repo_id="black-forest-labs/FLUX.1-Kontext-dev",
        kind="flux",
        usage_hint=(
            "Нужны HuggingFace login и принятая лицензия модели. "
            "Первый запуск загружает веса в GPU (~2–5 мин)."
        ),
    ),
    "qwen35-hauhau": ModelState(
        model_id="qwen35-hauhau",
        name="Qwen3.5 HauhauCS Aggressive",
        description="Qwen-Image-Edit: ретушь тела и лица. Опционально — фото-примеры (Image 2, 3…). Перевод — HauhauCS Qwen3.5 через Ollama.",
        size_gb="~20",
        gated=False,
        repo_id=QWEN_IMAGE_EDIT_REPO,
        kind="qwen_edit",
        usage_hint=(
            "Первый запуск: загрузка Qwen-Image-Edit-2509 в GPU 5–15 мин. "
            "Опционально — фото-примеры; в промпте укажите, что взять с Image 2, 3…"
        ),
    ),
    "qwen36-hauhau": ModelState(
        model_id="qwen36-hauhau",
        name="Qwen3.6 HauhauCS Aggressive",
        description="Qwen-Image-Edit: ретушь тела и лица. Опционально — фото-примеры (Image 2, 3…). Перевод — HauhauCS Qwen3.6 через Ollama.",
        size_gb="~20",
        gated=False,
        repo_id=QWEN_IMAGE_EDIT_REPO,
        kind="qwen_edit",
        usage_hint=(
            "Первый запуск: загрузка Qwen-Image-Edit-2509 в GPU 5–15 мин. "
            "Опционально — фото-примеры; в промпте укажите, что взять с Image 2, 3…"
        ),
    ),
    "qwen-edit-2511": ModelState(
        model_id="qwen-edit-2511",
        name="Qwen Edit 2511",
        description="Qwen-Image-Edit-2511: более новая правка по фото. Примеры Image 2, 3… как у 2509. Перевод — HauhauCS Qwen3.6.",
        size_gb="~20",
        gated=False,
        repo_id=QWEN_IMAGE_EDIT_2511_REPO,
        kind="qwen_edit",
        usage_hint=(
            "Отдельные веса от 2509 — в 24 ГБ одновременно не держатся. "
            "Первый запуск скачает Qwen-Image-Edit-2511. Официально ~40 шагов."
        ),
    ),
    "qwen-image-2512": ModelState(
        model_id="qwen-image-2512",
        name="Qwen Image 2512",
        description="Текст → картинка без исходного фото. Apache 2.0. Перевод — HauhauCS Qwen3.6.",
        size_gb="~20",
        gated=False,
        repo_id=QWEN_IMAGE_T2I_REPO,
        kind="qwen_t2i",
        usage_hint=(
            "Черновик только с промптом, без основного фото. "
            "Не смешивайте в одной генерации с задачами на правку фото."
        ),
    ),
    "joycaption": ModelState(
        model_id="joycaption",
        name="JoyCaption → FLUX",
        description="JoyCaption анализирует фото и строит промпт, затем FLUX Kontext редактирует изображение.",
        size_gb="~20",
        gated=False,
        repo_id="fancyfeast/llama-joycaption-beta-one-hf-llava",
        kind="joycaption",
        chain_to="flux-kontext",
        usage_hint=(
            "Сначала JoyCaption + подтверждение промпта, затем FLUX. "
            "Нужны JoyCaption (~8 GB) и FLUX Kontext."
        ),
    ),
    "test": ModelState(
        model_id="test",
        name="Тест",
        description="Проверка цепочки без генерации: JoyCaption, перевод, подтверждение промпта.",
        size_gb="0",
        gated=False,
        repo_id="fancyfeast/llama-joycaption-beta-one-hf-llava",
        kind="test",
        usage_hint=(
            "Та же цепочка, что JoyCaption, но GPU-генерация пропускается — "
            "возвращается подготовленное исходное фото."
        ),
    ),
}

DEFAULT_MODEL = "flux-kontext"
TEST_MODEL_ID = "test"
JOYCAPTION_FLOW_IDS = frozenset({"joycaption", TEST_MODEL_ID})

# Fixed inference params (prompt drives the edit; no user-facing strength slider).
FLUX_GUIDANCE_SCALE = 3.5
QWEN_TRUE_CFG_SCALE = 4.35
QWEN_INFERENCE_STEPS = 37
FLUX_INFERENCE_STEPS = 28

MODEL_STEP_PRESETS: dict[str, list[int]] = {
    "flux-kontext": [10, 15, 20, 28],
    "qwen35-hauhau": [10, 15, 20, 25, 37],
    "qwen36-hauhau": [10, 15, 20, 25, 37],
    "joycaption": [10, 15, 20, 28],
    "test": [10],
}
MODEL_DEFAULT_STEPS: dict[str, int] = {
    "flux-kontext": FLUX_INFERENCE_STEPS,
    "qwen35-hauhau": QWEN_INFERENCE_STEPS,
    "qwen36-hauhau": QWEN_INFERENCE_STEPS,
    "joycaption": FLUX_INFERENCE_STEPS,
    "test": 10,
}


def uses_joycaption_flow(model_id: str) -> bool:
    """JoyCaption + подтверждение промпта (для joycaption и тестовой модели)."""
    state = MODELS.get(model_id)
    return state is not None and state.kind in ("joycaption", "test")


def is_test_model(model_id: str) -> bool:
    state = MODELS.get(model_id)
    return state is not None and state.kind == "test"

_active_model: str | None = None


def _qwen_pipeline_owner() -> str | None:
    for mid in QWEN_EDIT_IDS:
        if MODELS[mid].pipe is not None:
            return mid
    return None


def _share_qwen_pipeline(target_id: str) -> bool:
    owner = _qwen_pipeline_owner()
    if owner is None:
        return False
    MODELS[target_id].pipe = MODELS[owner].pipe
    return True


def _model_ready(state: ModelState) -> bool:
    if state.kind == "test":
        return True
    if state.kind == "joycaption":
        return state.pipe is not None and state.processor is not None
    if state.kind == "qwen_edit":
        return state.pipe is not None or _qwen_pipeline_owner() is not None
    return state.pipe is not None


def list_models() -> list[dict]:
    """Model list for UI. Disk cache is checked once per repo (local HF snapshot)."""
    from prompt_helper import resolve_ollama_translator

    repo_cached: dict[str, bool] = {}
    out = []
    for m in MODELS.values():
        rid = m.repo_id
        if m.kind == "test":
            cached = True
        elif rid:
            if rid not in repo_cached:
                repo_cached[rid] = _repo_is_complete(rid)
            cached = repo_cached[rid]
        else:
            cached = False
        entry = {
                "id": m.model_id,
                "name": m.name,
                "description": m.description,
                "usage_hint": m.usage_hint,
                "size_gb": m.size_gb,
                "gated": m.gated,
                "ready": _model_ready(m),
                "loading": m.loading,
                "error": m.error,
                "cached": cached,
                "selectable": m.kind == "test" or cached or _model_ready(m) or m.kind == "qwen_edit",
                "joycaption_flow": uses_joycaption_flow(m.model_id),
                "step_presets": MODEL_STEP_PRESETS.get(
                    resolve_gpu_chain_model(m.model_id),
                    MODEL_STEP_PRESETS.get(m.model_id, [28]),
                ),
                "default_steps": MODEL_DEFAULT_STEPS.get(
                    resolve_gpu_chain_model(m.model_id),
                    MODEL_DEFAULT_STEPS.get(m.model_id, 28),
                ),
            }
        if m.model_id in QWEN_EDIT_IDS:
            entry["ollama_translator"] = resolve_ollama_translator(m.model_id)
        out.append(entry)
    return out


def _unload_other_models(keep_id: str) -> None:
    torch = _import_torch()
    keep = MODELS[keep_id]
    for mid, state in MODELS.items():
        if mid == keep_id:
            continue
        if keep.kind == "qwen_edit" and state.kind == "qwen_edit":
            continue
        state.pipe = None
        state.processor = None
    _free_gpu_memory()


def _load_flux_kontext(
    state: ModelState,
    pipeline: PipelineProgress | None = None,
    cancel_checker: CancelFn | None = None,
) -> None:
    torch = _import_torch()
    from diffusers import FluxKontextPipeline

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipe = _load_pipeline(
        FluxKontextPipeline, state.repo_id, pipeline, cancel_checker, torch_dtype=dtype
    )
    mode = _place_diffusion_pipe(
        pipe, pipeline=pipeline, label="FLUX Kontext", cancel_checker=cancel_checker
    )
    if hasattr(pipe, "vae") and pipe.vae is not None:
        pipe.vae.enable_tiling()
    state.pipe = pipe
    state.device_mode = mode


def _load_qwen_edit(
    state: ModelState,
    pipeline: PipelineProgress | None = None,
    cancel_checker: CancelFn | None = None,
) -> None:
    if _share_qwen_pipeline(state.model_id):
        if pipeline:
            pipeline.report("load_model", 1.0, detail="Qwen-Image-Edit уже в GPU")
        return

    torch = _import_torch()
    from diffusers import QwenImageEditPlusPipeline

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipe = _load_pipeline(
        QwenImageEditPlusPipeline,
        state.repo_id,
        pipeline,
        cancel_checker,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    mode = _place_diffusion_pipe(
        pipe, pipeline=pipeline, label="Qwen-Image-Edit-2509", cancel_checker=cancel_checker
    )
    # Tiling on Qwen VAE often produces flat/block color canvases; 24 GB does not need it.
    if hasattr(pipe, "vae") and pipe.vae is not None and hasattr(pipe.vae, "disable_tiling"):
        pipe.vae.disable_tiling()
    state.pipe = pipe
    state.device_mode = mode
    for mid in QWEN_EDIT_IDS:
        MODELS[mid].pipe = pipe
        MODELS[mid].device_mode = mode


def _load_joycaption(
    state: ModelState,
    pipeline: PipelineProgress | None = None,
    cancel_checker: CancelFn | None = None,
    progress_phase: str = "load_model",
) -> None:
    torch = _import_torch()
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def _load():
        _abort_if(cancel_checker)
        processor = AutoProcessor.from_pretrained(state.repo_id)
        if torch.cuda.is_available():
            if _gpu_vram_gb() >= 18:
                model = LlavaForConditionalGeneration.from_pretrained(
                    state.repo_id,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                )
                model.to("cuda")
            else:
                model = LlavaForConditionalGeneration.from_pretrained(
                    state.repo_id,
                    torch_dtype=dtype,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
        else:
            model = LlavaForConditionalGeneration.from_pretrained(
                state.repo_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            model.to("cpu")
        return processor, model

    processor, model = _run_with_load_pulse(
        pipeline,
        _load,
        cancel_checker=cancel_checker,
        phase_id=progress_phase,
        details=(
            [
                "загрузка JoyCaption...",
                "чтение весов анализатора...",
                "подготовка анализа фото...",
                "ещё немного...",
            ]
            if progress_phase == "translate"
            else None
        ),
    )
    model.eval()
    state.processor = processor
    state.pipe = model


def _joycaption_prompt(
    state: ModelState,
    image: Image.Image,
    user_prompt: str,
    pipeline: PipelineProgress | None = None,
    cancel_checker: CancelFn | None = None,
) -> str:
    torch = _import_torch()
    from transformers import StoppingCriteria, StoppingCriteriaList

    processor = state.processor
    model = state.pipe

    # Analysis image must already be small; clamp again as a safety net.
    w, h = image.size
    if max(w, h) > _MAX_SIDE_JOYCAPTION_ANALYSIS:
        image = image.copy()
        image.thumbnail(
            (_MAX_SIDE_JOYCAPTION_ANALYSIS, _MAX_SIDE_JOYCAPTION_ANALYSIS),
            Image.Resampling.LANCZOS,
        )

    instruction = (
        f"The user wants to edit this photo. User instruction: {user_prompt}. "
        "Write one detailed English instruction for an AI image editor. "
        "Describe only the desired visual change. Be explicit. One paragraph, no quotes."
    )
    convo = [
        {"role": "system", "content": "You are JoyCaption, an expert image captioning assistant."},
        {"role": "user", "content": instruction},
    ]
    convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[convo_string], images=[image], return_tensors="pt")

    if hasattr(model, "device"):
        device = model.device
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)

    class _CancelStop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):  # noqa: ANN001
            return bool(cancel_checker and cancel_checker())

    stop = threading.Event()
    aw, ah = image.size

    def pulse() -> None:
        tick = 0
        while not stop.wait(4):
            if cancel_checker and cancel_checker():
                return
            tick += 1
            if pipeline:
                pipeline.report(
                    "translate",
                    min(0.9, 0.25 + tick * 0.06),
                    detail=f"JoyCaption анализирует {aw}×{ah} ({tick * 4} с)...",
                    journal_replace=True,
                )

    worker = threading.Thread(target=pulse, daemon=True)
    worker.start()
    try:
        _abort_if(cancel_checker)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                stopping_criteria=StoppingCriteriaList([_CancelStop()]),
            )
        _abort_if(cancel_checker)
    finally:
        stop.set()
        worker.join(timeout=0.2)

    trimmed = output_ids[0, inputs["input_ids"].shape[1] :]
    return processor.tokenizer.decode(trimmed, skip_special_tokens=True).strip()


_LOADERS = {
    "flux-kontext": _load_flux_kontext,
    "qwen35-hauhau": _load_qwen_edit,
    "qwen36-hauhau": _load_qwen_edit,
    "joycaption": _load_joycaption,
}


def ensure_model_loaded(
    model_id: str,
    pipeline: PipelineProgress | None = None,
    cancel_checker: CancelFn | None = None,
    *,
    progress_phase: str = "load_model",
) -> None:
    global _active_model
    _abort_if(cancel_checker)
    if model_id not in MODELS:
        raise ValueError(f"Неизвестная модель: {model_id}")

    state = MODELS[model_id]
    _invalidate_gpu_models_if_mode_changed()
    if state.kind == "test":
        if pipeline:
            pipeline.report(progress_phase, 1.0, detail="тест — загрузка GPU не требуется")
        _active_model = model_id
        return
    if _model_ready(state):
        if pipeline:
            pipeline.report(progress_phase, 1.0, detail="модель уже в памяти")
        _active_model = model_id
        return

    with state.lock:
        _abort_if(cancel_checker)
        if _model_ready(state):
            if pipeline:
                pipeline.report(progress_phase, 1.0, detail="модель уже в памяти")
            _active_model = model_id
            return
        state.loading = True
        state.error = None
        try:
            _unload_other_models(model_id)
            loader = _LOADERS[model_id]
            try:
                loader(
                    state,
                    pipeline,
                    cancel_checker,
                    progress_phase=progress_phase,
                )
            except TypeError:
                loader(state, pipeline, cancel_checker)
            _active_model = model_id
            if pipeline:
                pipeline.report(progress_phase, 1.0, detail="модель готова")
        except Exception as exc:  # noqa: BLE001
            state.error = str(exc)
            raise
        finally:
            state.loading = False


def _edit_test(
    image: Image.Image,
    pipeline: PipelineProgress | None = None,
    cancel_checker: CancelFn | None = None,
) -> Image.Image:
    _abort_if(cancel_checker)
    if pipeline:
        pipeline.report("load_model", 1.0, detail="тест — FLUX не загружается")
        pipeline.report("generate", 0.0, detail="тест — генерация пропущена")
    _abort_if(cancel_checker)
    if pipeline:
        pipeline.report("generate", 1.0, detail="тест — возвращён оригинал")
    return image.copy()


def _edit_flux(
    state: ModelState,
    image: Image.Image,
    prompt: str,
    pipeline: PipelineProgress | None = None,
    cancel_checker: CancelFn | None = None,
    pause_checker: PauseFn | None = None,
    job_id: str | None = None,
    model_id: str = "flux-kontext",
    resume_job_id: str | None = None,
    inference_steps: int | None = None,
) -> Image.Image:
    torch = _import_torch()
    generator, _seed = _diffusion_generator(torch)
    steps = inference_steps or FLUX_INFERENCE_STEPS
    gen_h, gen_w = _pipeline_pixel_dims(state.pipe, image)
    if image.size != (gen_w, gen_h):
        image = image.resize((gen_w, gen_h), Image.Resampling.LANCZOS)
    if job_id:
        from job_logger import log_job

        preview = prompt if len(prompt) <= 220 else prompt[:217] + "..."
        log_job(job_id, f"FLUX prompt: {preview}")
    return _run_pipe(
        state.pipe,
        pipeline=pipeline,
        total_steps=steps,
        cancel_checker=cancel_checker,
        pause_checker=pause_checker,
        preview_image=image,
        job_id=job_id,
        model_id=model_id,
        flux_kontext=True,
        resume_job_id=resume_job_id,
        prompt=prompt,
        image=image,
        guidance_scale=FLUX_GUIDANCE_SCALE,
        generator=generator,
    )


def augment_prompt_for_qwen_reference(prompt: str, n_refs: int = 1) -> str:
    """Prefix Image 1/2/… hints when Qwen receives extra reference photos."""
    if n_refs <= 0:
        return prompt
    if re.search(r"image\s*\d+|\[image\s*\d+\]", prompt, re.I):
        return prompt
    parts = ["Image 1 is the photo to edit."]
    for i in range(n_refs):
        parts.append(f"Image {i + 2} is a reference example.")
    return " ".join(parts) + " " + prompt


def describe_reference_for_prompt(
    reference_bytes: bytes,
    pipeline: PipelineProgress | None = None,
    cancel_checker: CancelFn | None = None,
    *,
    prefer_cpu: bool = False,
) -> str:
    """JoyCaption description of optional reference photo (used in prompt for FLUX / approval)."""
    image = _prepare_image_for_analysis(reference_bytes)
    if prefer_cpu:
        return _run_joycaption_ephemeral_cpu(
            image,
            _REFERENCE_JOYCAPTION_INSTRUCTION,
            pipeline=pipeline,
            cancel_checker=cancel_checker,
        ).strip()
    _free_gpu_memory()
    ensure_model_loaded(
        "joycaption",
        pipeline=pipeline,
        cancel_checker=cancel_checker,
        progress_phase="translate",
    )
    state = MODELS["joycaption"]
    if not _model_ready(state):
        raise RuntimeError(state.error or "JoyCaption не загружен")
    return _joycaption_prompt(
        state,
        image,
        _REFERENCE_JOYCAPTION_INSTRUCTION,
        pipeline=pipeline,
        cancel_checker=cancel_checker,
    ).strip()


def _edit_qwen(
    state: ModelState,
    image: Image.Image,
    prompt: str,
    pipeline: PipelineProgress | None = None,
    cancel_checker: CancelFn | None = None,
    pause_checker: PauseFn | None = None,
    job_id: str | None = None,
    model_id: str = "qwen-image-edit",
    resume_job_id: str | None = None,
    inference_steps: int | None = None,
    reference_images: list[Image.Image] | None = None,
) -> Image.Image:
    torch = _import_torch()
    generator, _seed = _diffusion_generator(torch)
    steps = inference_steps or QWEN_INFERENCE_STEPS
    refs = [img for img in (reference_images or []) if img is not None]
    pipe_image: Image.Image | list[Image.Image] = [image, *refs] if refs else image
    prompt_text = augment_prompt_for_qwen_reference(prompt, len(refs)) if refs else prompt
    with torch.inference_mode():
        return _run_pipe(
            state.pipe,
            pipeline=pipeline,
            total_steps=steps,
            cancel_checker=cancel_checker,
            pause_checker=pause_checker,
            preview_image=image,
            job_id=job_id,
            model_id=model_id,
            resume_job_id=resume_job_id,
            image=pipe_image,
            prompt=prompt_text,
            negative_prompt=" ",
            true_cfg_scale=QWEN_TRUE_CFG_SCALE,
            generator=generator,
        )


def build_joycaption_prompt(
    image_bytes: bytes,
    user_prompt: str,
    pipeline: PipelineProgress | None = None,
    cancel_checker: CancelFn | None = None,
    *,
    prefer_cpu: bool = False,
) -> tuple[str, str]:
    """Сгенерировать промпт JoyCaption и отформатировать для FLUX."""
    from prompt_helper import format_for_flux
    from output_resolution import image_dimensions_label

    src_label = image_dimensions_label(image_bytes)
    image = _prepare_image_for_analysis(
        image_bytes, max_side=_MAX_SIDE_JOYCAPTION_ANALYSIS
    )
    aw, ah = image.size
    if pipeline:
        pipeline.report(
            "translate",
            0.05,
            detail=f"анализ фото {src_label} → {aw}×{ah} для JoyCaption",
        )

    if prefer_cpu:
        if pipeline:
            pipeline.report("translate", 0.08, detail="JoyCaption на CPU (GPU занят)")
        raw = _run_joycaption_ephemeral_cpu(
            image,
            user_prompt,
            pipeline=pipeline,
            cancel_checker=cancel_checker,
        )
    else:
        # Free leftover VRAM from a previous Qwen/FLUX run so JoyCaption can load.
        _free_gpu_memory()
        ensure_model_loaded(
            "joycaption",
            pipeline=pipeline,
            cancel_checker=cancel_checker,
            progress_phase="translate",
        )
        state = MODELS["joycaption"]
        if pipeline:
            pipeline.report(
                "translate",
                0.2,
                detail=f"JoyCaption анализирует {aw}×{ah}",
            )
        raw = _joycaption_prompt(
            state,
            image,
            user_prompt,
            pipeline=pipeline,
            cancel_checker=cancel_checker,
        )
        state.pipe = None
        state.processor = None
        _free_gpu_memory()
    formatted = format_for_flux(raw)
    if pipeline and not prefer_cpu:
        pipeline.report("translate", 1.0, detail="промпт JoyCaption готов")
    return formatted, raw


def _run_joycaption_ephemeral_cpu(
    image: Image.Image,
    user_prompt: str,
    pipeline: PipelineProgress | None = None,
    cancel_checker: CancelFn | None = None,
) -> str:
    """JoyCaption on CPU without unloading models on GPU."""
    torch = _import_torch()
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    repo = MODELS["joycaption"].repo_id
    _abort_if(cancel_checker)
    processor = AutoProcessor.from_pretrained(repo)
    _abort_if(cancel_checker)
    model = LlavaForConditionalGeneration.from_pretrained(
        repo,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to("cpu")
    model.eval()
    tmp = ModelState(
        model_id="joycaption",
        name="JoyCaption",
        description="",
        size_gb="",
        gated=False,
        repo_id=repo,
        kind="joycaption",
    )
    tmp.processor = processor
    tmp.pipe = model
    try:
        return _joycaption_prompt(
            tmp, image, user_prompt, pipeline=pipeline, cancel_checker=cancel_checker
        )
    finally:
        del model, processor, tmp
        import gc

        gc.collect()


def _edit_joycaption(
    state: ModelState,
    image: Image.Image,
    prompt: str,
    pipeline: PipelineProgress | None = None,
    cancel_checker: CancelFn | None = None,
    pause_checker: PauseFn | None = None,
    on_prompt: Callable[[str, str], None] | None = None,
    flux_prompt: str | None = None,
    job_id: str | None = None,
    resume_job_id: str | None = None,
) -> Image.Image:
    from prompt_helper import format_for_flux

    if flux_prompt:
        formatted = flux_prompt
    else:
        if pipeline:
            pipeline.report("translate", 0.15, detail="JoyCaption анализирует фото")
        edit_prompt = _joycaption_prompt(
            state, image, prompt, pipeline=pipeline, cancel_checker=cancel_checker
        )
        formatted = format_for_flux(edit_prompt)
        if on_prompt:
            on_prompt(formatted, edit_prompt)
        if pipeline:
            pipeline.report("translate", 1.0, detail="промпт JoyCaption готов")

        state.pipe = None
        state.processor = None
        _free_gpu_memory()

    chain_id = state.chain_to or "flux-kontext"
    if pipeline:
        pipeline.report("load_model", 0.05, detail="загрузка FLUX Kontext (после JoyCaption)")
    ensure_model_loaded(chain_id, pipeline=pipeline, cancel_checker=cancel_checker)

    chain = MODELS[chain_id]
    if hasattr(chain.pipe, "set_progress_bar_config"):
        chain.pipe.set_progress_bar_config(disable=True)
    if pipeline:
        pipeline.report("generate", 0.0, detail=_generate_start_detail(chain_id))
    return _edit_flux(
        chain,
        image,
        formatted,
        pipeline,
        cancel_checker,
        pause_checker=pause_checker,
        job_id=job_id,
        model_id=chain_id,
        resume_job_id=resume_job_id,
    )


_EDITORS = {
    "flux-kontext": _edit_flux,
    "qwen35-hauhau": _edit_qwen,
    "qwen36-hauhau": _edit_qwen,
    "joycaption": _edit_joycaption,
}


def edit_image(
    image_bytes: bytes,
    prompt: str,
    model_id: str = DEFAULT_MODEL,
    pipeline: PipelineProgress | None = None,
    cancel_checker: CancelFn | None = None,
    pause_checker: PauseFn | None = None,
    on_prompt: Callable[[str, str], None] | None = None,
    flux_prompt: str | None = None,
    job_id: str | None = None,
    resume_job_id: str | None = None,
    inference_steps: int | None = None,
    reference_image_bytes: bytes | None = None,
    reference_images_bytes: list[bytes] | None = None,
) -> bytes:
    _check_control(cancel_checker, pause_checker)
    steps = inference_steps_for_model(model_id, inference_steps)

    chain_id = model_id
    if uses_joycaption_flow(model_id):
        if is_test_model(model_id):
            pass
        elif flux_prompt:
            chain_id = MODELS["joycaption"].chain_to or "flux-kontext"
            ensure_model_loaded(chain_id, pipeline=pipeline, cancel_checker=cancel_checker)
        else:
            ensure_model_loaded("joycaption", pipeline=pipeline, cancel_checker=cancel_checker)
    else:
        ensure_model_loaded(model_id, pipeline=pipeline, cancel_checker=cancel_checker)

    gpu_info = get_gpu_info()
    gen_state = MODELS.get(resolve_gpu_chain_model(model_id)) or MODELS.get(model_id)
    if job_id:
        from job_logger import log_job

        mode = gen_state.device_mode if gen_state else "unknown"
        log_job(
            job_id,
            f"GPU info: {gpu_info.get('device_name')} "
            f"{gpu_info.get('vram_total_gb')}GB preferred={gpu_info.get('preferred_mode')} "
            f"loaded_mode={mode} steps={steps}",
        )
        if mode == "offload" and gpu_info.get("vram_total_gb", 0) >= 18:
            log_job(job_id, "WARN: CPU offload при VRAM>=18GB — генерация будет очень медленной")

    if pipeline:
        pipeline.report("prepare", 0.1, detail="исходное разрешение фото")

    qwen_mode = model_id in QWEN_EDIT_IDS or (
        uses_joycaption_flow(model_id) and chain_id in QWEN_EDIT_IDS
    )
    qwen_max = _MAX_SIDE_QWEN_GENERATION if qwen_mode else None
    image = _prepare_image(image_bytes, max_side=qwen_max)
    ref_bytes = list(reference_images_bytes or [])
    if not ref_bytes and reference_image_bytes:
        ref_bytes = [reference_image_bytes]
    reference_images = [_prepare_image(data, max_side=qwen_max) for data in ref_bytes]
    if pipeline:
        w, h = image.size
        detail = f"генерация в разрешении {w}×{h} px"
        if qwen_max and max(w, h) < max(Image.open(io.BytesIO(image_bytes)).size):
            detail += f" (уменьшено для Qwen, макс. сторона {qwen_max})"
        if reference_images:
            sizes = ", ".join(f"{img.size[0]}×{img.size[1]}" for img in reference_images)
            detail += f" · {len(reference_images)} фото-пример(а): {sizes}"
        pipeline.report("prepare", 1.0, detail=detail)

    _abort_if(cancel_checker)

    if uses_joycaption_flow(model_id):
        state = MODELS["joycaption"]
        if is_test_model(model_id):
            if pipeline:
                pipeline.report("generate", 0.0, detail="тест — без генерации на GPU")
            result = _edit_test(image, pipeline, cancel_checker)
        elif flux_prompt:
            chain = MODELS[chain_id]
            if not _model_ready(chain):
                raise RuntimeError(chain.error or "Модель не загружена")
            if pipeline:
                pipeline.report(
                    "generate",
                    0.0,
                    detail=_generate_start_detail(chain_id),
                    journal_replace=False,
                )
            result = _edit_flux(
                chain,
                image,
                flux_prompt,
                pipeline,
                cancel_checker,
                pause_checker=pause_checker,
                job_id=job_id,
                model_id=chain_id,
                resume_job_id=resume_job_id,
                inference_steps=steps,
            )
        else:
            if not _model_ready(state):
                raise RuntimeError(state.error or "Модель не загружена")
            result = _edit_joycaption(
                state,
                image,
                prompt,
                pipeline,
                cancel_checker,
                pause_checker=pause_checker,
                on_prompt=on_prompt,
                job_id=job_id,
                resume_job_id=resume_job_id,
            )
    else:
        state = MODELS[model_id]
        if not _model_ready(state):
            raise RuntimeError(state.error or "Модель не загружена")
        if pipeline:
            pipeline.report(
                "generate",
                0.0,
                detail=_generate_start_detail(model_id),
                journal_replace=False,
            )
        _check_control(cancel_checker, pause_checker)
        editor = _EDITORS[model_id]
        edit_kwargs: dict[str, Any] = {
            "pipeline": pipeline,
            "cancel_checker": cancel_checker,
            "pause_checker": pause_checker,
            "job_id": job_id,
            "resume_job_id": resume_job_id,
        }
        if model_id in QWEN_EDIT_IDS or model_id == "flux-kontext":
            edit_kwargs["model_id"] = model_id
            edit_kwargs["inference_steps"] = steps
        if model_id in QWEN_EDIT_IDS:
            edit_kwargs["reference_images"] = reference_images
        result = editor(state, image, prompt, **edit_kwargs)

    if job_id:
        from job_checkpoint import delete_checkpoint

        delete_checkpoint(job_id)
    if pipeline:
        pipeline.report("generate", 1.0, detail="генерация завершена")
    return _image_to_png(result)
