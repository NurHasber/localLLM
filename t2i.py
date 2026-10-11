"""Text-to-image jobs (Qwen-Image-2512), separate from the photo-edit queue."""

from __future__ import annotations

import threading
import time
from typing import Any

from job_logger import get_logger

T2I_ASPECTS: dict[str, tuple[int, int]] = {
    "1:1": (1024, 1024),
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "4:3": (1024, 768),
    "3:4": (768, 1024),
}

T2I_JOB_ID = "t2i"


class T2IStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.phase = "idle"
        self.message = ""
        self.error: str | None = None
        self.progress = 0
        self.step = 0
        self.total_steps = 0
        self.prompt = ""
        self.prompt_used = ""
        self.aspect = "1:1"
        self.width = 1024
        self.height = 1024
        self.steps = 50
        self.model_id = "qwen-image-2512"
        self.result: bytes | None = None
        self.cancel = False
        self.phase_started = 0.0

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            busy = self.phase in ("translate", "load_model", "generate")
            elapsed = int(time.monotonic() - self.phase_started) if busy and self.phase_started else 0
            return {
                "phase": self.phase,
                "busy": busy,
                "message": self.message,
                "error": self.error,
                "progress": self.progress,
                "step": self.step,
                "total_steps": self.total_steps,
                "prompt": self.prompt,
                "prompt_used": self.prompt_used,
                "aspect": self.aspect,
                "width": self.width,
                "height": self.height,
                "steps": self.steps,
                "model_id": self.model_id,
                "has_result": self.result is not None,
                "elapsed_sec": elapsed,
                "aspects": {k: {"width": w, "height": h} for k, (w, h) in T2I_ASPECTS.items()},
            }


T2I = T2IStore()


def start_t2i(
    prompt: str,
    *,
    aspect: str,
    steps: int,
    gpu_busy: bool,
    acquire_gpu,
    release_gpu,
    model_id: str = "qwen-image-2512",
) -> dict[str, Any]:
    text = (prompt or "").strip()
    if not text:
        raise ValueError("Введите промпт")
    if aspect not in T2I_ASPECTS:
        raise ValueError(f"Неизвестный формат: {aspect}")
    if gpu_busy:
        raise RuntimeError("GPU занят очередью правки или сортировкой")
    from image_editor import MODELS, T2I_KINDS, inference_steps_for_model

    state = MODELS.get(model_id)
    if state is None or state.kind not in T2I_KINDS:
        raise ValueError(f"Неизвестная T2I-модель: {model_id}")
    width, height = T2I_ASPECTS[aspect]
    n_steps = inference_steps_for_model(model_id, int(steps))
    with T2I._lock:
        if T2I.phase in ("translate", "load_model", "generate"):
            raise RuntimeError("Генерация уже идёт")
        T2I.phase = "translate"
        T2I.phase_started = time.monotonic()
        T2I.prompt = text
        T2I.prompt_used = ""
        T2I.aspect = aspect
        T2I.width = width
        T2I.height = height
        T2I.steps = n_steps
        T2I.model_id = model_id
        T2I.result = None
        T2I.error = None
        T2I.cancel = False
        T2I.progress = 0
        T2I.message = "Подготовка промпта…"

    def worker() -> None:
        log = get_logger()
        acquired = False
        try:
            from image_editor import abort_generation, generate_image
            from pipeline_progress import PipelineProgress
            from prompt_helper import prepare_prompt

            def aborted() -> bool:
                with T2I._lock:
                    return T2I.cancel

            def on_update(**fields) -> None:
                with T2I._lock:
                    T2I.message = fields.get("message") or T2I.message
                    T2I.progress = int(fields.get("progress") or 0)
                    T2I.step = int(fields.get("step") or 0)
                    T2I.total_steps = int(fields.get("total_steps") or 0)
                    status = fields.get("status") or T2I.phase
                    if status in ("translate", "load_model", "generate", "done"):
                        T2I.phase = status

            pipeline = PipelineProgress(on_update)
            with T2I._lock:
                mid = T2I.model_id
            english, note, _ui = prepare_prompt(text, mid)
            if aborted():
                with T2I._lock:
                    T2I.phase = "idle"
                    T2I.message = "Отменено"
                return
            used = english or text
            with T2I._lock:
                T2I.prompt_used = used
                if note:
                    T2I.message = note
            acquire_gpu()
            acquired = True
            if aborted():
                raise RuntimeError("Отменено")
            png = generate_image(
                used,
                model_id=mid,
                width=width,
                height=height,
                inference_steps=n_steps,
                pipeline=pipeline,
                cancel_checker=aborted,
                job_id=T2I_JOB_ID,
            )
            with T2I._lock:
                T2I.result = png
                T2I.phase = "done"
                T2I.progress = 100
                T2I.message = f"Готово {width}×{height}"
        except Exception as exc:  # noqa: BLE001
            log.exception("t2i")
            abort_generation(T2I_JOB_ID)
            with T2I._lock:
                if T2I.cancel:
                    T2I.phase = "idle"
                    T2I.message = "Отменено"
                    T2I.error = None
                else:
                    T2I.phase = "error"
                    T2I.error = str(exc)
                    T2I.message = f"Ошибка: {exc}"
        finally:
            if acquired:
                try:
                    release_gpu()
                except Exception:
                    pass
            try:
                from image_editor import abort_generation

                abort_generation(T2I_JOB_ID)
            except Exception:
                pass

    threading.Thread(target=worker, daemon=True, name="t2i-generate").start()
    return T2I.snapshot()


def request_t2i_cancel() -> dict[str, Any]:
    with T2I._lock:
        T2I.cancel = True
        T2I.message = "Остановка…"
    try:
        from image_editor import abort_generation

        abort_generation(T2I_JOB_ID)
    except Exception:
        pass
    return T2I.snapshot()
