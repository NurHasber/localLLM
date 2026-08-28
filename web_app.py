#!/usr/bin/env python3
"""Queue-only web server for local photo editing."""

from __future__ import annotations

import io
import os
import threading
from pathlib import Path

import portable_env  # noqa: F401 — sets HF cache under project root

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from gpu_supervisor import run_gpu_subprocess
from image_editor import (
    MODELS,
    abort_generation,
    build_joycaption_prompt,
    edit_image,
    get_gpu_info,
    inference_steps_for_model,
    is_test_model,
    list_models,
    release_gpu_vram,
    uses_joycaption_flow,
)
from job_errors import JobCancelled, JobPaused
from job_logger import get_logger, log_job, log_job_exception, log_generation_error, log_line, open_log_in_system_editor
from job_store import JOBS, MAX_REFERENCE_IMAGES, TERMINAL_STATUSES
from output_resolution import image_dimensions_label
from pipeline_progress import PHASE_TOTAL, PipelineProgress
from prompt_helper import QWEN_MODEL_IDS, format_for_flux, format_for_qwen, ollama_available, prompt_is_english

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"

SERVER_PORT = int(os.environ.get("PHOTO_EDIT_PORT", "7860"))

app = FastAPI(title="Photo Edit")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_queue_runner_lock = threading.Lock()


class QueueReorderBody(BaseModel):
    order: list[str]


def _install_quiet_thread_interrupts() -> None:
    if not hasattr(threading, "excepthook"):
        return
    default_hook = threading.excepthook

    def hook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type in (KeyboardInterrupt, JobCancelled, JobPaused):
            return
        active_id = JOBS.active_job_id()
        if active_id and args.exc_value is not None:
            job = JOBS.get(active_id)
            if job and job.status == "paused":
                return
            log_job_exception(active_id, "thread crash", args.exc_value)
            JOBS.log_activity(active_id, f"Ошибка потока: {args.exc_value}", force=True)
            JOBS.update(
                active_id,
                status="error",
                error=str(args.exc_value),
                message=f"Ошибка: {args.exc_value}",
            )
        default_hook(args)

    threading.excepthook = hook


def _release_gpu_if_needed() -> None:
    if JOBS.is_gpu_busy():
        try:
            JOBS.gpu_lock().release()
        except RuntimeError:
            pass


def _request_prompt_approval(job_id: str, pipeline: PipelineProgress) -> None:
    job = JOBS.get(job_id)
    if not job or not job.prompt_used:
        raise RuntimeError("Промпт для подтверждения не подготовлен")
    pipeline.report("approve", 0.0, detail="проверьте и подтвердите промпт")
    JOBS.log_activity(job_id, "Ожидание подтверждения промпта")
    JOBS.update(
        job_id,
        status="approve",
        phase=2,
        progress=0,
        message="2. Подтверждение промпта — проверьте текст",
    )


def _auto_approve_after_prep(job_id: str, pipeline: PipelineProgress) -> None:
    ok, reason = JOBS.auto_approve_prompt(job_id)
    if not ok:
        raise RuntimeError(reason or "Не удалось подтвердить промпт")
    pipeline.report("approve", 1.0, detail="английский промпт — подтверждение не требуется")
    JOBS.log_activity(job_id, "Промпт подтверждён автоматически (английский)")


def _job_pipeline(job_id: str) -> PipelineProgress:
    def on_update(**fields) -> None:
        log_journal = fields.pop("log_journal", True)
        journal_replace = fields.pop("journal_replace", False)
        checkpoint = bool(fields.pop("checkpoint", False))
        JOBS.update(job_id, **fields)
        msg = fields.get("message")
        if msg and log_journal:
            JOBS.log_activity(
                job_id,
                msg,
                phase=fields.get("phase"),
                status=fields.get("status"),
                step=fields.get("step") if fields.get("step") is not None else 0,
                total_steps=fields.get("total_steps") if fields.get("total_steps") is not None else 0,
                replace_in_progress=journal_replace,
                checkpoint=checkpoint,
            )
        if msg and fields.get("status") in ("generate", "load_model", "done", "error", "cancelled"):
            log_job(job_id, msg)

    return PipelineProgress(on_update)


def _has_resume_checkpoint(job_id: str) -> bool:
    from job_checkpoint import has_checkpoint

    return has_checkpoint(job_id)


def _run_prep_job(
    job_id: str,
    raw: bytes,
    prompt: str,
    model_id: str,
    reference_list: list[bytes] | None = None,
) -> None:
    JOBS.set_prep_worker_tid(threading.get_ident())
    pipeline = _job_pipeline(job_id)
    cancel_checker = lambda: JOBS.should_abort(job_id)
    gpu_acquired = False
    JOBS.register_interrupt(job_id, lambda: abort_generation(job_id))

    refs = [data for data in (reference_list or []) if data]
    ref_kb = sum(len(data) for data in refs)
    log_job(
        job_id,
        f"PREP size={image_dimensions_label(raw)} image_kb={len(raw) // 1024} "
        f"ref_n={len(refs)} ref_kb={ref_kb // 1024} prompt_len={len(prompt)}",
    )

    try:
        JOBS.log_activity(job_id, "Подготовка промпта")

        if prompt_is_english(prompt):
            JOBS.update(
                job_id,
                status="translate",
                phase=1,
                phase_total=PHASE_TOTAL,
                progress=0,
                message="1. Промпт на английском — JoyCaption пропущен",
            )
            pipeline.report("translate", 0.0, detail="промпт на английском")
            prompt_used = format_for_qwen(prompt) if model_id in QWEN_MODEL_IDS else format_for_flux(prompt)
            preview = prompt if len(prompt) <= 160 else prompt[:157] + "..."
            JOBS.log_activity(job_id, f"Промпт: {preview}")
            JOBS.update(job_id, prompt_used=prompt_used, prompt_translation=prompt.strip(), note=None)
            pipeline.report("translate", 1.0, detail="промпт готов")
        else:
            JOBS.update(
                job_id,
                status="translate",
                phase=1,
                phase_total=PHASE_TOTAL,
                progress=0,
                message="1. JoyCaption анализирует фото...",
            )
            pipeline.report("translate", 0.0, detail="JoyCaption")
            JOBS.log_activity(job_id, "JoyCaption: анализ фото")
            if JOBS.is_gpu_busy():
                english_prompt, joy_raw = build_joycaption_prompt(
                    raw, prompt, pipeline=pipeline, cancel_checker=cancel_checker, prefer_cpu=True
                )
            else:
                JOBS.gpu_lock().acquire()
                gpu_acquired = True
                english_prompt, joy_raw = build_joycaption_prompt(
                    raw, prompt, pipeline=pipeline, cancel_checker=cancel_checker
                )
                JOBS.gpu_lock().release()
                gpu_acquired = False
            prompt_used = format_for_qwen(joy_raw) if model_id in QWEN_MODEL_IDS else english_prompt
            preview = joy_raw if len(joy_raw) <= 160 else joy_raw[:157] + "..."
            JOBS.log_activity(job_id, f"JoyCaption: {preview}")
            JOBS.update(job_id, prompt_used=prompt_used, prompt_translation=joy_raw, note=None)
            pipeline.report("translate", 1.0, detail="промпт готов")

        if cancel_checker():
            raise JobCancelled()

        if refs:
            if prompt_is_english(prompt):
                n = len(refs)
                JOBS.log_activity(job_id, f"{n} фото-пример(а) сохранены — JoyCaption пропущен")
                pipeline.report("translate", 0.95, detail="фото-примеры без JoyCaption")
                job = JOBS.get(job_id)
                prompt_used = (job.prompt_used if job else "") or ""
                if model_id in QWEN_MODEL_IDS:
                    from image_editor import augment_prompt_for_qwen_reference

                    prompt_used = augment_prompt_for_qwen_reference(prompt_used, n)
                    JOBS.update(job_id, prompt_used=prompt_used)
            else:
                from image_editor import augment_prompt_for_qwen_reference, describe_reference_for_prompt

                JOBS.log_activity(job_id, f"Анализ {len(refs)} фото-пример(ов)")
                pipeline.report("translate", 0.92, detail="анализ фото-примеров")
                job = JOBS.get(job_id)
                prompt_used = (job.prompt_used if job else "") or ""
                descriptions: list[str] = []
                for index, ref_bytes in enumerate(refs):
                    if cancel_checker():
                        raise JobCancelled()
                    ref_desc = describe_reference_for_prompt(
                        ref_bytes,
                        pipeline=pipeline,
                        cancel_checker=cancel_checker,
                        prefer_cpu=JOBS.is_gpu_busy(),
                    )
                    if ref_desc:
                        descriptions.append(f"Image {index + 2}: {ref_desc}")
                        preview = ref_desc if len(ref_desc) <= 120 else ref_desc[:117] + "..."
                        JOBS.log_activity(job_id, f"Пример {index + 2}: {preview}")
                if descriptions:
                    prompt_used = f"{prompt_used.rstrip('.')}. Reference example photos: " + " ".join(
                        descriptions
                    )
                if model_id in QWEN_MODEL_IDS:
                    prompt_used = augment_prompt_for_qwen_reference(prompt_used, len(refs))
                JOBS.update(job_id, prompt_used=prompt_used)

        if cancel_checker():
            raise JobCancelled()
        if prompt_is_english(prompt):
            _auto_approve_after_prep(job_id, pipeline)
        else:
            _request_prompt_approval(job_id, pipeline)
    except KeyboardInterrupt:
        log_job(job_id, "STOPPED KeyboardInterrupt")
        JOBS.log_activity(job_id, "Остановлено (Ctrl+C)", force=True)
        job = JOBS.get(job_id)
        if job and job.status in ("generate", "translate"):
            JOBS.requeue_after_cancel(job_id, job.status)
    except JobCancelled:
        log_job(job_id, "CANCELLED during prep")
        JOBS.log_activity(job_id, "Отменено — снова в очереди", force=True)
        JOBS.requeue_after_cancel(job_id)
    except Exception as exc:  # noqa: BLE001
        log_job_exception(job_id, "prep failed", exc)
        JOBS.log_activity(job_id, f"Ошибка: {exc}", force=True)
        JOBS.update(job_id, status="error", error=str(exc), message=f"Ошибка: {exc}")
    finally:
        JOBS.clear_interrupt(job_id)
        JOBS.clear_prep_active(job_id)
        JOBS.set_prep_worker_tid(None)
        if gpu_acquired and JOBS.is_gpu_busy():
            JOBS.gpu_lock().release()
        job = JOBS.get(job_id)
        if job and job.cancel_requested:
            release_gpu_vram()
        if job and job.status in TERMINAL_STATUSES:
            JOBS.finish_job(job_id)
        advance_queue()


def _run_gpu_job(job_id: str, raw: bytes, model_id: str, reference_list: list[bytes] | None = None) -> None:
    JOBS.set_gpu_worker_tid(threading.get_ident())
    pipeline = _job_pipeline(job_id)
    cancel_checker = lambda: JOBS.should_abort(job_id)
    gpu_acquired = False
    JOBS.register_interrupt(job_id, lambda: abort_generation(job_id))

    job = JOBS.get(job_id)
    english_prompt = (job.prompt_used if job else None) or ""
    inference_steps = (job.inference_steps if job else 0) or None
    refs = [data for data in (reference_list or []) if data]
    if not refs and job:
        refs = job.reference_bytes_list()
    if not english_prompt:
        JOBS.update(job_id, status="error", error="Нет промпта", message="Ошибка: нет промпта")
        return

    effective_id = model_id
    if uses_joycaption_flow(model_id) and not is_test_model(model_id):
        effective_id = MODELS["joycaption"].chain_to or "flux-kontext"
    effective_name = MODELS[effective_id].name if effective_id in MODELS else effective_id

    if refs and model_id in QWEN_MODEL_IDS:
        from image_editor import augment_prompt_for_qwen_reference

        english_prompt = augment_prompt_for_qwen_reference(english_prompt, len(refs))

    log_job(
        job_id,
        f"GPU model={model_id} effective={effective_id} size={image_dimensions_label(raw)} kb={len(raw)//1024}",
    )
    preview = english_prompt if len(english_prompt) <= 220 else english_prompt[:217] + "..."
    JOBS.log_activity(job_id, f"Промпт: {preview}")

    try:
        if is_test_model(model_id):
            JOBS.log_activity(job_id, "Тест — GPU пропущен")
            result = edit_image(
                raw,
                english_prompt,
                model_id=model_id,
                pipeline=pipeline,
                cancel_checker=cancel_checker,
                pause_checker=lambda: JOBS.should_pause(job_id),
                flux_prompt=english_prompt if uses_joycaption_flow(model_id) else None,
                job_id=job_id,
                resume_job_id=job_id if _has_resume_checkpoint(job_id) else None,
                reference_images_bytes=refs or None,
            )
        else:
            JOBS.gpu_lock().acquire()
            gpu_acquired = True
            JOBS.log_activity(job_id, f"Загрузка «{effective_name}»")
            if refs:
                JOBS.log_activity(job_id, f"{len(refs)} фото-пример(а) переданы в генерацию")
            result = run_gpu_subprocess(
                job_id,
                raw,
                model_id,
                english_prompt,
                flux_prompt=english_prompt if uses_joycaption_flow(model_id) else None,
                pipeline=pipeline,
                cancel_checker=cancel_checker,
                pause_checker=lambda: JOBS.should_pause(job_id),
                on_process=JOBS.set_gpu_process,
                resume=_has_resume_checkpoint(job_id),
                inference_steps=inference_steps,
                reference_raw=refs or None,
            )
        if cancel_checker():
            raise JobCancelled()
        JOBS.update(job_id, result=result)
        pipeline.report("done", 1.0, detail="результат сохранён")
        log_job(job_id, f"DONE bytes={len(result)}")
    except KeyboardInterrupt:
        log_job(job_id, "STOPPED KeyboardInterrupt")
        JOBS.log_activity(job_id, "Остановлено (Ctrl+C)", force=True)
        job = JOBS.get(job_id)
        if job and job.status == "generate":
            JOBS.requeue_after_cancel(job_id, "generate")
    except JobCancelled:
        if JOBS.should_pause(job_id):
            log_job(job_id, "PAUSED during GPU (cancel race)")
            JOBS.log_activity(job_id, "Пауза — GPU освобождён", force=True)
            JOBS.finish_pause(job_id)
        else:
            log_job(job_id, "CANCELLED during GPU")
            JOBS.log_activity(job_id, "Отменено — снова в очереди", force=True)
            JOBS.requeue_after_cancel(job_id)
    except JobPaused:
        log_job(job_id, "PAUSED during GPU")
        from job_checkpoint import checkpoint_completed_step

        ck_step = checkpoint_completed_step(job_id)
        if ck_step:
            JOBS.log_activity(
                job_id,
                f"Пауза — откат к чекпоинту шага {ck_step}",
                force=True,
                checkpoint=True,
                step=ck_step,
                total_steps=(JOBS.get(job_id).total_steps if JOBS.get(job_id) else 0) or 0,
            )
        else:
            JOBS.log_activity(job_id, "Пауза — GPU освобождён", force=True)
        try:
            JOBS.finish_pause(job_id)
        except Exception as exc:  # noqa: BLE001
            log_job_exception(job_id, "finish_pause failed", exc)
            JOBS.update(
                job_id,
                status="paused",
                message="На паузе",
            )
        # Subprocess kill already freed VRAM — do not import torch in this thread.
    except Exception as exc:  # noqa: BLE001
        job = JOBS.get(job_id)
        if job and (job.status == "paused" or job.pause_requested):
            log_job(job_id, f"PAUSED (treated as pause, was: {exc})")
            JOBS.log_activity(job_id, "Пауза — GPU освобождён", force=True)
            try:
                JOBS.finish_pause(job_id)
            except Exception as pause_exc:  # noqa: BLE001
                log_job_exception(job_id, "finish_pause failed", pause_exc)
                JOBS.update(job_id, status="paused", message="На паузе")
        elif _has_resume_checkpoint(job_id) and not (job and job.cancel_requested):
            log_job(job_id, f"EMERGENCY PAUSE after error: {exc}")
            from job_checkpoint import checkpoint_completed_step

            ck_step = checkpoint_completed_step(job_id)
            JOBS.log_activity(
                job_id,
                (
                    f"Экстренная пауза — откат к чекпоинту шага {ck_step}"
                    if ck_step
                    else f"Экстренная пауза после сбоя GPU: {exc}"
                ),
                force=True,
                checkpoint=bool(ck_step),
                step=ck_step or 0,
                total_steps=(job.total_steps if job else 0) or 0,
            )
            try:
                JOBS.finish_pause(job_id)
            except Exception as pause_exc:  # noqa: BLE001
                log_job_exception(job_id, "finish_pause failed", pause_exc)
                JOBS.update(job_id, status="paused", message="На паузе")
        else:
            log_job_exception(job_id, "job failed", exc)
            log_generation_error(
                job_id,
                exc,
                model_id=model_id,
                phase="gpu_job",
                resumed=_has_resume_checkpoint(job_id),
            )
            JOBS.log_activity(job_id, f"Ошибка: {exc}", force=True)
            JOBS.update(job_id, status="error", error=str(exc), message=f"Ошибка: {exc}")
    finally:
        JOBS.clear_interrupt(job_id)
        JOBS.set_active_job(None)
        JOBS.set_gpu_worker_tid(None)
        JOBS.kill_gpu_process()
        if gpu_acquired:
            try:
                JOBS.gpu_lock().release()
            except RuntimeError:
                pass
        job = JOBS.get(job_id)
        if job and job.cancel_requested:
            try:
                release_gpu_vram()
            except Exception:
                pass
        if job and job.status in TERMINAL_STATUSES:
            JOBS.finish_job(job_id)
        advance_queue()


def _schedule_prep() -> None:
    with _queue_runner_lock:
        job = JOBS.claim_next_for_prep()
        if not job:
            return
        threading.Thread(
            target=_run_prep_job,
            args=(
                job.id,
                job.image_bytes or b"",
                job.prompt,
                job.model_id,
                job.reference_bytes_list(),
            ),
            daemon=True,
        ).start()


def _schedule_gpu() -> None:
    with _queue_runner_lock:
        job = JOBS.claim_next_for_gpu()
        if not job:
            return
        JOBS.update(
            job.id,
            status="generate",
            phase=5,
            phase_total=PHASE_TOTAL,
            progress=0,
            message="5. Генерация на GPU...",
        )
        threading.Thread(
            target=_run_gpu_job,
            args=(
                job.id,
                job.image_bytes or b"",
                job.model_id,
                job.reference_bytes_list(),
            ),
            daemon=True,
        ).start()


def advance_queue() -> None:
    _schedule_prep()
    if not JOBS.has_paused_gpu_jobs():
        _schedule_gpu()


def _job_reference(job, index: int = 0):
    if not job or not job.reference_images:
        return None
    if index < 0 or index >= len(job.reference_images):
        return None
    return job.reference_images[index]


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(STATIC / "favicon.svg", media_type="image/svg+xml")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(
        (STATIC / "app.html").read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"},
    )


@app.get("/api/models")
def api_models() -> dict:
    return {
        "models": list_models(),
        "ollama": ollama_available(),
        "port": SERVER_PORT,
        "gpu": get_gpu_info(),
        "max_reference_images": MAX_REFERENCE_IMAGES,
    }


@app.post("/api/log/open")
def api_open_log() -> JSONResponse:
    try:
        path = open_log_in_system_editor()
    except OSError as exc:
        raise HTTPException(500, f"Не удалось открыть лог: {exc}") from exc
    return JSONResponse({"ok": True, "path": str(path)})


@app.get("/api/status")
def api_status() -> dict:
    active_id = JOBS.active_job_id()
    active_job = JOBS.get(active_id) if active_id else None
    queue = JOBS.queue_snapshot()
    payload: dict = {
        "gpu_busy": JOBS.is_gpu_busy(),
        "has_work": queue["has_work"],
        "pending_count": queue["pending_count"],
        "port": SERVER_PORT,
        "gpu": get_gpu_info(),
    }
    if active_job:
        payload["active_job"] = active_job.to_dict()
    return payload


@app.get("/api/queue")
def api_queue(finished_page: int = 1) -> JSONResponse:
    return JSONResponse(JOBS.queue_snapshot(finished_page=max(1, finished_page)))


@app.put("/api/queue/reorder")
def api_queue_reorder(body: QueueReorderBody) -> JSONResponse:
    ok, reason = JOBS.reorder_pending(body.order)
    if not ok:
        raise HTTPException(409, reason)
    return JSONResponse({"ok": True, "queue": JOBS.queue_snapshot()})


@app.delete("/api/queue/{job_id}")
def api_queue_remove(job_id: str) -> JSONResponse:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Задача не найдена")
    if job.status not in TERMINAL_STATUSES and job.status not in (
        "draft", "queued", "approve", "approved", "queued_gpu", "paused"
    ):
        raise HTTPException(409, "Сначала отмените задачу")
    ok, reason = JOBS.remove_queued(job_id)
    if not ok:
        raise HTTPException(409, reason)
    advance_queue()
    return JSONResponse({"ok": True, "queue": JOBS.queue_snapshot()})


@app.post("/api/queue/draft")
async def api_queue_create_draft() -> JSONResponse:
    job, reason = JOBS.create_draft()
    if not job:
        raise HTTPException(409, reason)
    return JSONResponse({"ok": True, "job_id": job.id, "queue": JOBS.queue_snapshot()})


@app.put("/api/queue/draft/{job_id}")
@app.post("/api/queue/draft/{job_id}")
async def api_queue_update_draft(
    job_id: str,
    prompt: str = Form(""),
    image: UploadFile | None = File(None),
    reference_image: UploadFile | None = File(None),
    reference_images: list[UploadFile] = File(default=[]),
    clear_reference: str = Form(""),
    remove_reference: str = Form(""),
) -> JSONResponse:
    image_bytes: bytes | None = None
    image_name: str | None = None
    if image is not None and image.filename:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(400, "Файл пустой")
        image_name = image.filename
    uploads: list[UploadFile] = []
    if reference_image is not None and reference_image.filename:
        uploads.append(reference_image)
    if reference_images:
        items = reference_images if isinstance(reference_images, list) else [reference_images]
        uploads.extend(item for item in items if item is not None and getattr(item, "filename", None))
    append_references: list[tuple[str, bytes]] = []
    for item in uploads:
        data = await item.read()
        if not data:
            raise HTTPException(400, "Файл примера пустой")
        append_references.append((item.filename or "reference", data))
    remove_index: int | None = None
    if remove_reference.strip() != "":
        try:
            remove_index = int(remove_reference.strip())
        except ValueError as exc:
            raise HTTPException(400, "Неверный номер фото-примера") from exc
    ok, reason = JOBS.update_draft(
        job_id,
        prompt=prompt,
        image_bytes=image_bytes,
        image_name=image_name,
        append_references=append_references or None,
        clear_reference=clear_reference.strip().lower() in ("1", "true", "yes"),
        remove_reference_index=remove_index,
    )
    if not ok:
        raise HTTPException(409, reason or "Черновик недоступен — обновите страницу")
    return JSONResponse({"ok": True, "queue": JOBS.queue_snapshot()})


@app.post("/api/queue/prepare")
async def api_queue_prepare() -> JSONResponse:
    count, reason, prepared_ids = JOBS.prepare_drafts()
    if count == 0:
        raise HTTPException(400, reason)
    for job_id in prepared_ids:
        JOBS.log_activity(job_id, "В очередь подготовки")
    _schedule_prep()
    return JSONResponse({"ok": True, "prepared": count, "queue": JOBS.queue_snapshot()})


@app.post("/api/queue/start-gpu")
async def api_queue_start_gpu(model: str = Form(""), steps: str = Form("")) -> JSONResponse:
    model = model.strip()
    session = JOBS.gpu_session_model()
    if JOBS.gpu_session_active() and session:
        model = session
    elif not model:
        raise HTTPException(400, "Выберите модель")
    if model not in MODELS:
        raise HTTPException(400, f"Неизвестная модель: {model}")
    try:
        inference_steps = inference_steps_for_model(
            model, int(steps) if steps.strip() else None
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    count, promoted_ids = JOBS.promote_approved_to_gpu(model, inference_steps)
    if count == 0:
        raise HTTPException(400, "Нет подтверждённых промптов")
    for job_id in promoted_ids:
        JOBS.log_activity(job_id, f"Модель: «{model}», шагов: {inference_steps}")
    _schedule_gpu()
    return JSONResponse(
        {
            "ok": True,
            "started": count,
            "model": model,
            "inference_steps": inference_steps,
            "queue": JOBS.queue_snapshot(),
        }
    )


@app.post("/api/job/{job_id}/pause")
def api_pause_job(job_id: str) -> JSONResponse:
    if not JOBS.get(job_id):
        raise HTTPException(404, "Задача не найдена")
    ok, reason = JOBS.request_pause(job_id)
    if not ok:
        raise HTTPException(409, reason)
    log_job(job_id, "PAUSE via API")
    JOBS.ensure_paused(job_id)
    return JSONResponse({"ok": True, "queue": JOBS.queue_snapshot()})


@app.post("/api/job/{job_id}/requeue")
def api_requeue_from_result(job_id: str) -> JSONResponse:
    job, reason = JOBS.create_draft_from_result(job_id)
    if not job:
        raise HTTPException(409, reason or "Не удалось вернуть в работу")
    JOBS.log_activity(job.id, "Создан из готового результата")
    log_job(job.id, f"REQUEUE from result of {job_id[:8]}")
    return JSONResponse(
        {
            "ok": True,
            "job_id": job.id,
            "queue": JOBS.queue_snapshot(),
        }
    )


@app.post("/api/job/{job_id}/resume")
def api_resume_job(job_id: str) -> JSONResponse:
    ok, reason = JOBS.resume_paused_job(job_id)
    if not ok:
        raise HTTPException(409, reason)
    JOBS.log_activity(job_id, "Возобновление генерации")
    _schedule_gpu()
    return JSONResponse({"ok": True, "queue": JOBS.queue_snapshot()})


@app.post("/api/job/{job_id}/demote-gpu")
def api_demote_gpu(job_id: str) -> JSONResponse:
    ok, reason = JOBS.demote_from_gpu(job_id)
    if not ok:
        raise HTTPException(409, reason)
    JOBS.log_activity(job_id, "Убрано из очереди генерации")
    return JSONResponse({"ok": True, "queue": JOBS.queue_snapshot()})


@app.get("/api/job/{job_id}")
def api_job(job_id: str) -> JSONResponse:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Задача не найдена")
    return JSONResponse(job.to_dict())


@app.put("/api/job/{job_id}/prompt")
async def api_edit_requeued_prompt(job_id: str, prompt: str = Form("")) -> JSONResponse:
    ok, reason = JOBS.edit_requeued_prompt(job_id, prompt)
    if not ok:
        raise HTTPException(409, reason)
    JOBS.log_activity(job_id, "Промпт изменён — повторная подготовка")
    _schedule_prep()
    advance_queue()
    return JSONResponse({"ok": True, "queue": JOBS.queue_snapshot()})


@app.post("/api/job/{job_id}/approve")
async def api_approve_job(job_id: str, prompt_used: str = Form("")) -> JSONResponse:
    ok, reason = JOBS.approve_prompt(job_id, prompt_used.strip() or None)
    if not ok:
        raise HTTPException(409, reason)
    JOBS.log_activity(job_id, "Промпт подтверждён")
    return JSONResponse({"ok": True, "queue": JOBS.queue_snapshot()})


@app.post("/api/gpu/reset")
def api_gpu_reset() -> JSONResponse:
    """Emergency: kill GPU subprocess and free VRAM."""
    from image_editor import release_gpu_vram

    JOBS.kill_gpu_process()
    JOBS.punch_gpu_workers()
    release_gpu_vram()
    busy = JOBS.is_gpu_busy()
    return JSONResponse(
        {
            "ok": True,
            "gpu_busy": busy,
            "hint": "Перезапустите prod.bat, если GPU всё ещё занят",
        }
    )


@app.post("/api/job/{job_id}/cancel")
def api_cancel_job(job_id: str) -> JSONResponse:
    if not JOBS.get(job_id):
        raise HTTPException(404, "Задача не найдена")
    abort_generation(job_id)
    ok, reason = JOBS.request_cancel(job_id)
    if not ok:
        raise HTTPException(409, reason)
    log_job(job_id, "CANCEL via API")
    # If worker is stuck in CUDA, punch again after marking cancelled
    JOBS.punch_gpu_workers()
    advance_queue()
    return JSONResponse({"ok": True, "queue": JOBS.queue_snapshot()})


@app.get("/api/job/{job_id}/thumb")
def api_job_thumb(job_id: str) -> Response:
    from PIL import Image

    job = JOBS.get(job_id)
    if not job or not job.image_bytes:
        raise HTTPException(404, "Миниатюра недоступна")
    img = Image.open(io.BytesIO(job.image_bytes)).convert("RGB")
    img.thumbnail((96, 96), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


@app.get("/api/job/{job_id}/reference-thumb")
def api_job_reference_thumb(job_id: str, index: int = 0) -> Response:
    from PIL import Image

    job = JOBS.get(job_id)
    ref = _job_reference(job, index)
    if not ref:
        raise HTTPException(404, "Миниатюра примера недоступна")
    img = Image.open(io.BytesIO(ref.data)).convert("RGB")
    img.thumbnail((96, 96), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


def _image_media_type(name: str, data: bytes) -> str:
    """Best-effort MIME from filename, then magic bytes."""
    lower = (name or "").lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith(".gif"):
        return "image/gif"
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            fmt = (img.format or "").upper()
        if fmt == "PNG":
            return "image/png"
        if fmt == "WEBP":
            return "image/webp"
        if fmt == "GIF":
            return "image/gif"
    except Exception:
        pass
    return "image/jpeg"


@app.get("/api/job/{job_id}/reference-original")
def api_job_reference_original(job_id: str, index: int = 0) -> Response:
    job = JOBS.get(job_id)
    ref = _job_reference(job, index)
    if not ref:
        raise HTTPException(404, "Фото-пример недоступен")
    media = _image_media_type(ref.name, ref.data)
    return Response(content=ref.data, media_type=media)


@app.get("/api/job/{job_id}/result")
def api_job_result(job_id: str) -> Response:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Задача не найдена")
    if job.status == "error":
        raise HTTPException(500, job.error or "Ошибка")
    if job.status != "done" or not job.result:
        raise HTTPException(409, "Результат ещё не готов")
    return Response(content=job.result, media_type="image/png")


@app.get("/api/job/{job_id}/original")
def api_job_original(job_id: str) -> Response:
    job = JOBS.get(job_id)
    if not job or not job.image_bytes:
        raise HTTPException(404, "Оригинал недоступен")
    media = _image_media_type(job.image_name, job.image_bytes)
    return Response(content=job.image_bytes, media_type=media)


def main() -> None:
    _install_quiet_thread_interrupts()
    STATIC.mkdir(parents=True, exist_ok=True)
    get_logger()
    log_line(f"=== Photo Edit :{SERVER_PORT} ===")
    print(f"http://127.0.0.1:{SERVER_PORT}")
    try:
        uvicorn.run(app, host="127.0.0.1", port=SERVER_PORT, log_level="info")
    except KeyboardInterrupt:
        print("\nОстановлено.")
    finally:
        _release_gpu_if_needed()


if __name__ == "__main__":
    main()
