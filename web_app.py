#!/usr/bin/env python3
"""Photone web server: photo queue, T2I, music."""

from __future__ import annotations

import io
import os
import sys
import threading
from pathlib import Path
from urllib.parse import quote

import portable_env  # noqa: F401 — sets HF cache under project root

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import uvicorn
from brand import APP_NAME
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from gpu_supervisor import run_gpu_subprocess
from image_editor import (
    MODELS,
    QWEN_EDIT_IDS,
    abort_generation,
    build_joycaption_prompt,
    edit_image,
    get_gpu_info,
    inference_steps_for_model,
    is_test_model,
    list_loras,
    list_models,
    list_t2i_models,
    release_gpu_vram,
    resolve_gpu_chain_model,
    resolve_lora_path,
    uses_joycaption_flow,
)
from job_errors import JobCancelled, JobPaused
from job_logger import get_logger, log_job, log_job_exception, log_generation_error, log_line, open_log_in_system_editor
from job_store import JOBS, MAX_REFERENCE_IMAGES, TERMINAL_STATUSES
from output_resolution import image_dimensions_label
from photo_sort import (
    SORT,
    begin_scan,
    request_cancel,
    request_pause,
    run_cluster,
    start_caption,
    start_copy,
)
from music import (
    MUSIC,
    create_library_folder,
    library_file_bytes,
    list_library,
    move_library_item,
    rename_library_file,
    delete_library_file,
    request_music_cancel,
    result_bytes as music_result_bytes,
    start_music,
    stats_boards,
)
from video import (
    VIDEO,
    catalog as video_catalog,
    request_video_cancel,
    result_file as video_result_file,
    start_video,
)
from speech import (
    SPEECH_ROOT,
    catalog as speech_catalog,
    create_speech_folder,
    delete_speech_file,
    list_speech_library,
    move_speech_item,
    rename_speech_file,
    request_speech_cancel,
    resolve_speech_library_path,
    result_file as speech_result_file,
    save_speech_recording,
    speech_file_bytes,
    start_speech,
    SPEECH,
)
from t2i import T2I, T2I_ASPECTS, request_t2i_cancel, start_t2i
from pipeline_progress import PHASE_TOTAL, PipelineProgress
from prompt_helper import format_for_qwen, ollama_available, prepare_music_fields, prompt_is_english

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"

SERVER_PORT = int(os.environ.get("PHOTO_EDIT_PORT", "7860"))

app = FastAPI(title=APP_NAME)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_queue_runner_lock = threading.Lock()


class QueueReorderBody(BaseModel):
    order: list[str]


class SortScanBody(BaseModel):
    folder: str
    recursive: bool = False


class SortApplyBody(BaseModel):
    dest: str | None = None


class T2IStartBody(BaseModel):
    prompt: str
    aspect: str = "1:1"
    steps: int = 50
    model_id: str = "qwen-image-2512"


class MusicStartBody(BaseModel):
    style: str
    lyrics: str = ""
    cot: str = "off"
    seed: int = 42
    model_id: str = "yue2-3b"
    duration_sec: float | None = None
    source_path: str = ""
    edit: str = ""
    lang: str = "en"


class MusicPrepareBody(BaseModel):
    style: str
    lyrics: str = ""
    model_id: str = "yue2-3b"
    edit: str = ""
    lang: str = "en"


class MusicLibraryMkdirBody(BaseModel):
    path: str = ""
    name: str


class MusicLibraryMoveBody(BaseModel):
    src: str
    dest: str = ""
    cwd: str = ""


class MusicLibraryRenameBody(BaseModel):
    src: str
    name: str
    cwd: str = ""


class MusicLibraryDeleteBody(BaseModel):
    src: str
    cwd: str = ""


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
            prompt_used = format_for_qwen(prompt)
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
            prompt_used = format_for_qwen(joy_raw)
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
        effective_id = MODELS["joycaption"].chain_to or "qwen36-hauhau"
    effective_name = MODELS[effective_id].name if effective_id in MODELS else effective_id

    if refs and resolve_gpu_chain_model(model_id) in QWEN_EDIT_IDS:
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
                approved_prompt=english_prompt if uses_joycaption_flow(model_id) else None,
                job_id=job_id,
                resume_job_id=job_id if _has_resume_checkpoint(job_id) else None,
                reference_images_bytes=refs or None,
                lora_path=resolve_lora_path(job.lora_id) if job and job.lora_id else None,
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
                approved_prompt=english_prompt if uses_joycaption_flow(model_id) else None,
                pipeline=pipeline,
                cancel_checker=cancel_checker,
                pause_checker=lambda: JOBS.should_pause(job_id),
                on_process=JOBS.set_gpu_process,
                resume=_has_resume_checkpoint(job_id),
                inference_steps=inference_steps,
                reference_raw=refs or None,
                lora_id=(job.lora_id if job else "") or "",
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
    ico = STATIC / "favicon.ico"
    if ico.exists():
        return FileResponse(ico, media_type="image/x-icon")
    return FileResponse(STATIC / "favicon.svg", media_type="image/svg+xml")


@app.get("/offline", response_class=HTMLResponse)
def offline_page() -> HTMLResponse:
    return HTMLResponse(
        (STATIC / "offline.html").read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/sw.js", include_in_schema=False)
def service_worker() -> Response:
    return Response(
        content=(STATIC / "sw.js").read_text(encoding="utf-8"),
        media_type="application/javascript; charset=utf-8",
        headers={"Cache-Control": "no-cache", "Service-Worker-Allowed": "/"},
    )


def _html_page(name: str) -> HTMLResponse:
    return HTMLResponse(
        (STATIC / name).read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"},
    )


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return _html_page("app.html")


@app.get("/sort", response_class=HTMLResponse)
def sort_page() -> HTMLResponse:
    return _html_page("sort.html")


@app.get("/generate", response_class=HTMLResponse)
def generate_page() -> HTMLResponse:
    return _html_page("generate.html")


@app.get("/music", response_class=HTMLResponse)
def music_page() -> HTMLResponse:
    return _html_page("music.html")


@app.get("/speech", response_class=HTMLResponse)
def speech_page() -> HTMLResponse:
    return _html_page("speech.html")


@app.get("/video", response_class=HTMLResponse)
def video_page() -> HTMLResponse:
    return _html_page("video.html")


@app.get("/api/models")
def api_models() -> dict:
    return {
        "models": list_models(),
        "ollama": ollama_available(),
        "port": SERVER_PORT,
        "gpu": get_gpu_info(),
        "max_reference_images": MAX_REFERENCE_IMAGES,
        "loras": list_loras(),
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
    snap = SORT.snapshot()
    t2i = T2I.snapshot()
    music = MUSIC.snapshot()
    payload: dict = {
        "gpu_busy": JOBS.is_gpu_busy(),
        "has_work": queue["has_work"],
        "pending_count": queue["pending_count"],
        "port": SERVER_PORT,
        "gpu": get_gpu_info(),
        "sort_phase": snap["phase"],
        "sort_message": snap["message"],
        "sort_busy": snap["busy"],
        "sort_elapsed_sec": snap["elapsed_sec"],
        "sort_work_title": snap["work_title"],
        "t2i_busy": t2i["busy"],
        "t2i_message": t2i["message"],
        "music_busy": music["busy"],
        "music_message": music["message"],
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


@app.post("/api/queue/skip-prep")
async def api_queue_skip_prep() -> JSONResponse:
    count, reason, skipped_ids = JOBS.skip_prep_drafts()
    if count == 0:
        raise HTTPException(400, reason)
    for job_id in skipped_ids:
        JOBS.log_activity(job_id, "Без подготовки JoyCaption — промпт как есть")
    return JSONResponse({"ok": True, "skipped": count, "queue": JOBS.queue_snapshot()})


@app.post("/api/queue/start-gpu")
async def api_queue_start_gpu(
    model: str = Form(""),
    steps: str = Form(""),
    lora: str = Form(""),
) -> JSONResponse:
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
        lora_id = ""
        if lora.strip():
            resolve_lora_path(lora.strip())
            lora_id = Path(lora.strip()).name
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    count, promoted_ids = JOBS.promote_approved_to_gpu(model, inference_steps, lora_id=lora_id)
    if count == 0:
        raise HTTPException(400, "Нет подтверждённых промптов")
    for job_id in promoted_ids:
        JOBS.log_activity(job_id, f"Модель: «{model}», шагов: {inference_steps}" + (f", LoRA {lora_id}" if lora_id else ""))
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


def _sort_status_payload() -> dict:
    snap = SORT.snapshot()
    snap["gpu_busy"] = JOBS.is_gpu_busy()
    snap["ollama"] = ollama_available()
    return snap


@app.get("/api/sort/status")
def api_sort_status() -> dict:
    return _sort_status_payload()


@app.post("/api/sort/scan")
def api_sort_scan(body: SortScanBody) -> JSONResponse:
    try:
        begin_scan(body.folder, body.recursive)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return JSONResponse(_sort_status_payload())


@app.post("/api/sort/caption/start")
def api_sort_caption_start() -> JSONResponse:
    def acquire() -> None:
        if not JOBS.gpu_lock().acquire(blocking=False):
            raise RuntimeError("GPU занят — дождитесь генерации")

    def release() -> None:
        try:
            JOBS.gpu_lock().release()
        except RuntimeError:
            pass

    try:
        start_caption(JOBS.is_gpu_busy(), acquire, release)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return JSONResponse(_sort_status_payload())


@app.post("/api/sort/caption/pause")
def api_sort_caption_pause() -> JSONResponse:
    return JSONResponse(request_pause() | {"gpu_busy": JOBS.is_gpu_busy(), "ollama": ollama_available()})


@app.post("/api/sort/caption/stop")
def api_sort_caption_stop() -> JSONResponse:
    return JSONResponse(request_cancel() | {"gpu_busy": JOBS.is_gpu_busy(), "ollama": ollama_available()})


@app.post("/api/sort/cluster")
def api_sort_cluster() -> JSONResponse:
    try:
        run_cluster()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return JSONResponse(_sort_status_payload())


@app.post("/api/sort/apply")
def api_sort_apply(body: SortApplyBody) -> JSONResponse:
    try:
        start_copy(body.dest)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return JSONResponse(_sort_status_payload())


def _t2i_payload() -> dict:
    snap = T2I.snapshot()
    snap["gpu_busy"] = JOBS.is_gpu_busy()
    snap["ollama"] = ollama_available()
    return snap


@app.get("/api/t2i/models")
def api_t2i_models() -> dict:
    return {
        "models": list_t2i_models(),
        "aspects": {k: {"width": w, "height": h} for k, (w, h) in T2I_ASPECTS.items()},
        "ollama": ollama_available(),
        "gpu": get_gpu_info(),
    }


@app.get("/api/t2i/status")
def api_t2i_status() -> dict:
    return _t2i_payload()


@app.post("/api/t2i/start")
def api_t2i_start(body: T2IStartBody) -> JSONResponse:
    def acquire() -> None:
        if not JOBS.gpu_lock().acquire(blocking=False):
            raise RuntimeError("GPU занят — дождитесь правки или сортировки")

    def release() -> None:
        try:
            JOBS.gpu_lock().release()
        except RuntimeError:
            pass

    try:
        start_t2i(
            body.prompt,
            aspect=body.aspect,
            steps=body.steps,
            model_id=body.model_id,
            gpu_busy=JOBS.is_gpu_busy(),
            acquire_gpu=acquire,
            release_gpu=release,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return JSONResponse(_t2i_payload())


@app.post("/api/t2i/stop")
def api_t2i_stop() -> JSONResponse:
    return JSONResponse(request_t2i_cancel() | {"gpu_busy": JOBS.is_gpu_busy(), "ollama": ollama_available()})


@app.get("/api/t2i/result")
def api_t2i_result() -> Response:
    with T2I._lock:
        data = T2I.result
    if not data:
        raise HTTPException(404, "Результат ещё не готов")
    return Response(content=data, media_type="image/png")


def _music_payload() -> dict:
    snap = MUSIC.snapshot()
    snap["gpu_busy"] = JOBS.is_gpu_busy()
    return snap


@app.get("/api/music/status")
def api_music_status() -> dict:
    return _music_payload()


@app.get("/api/music/stats")
def api_music_stats() -> dict:
    return stats_boards()


@app.post("/api/music/prepare")
def api_music_prepare(body: MusicPrepareBody) -> JSONResponse:
    """RU → EN via Ollama. English fields pass through without calling Ollama."""
    try:
        payload = prepare_music_fields(
            body.style,
            body.lyrics,
            sfx=body.model_id == "moss-sfx-v2",
            edit=body.edit,
            lang=body.lang,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse(payload)


@app.post("/api/music/start")
def api_music_start(body: MusicStartBody) -> JSONResponse:
    def acquire() -> None:
        if not JOBS.gpu_lock().acquire(blocking=False):
            raise RuntimeError("GPU занят — дождитесь правки, сортировки или картинки")

    def release() -> None:
        try:
            JOBS.gpu_lock().release()
        except RuntimeError:
            pass

    style, lyrics = body.style, body.lyrics
    try:
        source_path = (body.source_path or "").strip()
        edit = body.edit
        model_id = "acestep-1.5" if source_path else body.model_id
        prep = prepare_music_fields(
            style,
            lyrics,
            sfx=model_id == "moss-sfx-v2",
            edit=edit,
            lang="en" if model_id == "moss-sfx-v2" else body.lang,
        )
        style, lyrics, edit = prep["style"], prep["lyrics"], prep.get("edit") or ""
        start_music(
            style=style,
            lyrics=lyrics,
            cot=body.cot,
            seed=body.seed,
            model_id=model_id,
            duration_sec=body.duration_sec,
            source_rel=source_path or None,
            edit=edit,
            lang=prep.get("lang") or body.lang,
            gpu_busy=JOBS.is_gpu_busy(),
            acquire_gpu=acquire,
            release_gpu=release,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return JSONResponse(_music_payload())


@app.post("/api/music/stop")
def api_music_stop() -> JSONResponse:
    return JSONResponse(request_music_cancel() | {"gpu_busy": JOBS.is_gpu_busy()})


@app.get("/api/music/result")
def api_music_result() -> Response:
    data = music_result_bytes()
    if not data:
        raise HTTPException(404, "Результат ещё не готов")
    return Response(content=data, media_type="audio/flac")


@app.get("/api/music/library")
def api_music_library(path: str = "") -> JSONResponse:
    try:
        return JSONResponse(list_library(path))
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/music/library/mkdir")
def api_music_library_mkdir(body: MusicLibraryMkdirBody) -> JSONResponse:
    try:
        return JSONResponse(create_library_folder(body.path, body.name))
    except FileExistsError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/music/library/move")
def api_music_library_move(body: MusicLibraryMoveBody) -> JSONResponse:
    try:
        return JSONResponse(move_library_item(body.src, body.dest, body.cwd))
    except FileExistsError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/music/library/rename")
def api_music_library_rename(body: MusicLibraryRenameBody) -> JSONResponse:
    try:
        return JSONResponse(rename_library_file(body.src, body.name, body.cwd))
    except FileExistsError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/music/library/delete")
def api_music_library_delete(body: MusicLibraryDeleteBody) -> JSONResponse:
    try:
        return JSONResponse(delete_library_file(body.src, body.cwd))
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


def _inline_audio_disposition(name: str) -> str:
    encoded = quote(name)
    if encoded == name:
        return f'inline; filename="{name}"'
    return f"inline; filename*=UTF-8''{encoded}"


@app.get("/api/video/catalog")
def api_video_catalog() -> JSONResponse:
    return JSONResponse(video_catalog())


@app.get("/api/video/status")
def api_video_status() -> JSONResponse:
    return JSONResponse(VIDEO.snapshot() | {"gpu_busy": JOBS.is_gpu_busy()})


@app.post("/api/video/start")
async def api_video_start(
    model_id: str = Form(...),
    prompt: str = Form(...),
    seconds: float = Form(3),
    quality: str = Form("480"),
    seed: int = Form(42),
    image: UploadFile | None = File(None),
) -> JSONResponse:
    image_path = None
    if image is not None and image.filename:
        suffix = Path(image.filename).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            raise HTTPException(400, "Фото: png, jpg или webp")
        folder = ROOT / "Video" / "_stills"
        folder.mkdir(parents=True, exist_ok=True)
        image_path = folder / f"still{suffix}"
        image_path.write_bytes(await image.read())

    def acquire() -> None:
        if not JOBS.gpu_lock().acquire(blocking=False):
            raise RuntimeError("GPU занят — дождитесь другой генерации")

    def release() -> None:
        try:
            JOBS.gpu_lock().release()
        except RuntimeError:
            pass

    try:
        payload = start_video(
            model_id=model_id,
            prompt=prompt,
            image=image_path,
            seconds=seconds,
            seed=seed,
            quality=quality,
            gpu_busy=JOBS.is_gpu_busy(),
            acquire_gpu=acquire,
            release_gpu=release,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return JSONResponse(payload | {"gpu_busy": JOBS.is_gpu_busy()})


@app.post("/api/video/stop")
def api_video_stop() -> JSONResponse:
    return JSONResponse(request_video_cancel() | {"gpu_busy": JOBS.is_gpu_busy()})


@app.get("/api/video/result")
def api_video_result() -> FileResponse:
    file = video_result_file()
    if file is None:
        raise HTTPException(404, "Видео ещё не готово")
    return FileResponse(path=str(file), media_type="video/mp4", headers={"Cache-Control": "no-store"})


@app.get("/api/speech/catalog")
def api_speech_catalog() -> JSONResponse:
    return JSONResponse(speech_catalog())


@app.get("/api/speech/status")
def api_speech_status() -> JSONResponse:
    return JSONResponse(SPEECH.snapshot() | {"gpu_busy": JOBS.is_gpu_busy()})


@app.post("/api/speech/start")
async def api_speech_start(
    engine: str = Form(...),
    lang: str = Form("ru"),
    voice: str = Form(""),
    text: str = Form(...),
    reference: UploadFile | None = File(None),
    library_ref: str = Form(""),
) -> JSONResponse:
    ref_path = None
    if reference is not None and reference.filename:
        suffix = Path(reference.filename).suffix.lower()
        if suffix not in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}:
            raise HTTPException(400, "Референс: wav, mp3, flac, ogg или m4a")
        folder = SPEECH_ROOT / "_refs"
        folder.mkdir(parents=True, exist_ok=True)
        ref_path = folder / f"ref{suffix}"
        ref_path.write_bytes(await reference.read())
    elif library_ref.strip():
        try:
            ref_path = resolve_speech_library_path(library_ref)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not ref_path.is_file() or ref_path.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}:
            raise HTTPException(400, "Референс из библиотеки не найден")

    def acquire() -> None:
        if not JOBS.gpu_lock().acquire(blocking=False):
            raise RuntimeError("GPU занят — дождитесь правки, сортировки, картинки или музыки")

    def release() -> None:
        try:
            JOBS.gpu_lock().release()
        except RuntimeError:
            pass

    try:
        payload = start_speech(
            engine=engine,
            lang=lang,
            voice=voice,
            text=text,
            reference=ref_path,
            gpu_busy=JOBS.is_gpu_busy(),
            acquire_gpu=acquire,
            release_gpu=release,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return JSONResponse(payload | {"gpu_busy": JOBS.is_gpu_busy()})


@app.post("/api/speech/stop")
def api_speech_stop() -> JSONResponse:
    return JSONResponse(request_speech_cancel() | {"gpu_busy": JOBS.is_gpu_busy()})


@app.get("/api/speech/result")
def api_speech_result() -> FileResponse:
    file = speech_result_file()
    if file is None:
        raise HTTPException(404, "Речь ещё не готова")
    return FileResponse(
        path=str(file),
        media_type="audio/wav",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/speech/library")
def api_speech_library(path: str = "") -> JSONResponse:
    try:
        return JSONResponse(list_speech_library(path))
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/speech/library/mkdir")
def api_speech_library_mkdir(body: MusicLibraryMkdirBody) -> JSONResponse:
    try:
        return JSONResponse(create_speech_folder(body.path, body.name))
    except FileExistsError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/speech/library/move")
def api_speech_library_move(body: MusicLibraryMoveBody) -> JSONResponse:
    try:
        return JSONResponse(move_speech_item(body.src, body.dest, body.cwd))
    except FileExistsError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/speech/library/rename")
def api_speech_library_rename(body: MusicLibraryRenameBody) -> JSONResponse:
    try:
        return JSONResponse(rename_speech_file(body.src, body.name, body.cwd))
    except FileExistsError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/speech/library/delete")
def api_speech_library_delete(body: MusicLibraryDeleteBody) -> JSONResponse:
    try:
        return JSONResponse(delete_speech_file(body.src, body.cwd))
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/speech/library/record")
async def api_speech_library_record(file: UploadFile = File(...)) -> JSONResponse:
    raw = await file.read()
    try:
        return JSONResponse(save_speech_recording(raw))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/speech/file")
def api_speech_file(path: str) -> FileResponse:
    try:
        file, mime = speech_file_bytes(path)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(404, str(exc)) from exc
    return FileResponse(
        path=str(file),
        media_type=mime,
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": _inline_audio_disposition(file.name),
        },
    )


@app.get("/api/music/file")
def api_music_file(path: str) -> FileResponse:
    try:
        file, mime = library_file_bytes(path)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(404, str(exc)) from exc
    return FileResponse(
        path=str(file),
        media_type=mime,
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": _inline_audio_disposition(file.name),
        },
    )


def run_server() -> None:
    uvicorn.run(app, host="127.0.0.1", port=SERVER_PORT, log_level="info")


def main() -> None:
    _install_quiet_thread_interrupts()
    STATIC.mkdir(parents=True, exist_ok=True)
    get_logger()
    log_line(f"=== {APP_NAME} :{SERVER_PORT} ===")
    print(f"{APP_NAME}  http://127.0.0.1:{SERVER_PORT}")
    want_tray = sys.platform == "win32" and os.environ.get("PHOTO_EDIT_NO_TRAY") != "1"
    if want_tray:
        from win_tray import (
            PANEL_URL,
            acquire_single_instance,
            kill_self_tree,
            release_mutex,
            set_app_id,
        )

        set_app_id()
        mutex = acquire_single_instance("Local\\LocalPhotoAiPanel")
        if mutex is None:
            import webbrowser

            webbrowser.open(PANEL_URL)
            return

        def worker() -> None:
            try:
                run_server()
            except Exception:
                get_logger().exception("server")

        def on_quit() -> None:
            kill_self_tree()

        threading.Thread(target=worker, name="uvicorn", daemon=True).start()
        try:
            from tray_runtime import run_tray_blocking

            run_tray_blocking(on_quit=on_quit)
        finally:
            release_mutex(mutex)
            os._exit(0)
        return
    try:
        run_server()
    except KeyboardInterrupt:
        print("\nОстановлено.")
    finally:
        _release_gpu_if_needed()


if __name__ == "__main__":
    main()
