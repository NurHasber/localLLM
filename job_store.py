"""In-memory job queue for async image editing with progress."""

from __future__ import annotations

import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

TERMINAL_STATUSES = frozenset({"done", "error", "cancelled"})
PAUSED_STATUS = "paused"
DRAFT_STATUS = "draft"
DRAFT_STATUSES = frozenset({DRAFT_STATUS})
PREP_WAIT_STATUSES = frozenset({"queued", "translate", "approve"})
APPROVED_STATUSES = frozenset({"approved"})
GPU_WAIT_STATUSES = frozenset({"queued_gpu"})
REORDERABLE_STATUSES = frozenset({"queued", "approve", "approved", "queued_gpu", PAUSED_STATUS})
PENDING_DELETABLE_STATUSES = frozenset({"draft", "queued", "approve", "approved", "queued_gpu", PAUSED_STATUS})
MAX_DRAFT_ITEMS = 10
FINISHED_PAGE_SIZE = 5
MAX_REFERENCE_IMAGES = 3


def _image_size_label(image_bytes: bytes | None) -> str | None:
    if not image_bytes:
        return None
    try:
        from output_resolution import image_dimensions_label

        return image_dimensions_label(image_bytes)
    except Exception:
        return None


@dataclass
class ReferencePhoto:
    name: str
    data: bytes


@dataclass
class Job:
    id: str
    status: str = "queued"
    progress: int = 0
    phase: int = 0
    phase_total: int = 6
    step: int = 0
    total_steps: int = 0
    message: str = "В очереди..."
    model_id: str = ""
    inference_steps: int = 0
    lora_id: str = ""
    prompt: str = ""
    output_max_side: int = 0
    image_name: str = ""
    image_bytes: bytes | None = None
    reference_images: list[ReferencePhoto] = field(default_factory=list)
    prompt_used: str | None = None
    prompt_translation: str | None = None
    note: str | None = None
    result: bytes | None = None
    preview: bytes | None = None
    preview_step: int = 0
    preview_total_steps: int = 0
    error: str | None = None
    cancel_requested: bool = False
    was_cancelled: bool = False
    pause_requested: bool = False
    paused_at_step: int = 0
    generation_started_at: float | None = None
    generation_finished_at: float | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    activities: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        from image_editor import uses_joycaption_flow

        return {
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "phase": self.phase,
            "phase_total": self.phase_total,
            "step": self.step,
            "total_steps": self.total_steps,
            "message": self.message,
            "model_id": self.model_id,
            "inference_steps": self.inference_steps,
            "lora_id": self.lora_id,
            "joycaption_flow": uses_joycaption_flow(self.model_id),
            "prompt": self.prompt,
            "output_max_side": self.output_max_side,
            "image_size": _image_size_label(self.image_bytes),
            "image_name": self.image_name,
            "has_thumb": self.image_bytes is not None,
            "has_original": self.image_bytes is not None,
            "reference_image_name": self.reference_images[0].name if self.reference_images else "",
            "has_reference_thumb": bool(self.reference_images),
            "reference_count": len(self.reference_images),
            "references": [
                {"index": i, "name": ref.name} for i, ref in enumerate(self.reference_images)
            ],
            "prompt_used": self.prompt_used,
            "prompt_translation": self.prompt_translation,
            "note": self.note,
            "error": self.error,
            "has_preview": self.preview is not None,
            "preview_step": self.preview_step,
            "preview_total_steps": self.preview_total_steps,
            "activities": self.activities,
            "done": self.status == "done",
            "failed": self.status == "error",
            "cancelled": self.status == "cancelled",
            "awaiting_approval": self.status == "approve",
            "can_approve": self.status == "approve",
            "approved": self.status == "approved",
            "cancellable": self.status in ("generate", "translate"),
            "cancel_requested": self.cancel_requested,
            "was_cancelled": self.was_cancelled,
            "queued": self.status == "queued",
            "queued_gpu": self.status == "queued_gpu",
            "paused": self.status == PAUSED_STATUS,
            "can_resume": self.status == PAUSED_STATUS,
            "paused_at_step": self.paused_at_step,
            "pausable": self.status == "generate",
            "preparing": self.status == "translate",
            "is_draft": self.status == DRAFT_STATUS,
            "draft_ready": bool(
                self.status == DRAFT_STATUS
                and self.image_bytes
                and self.prompt.strip()
            ),
            "has_result": self.status == "done" and self.result is not None,
            "can_requeue": self.status == "done" and self.result is not None,
            "generation_started_at": self.generation_started_at,
            "generation_finished_at": self.generation_finished_at,
            "updated_at": self.updated_at,
        }

    def reference_bytes_list(self) -> list[bytes]:
        return [ref.data for ref in self.reference_images]


MAX_ACTIVITY_LOG = 400
APPROVAL_TIMEOUT_SEC = 3600


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._gpu_lock = threading.Lock()
        self._active_job_id: str | None = None
        self._prep_active_id: str | None = None
        self._pending_ids: list[str] = []
        self._approval_events: dict[str, threading.Event] = {}
        self._approval_outcome: dict[str, tuple[bool, str | None]] = {}
        self._interrupt_hooks: dict[str, Callable[[], None]] = {}
        self._gpu_worker_tid: int | None = None
        self._prep_worker_tid: int | None = None
        self._gpu_process: subprocess.Popen | None = None
        self._gpu_session_model: str | None = None
        self._gpu_session_steps: int = 0
        self._gpu_session_lora: str = ""

    def gpu_session_model(self) -> str | None:
        with self._lock:
            return self._gpu_session_model

    def gpu_session_steps(self) -> int:
        with self._lock:
            return self._gpu_session_steps

    def gpu_session_active_unlocked(self) -> bool:
        if self._active_job_id:
            job = self._jobs.get(self._active_job_id)
            if job and job.status == "generate":
                return True
        if self._gpu_process is not None and self._gpu_process.poll() is None:
            return True
        for job_id in self._pending_ids:
            job = self._jobs.get(job_id)
            if job and job.status in GPU_WAIT_STATUSES:
                return True
        return self._gpu_session_model is not None and any(
            (j := self._jobs.get(jid)) and j.status in GPU_WAIT_STATUSES
            for jid in self._pending_ids
        )

    def has_paused_gpu_jobs_unlocked(self) -> bool:
        for job_id in self._pending_ids:
            job = self._jobs.get(job_id)
            if job and job.status == PAUSED_STATUS:
                return True
        return False

    def has_paused_gpu_jobs(self) -> bool:
        with self._lock:
            return self.has_paused_gpu_jobs_unlocked()

    def gpu_session_active(self) -> bool:
        with self._lock:
            return self.gpu_session_active_unlocked()

    def _sync_gpu_session_unlocked(self) -> None:
        has_gpu_work = self._active_job_id is not None
        if not has_gpu_work and self._gpu_process is not None and self._gpu_process.poll() is None:
            has_gpu_work = True
        if not has_gpu_work:
            for jid in self._pending_ids:
                job = self._jobs.get(jid)
                if job and job.status in GPU_WAIT_STATUSES:
                    has_gpu_work = True
                    break
        if not has_gpu_work or self.has_paused_gpu_jobs_unlocked():
            self._gpu_session_model = None
            self._gpu_session_steps = 0
            self._gpu_session_lora = ""

    def set_gpu_process(self, proc) -> None:
        with self._lock:
            self._gpu_process = proc

    def get_gpu_process(self):
        with self._lock:
            return self._gpu_process

    def kill_gpu_process(self) -> None:
        from gpu_supervisor import kill_process

        with self._lock:
            proc = self._gpu_process
        kill_process(proc)
        with self._lock:
            if self._gpu_process is proc:
                self._gpu_process = None

    def set_gpu_worker_tid(self, tid: int | None) -> None:
        with self._lock:
            self._gpu_worker_tid = tid

    def set_prep_worker_tid(self, tid: int | None) -> None:
        with self._lock:
            self._prep_worker_tid = tid

    def punch_gpu_workers(self) -> None:
        from cancel_utils import inject_thread_exception
        from job_errors import JobCancelled

        with self._lock:
            tids = {t for t in (self._gpu_worker_tid, self._prep_worker_tid) if t}
        for tid in tids:
            inject_thread_exception(tid, JobCancelled)

    def active_job_id(self) -> str | None:
        with self._lock:
            return self._active_job_id

    def set_active_job(self, job_id: str | None) -> None:
        with self._lock:
            self._active_job_id = job_id

    def prep_active_id(self) -> str | None:
        with self._lock:
            return self._prep_active_id

    def clear_prep_active(self, job_id: str) -> None:
        with self._lock:
            if self._prep_active_id == job_id:
                self._prep_active_id = None

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending_ids)

    def has_work(self) -> bool:
        with self._lock:
            if self._active_job_id:
                job = self._jobs.get(self._active_job_id)
                if job and job.status not in TERMINAL_STATUSES:
                    return True
            if self._prep_active_id:
                job = self._jobs.get(self._prep_active_id)
                if job and job.status not in TERMINAL_STATUSES:
                    return True
            for job_id in self._pending_ids:
                job = self._jobs.get(job_id)
                if job and job.status not in TERMINAL_STATUSES:
                    return True
            return False

    def is_gpu_busy(self) -> bool:
        with self._lock:
            if self._gpu_process is not None and self._gpu_process.poll() is None:
                return True
        return self._gpu_lock.locked()

    def register_interrupt(self, job_id: str, hook: Callable[[], None]) -> None:
        with self._lock:
            self._interrupt_hooks[job_id] = hook

    def clear_interrupt(self, job_id: str) -> None:
        with self._lock:
            self._interrupt_hooks.pop(job_id, None)

    def trigger_interrupt(self, job_id: str) -> None:
        with self._lock:
            hook = self._interrupt_hooks.get(job_id)
        if hook:
            try:
                hook()
            except Exception:
                pass

    def create(self, model_id: str, *, status: str = "queued") -> Job:
        job = Job(id=uuid.uuid4().hex, model_id=model_id, status=status)
        if status == DRAFT_STATUS:
            job.message = "Черновик — укажите промпт и загрузите фото"
        with self._lock:
            self._jobs[job.id] = job
            self._approval_events[job.id] = threading.Event()
            self._approval_outcome.pop(job.id, None)
        return job

    def create_draft(self) -> tuple[Job | None, str]:
        with self._lock:
            draft_count = sum(
                1
                for jid in self._pending_ids
                if (j := self._jobs.get(jid)) and j.status == DRAFT_STATUS
            )
            if draft_count >= MAX_DRAFT_ITEMS:
                return None, f"Максимум {MAX_DRAFT_ITEMS} элементов в очереди"
        job = self.create("", status=DRAFT_STATUS)
        with self._lock:
            if job.id not in self._pending_ids:
                self._pending_ids.append(job.id)
            job.updated_at = time.time()
        return job, ""

    def create_draft_from_result(self, source_job_id: str) -> tuple[Job | None, str]:
        with self._lock:
            source = self._jobs.get(source_job_id)
            if not source:
                return None, "Задача не найдена"
            if source.status != "done" or not source.result:
                return None, "Нет готового результата"
            draft_count = sum(
                1
                for jid in self._pending_ids
                if (j := self._jobs.get(jid)) and j.status == DRAFT_STATUS
            )
            if draft_count >= MAX_DRAFT_ITEMS:
                return None, f"Максимум {MAX_DRAFT_ITEMS} элементов в очереди"
            result_bytes = source.result
            base_name = source.image_name or "photo"
            copied_refs = [ReferencePhoto(name=ref.name, data=ref.data) for ref in source.reference_images]

        job, reason = self.create_draft()
        if not job:
            return None, reason

        if "." in base_name:
            stem = base_name.rsplit(".", 1)[0]
        else:
            stem = base_name
        image_name = f"{stem}-result.png"

        with self._lock:
            job = self._jobs.get(job.id)
            if not job:
                return None, "Не удалось создать черновик"
            job.image_bytes = result_bytes
            job.prompt = ""
            job.image_name = image_name
            job.reference_images = copied_refs
            job.message = "Черновик — укажите новый промпт для результата"
            job.note = f"из результата «{base_name}»"
            try:
                from output_resolution import image_pixel_size

                w, h = image_pixel_size(result_bytes)
                job.output_max_side = max(w, h)
            except Exception:
                job.output_max_side = 0
            job.updated_at = time.time()

        return job, ""

    def update_draft(
        self,
        job_id: str,
        *,
        prompt: str | None = None,
        image_bytes: bytes | None = None,
        image_name: str | None = None,
        append_references: list[tuple[str, bytes]] | None = None,
        clear_reference: bool = False,
        remove_reference_index: int | None = None,
        model_id: str | None = None,
    ) -> tuple[bool, str]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False, "Элемент не найден"
            if job.status != DRAFT_STATUS:
                return False, "Редактировать можно только черновики до подготовки промптов"
            if prompt is not None:
                job.prompt = prompt.strip()
            if image_bytes is not None:
                job.image_bytes = image_bytes
                try:
                    from output_resolution import image_pixel_size

                    w, h = image_pixel_size(image_bytes)
                    job.output_max_side = max(w, h)
                except Exception:
                    job.output_max_side = 0
            if image_name is not None:
                job.image_name = image_name or "photo"
            if clear_reference:
                job.reference_images = []
            if remove_reference_index is not None:
                if 0 <= remove_reference_index < len(job.reference_images):
                    job.reference_images.pop(remove_reference_index)
                else:
                    return False, "Нет такого фото-примера"
            if append_references:
                room = MAX_REFERENCE_IMAGES - len(job.reference_images)
                if room <= 0:
                    return False, f"Максимум {MAX_REFERENCE_IMAGES} фото-примера"
                if len(append_references) > room:
                    return False, f"Можно добавить ещё {room} фото-пример(а), максимум {MAX_REFERENCE_IMAGES}"
                for name, data in append_references:
                    job.reference_images.append(ReferencePhoto(name=name or "reference", data=data))
            if model_id is not None:
                job.model_id = model_id
            if job.image_bytes and job.prompt.strip():
                n_ref = len(job.reference_images)
                if n_ref == 1:
                    ref_note = " + 1 фото-пример"
                elif n_ref > 1:
                    ref_note = f" + {n_ref} фото-примера"
                else:
                    ref_note = ""
                job.message = f"Готов к подготовке промпта{ref_note}"
            else:
                job.message = "Черновик — укажите промпт и загрузите фото"
            job.updated_at = time.time()
            return True, ""

    def draft_ready_count(self) -> int:
        with self._lock:
            return sum(
                1
                for jid in self._pending_ids
                if (j := self._jobs.get(jid))
                and j.status == DRAFT_STATUS
                and j.image_bytes
                and j.prompt.strip()
            )

    def has_drafts(self) -> bool:
        with self._lock:
            return any(
                (j := self._jobs.get(jid)) and j.status == DRAFT_STATUS
                for jid in self._pending_ids
            )

    def prepare_drafts(
        self,
    ) -> tuple[int, str, list[str]]:
        with self._lock:
            to_prepare: list[Job] = []
            for job_id in self._pending_ids:
                job = self._jobs.get(job_id)
                if (
                    job
                    and job.status == DRAFT_STATUS
                    and job.image_bytes
                    and job.prompt.strip()
                ):
                    to_prepare.append(job)
            if not to_prepare:
                return 0, "Нет готовых элементов (нужны фото и промпт)", []
            for job in to_prepare:
                if job.image_bytes:
                    try:
                        from output_resolution import image_pixel_size

                        w, h = image_pixel_size(job.image_bytes)
                        job.output_max_side = max(w, h)
                    except Exception:
                        pass
                job.status = "queued"
                job.message = "В очереди — скоро начнётся подготовка промпта"
                job.updated_at = time.time()
            self._update_queue_positions_unlocked()
            return len(to_prepare), "", [j.id for j in to_prepare]

    def skip_prep_drafts(self) -> tuple[int, str, list[str]]:
        """Mark ready drafts approved using the user's prompt as-is (no JoyCaption)."""
        from image_editor import augment_prompt_for_qwen_reference
        from prompt_helper import format_for_qwen, prompt_is_english

        with self._lock:
            to_skip: list[Job] = []
            for job_id in self._pending_ids:
                job = self._jobs.get(job_id)
                if (
                    job
                    and job.status == DRAFT_STATUS
                    and job.image_bytes
                    and job.prompt.strip()
                ):
                    to_skip.append(job)
            if not to_skip:
                return 0, "Нет готовых черновиков (нужны фото и промпт)", []
            for job in to_skip:
                if job.image_bytes:
                    try:
                        from output_resolution import image_pixel_size

                        w, h = image_pixel_size(job.image_bytes)
                        job.output_max_side = max(w, h)
                    except Exception:
                        pass
                prompt_used = format_for_qwen(job.prompt)
                n_refs = len(job.reference_images)
                if n_refs:
                    prompt_used = augment_prompt_for_qwen_reference(prompt_used, n_refs)
                job.prompt_used = prompt_used
                job.prompt_translation = job.prompt.strip()
                job.note = None if prompt_is_english(job.prompt) else (
                    "Промпт отправлен без JoyCaption — русский текст ушёл в модель как есть"
                )
                job.status = "approved"
                job.message = "Без подготовки — в очередь на генерацию"
                job.updated_at = time.time()
            self._update_queue_positions_unlocked()
            return len(to_skip), "", [j.id for j in to_skip]

    def attach_payload(
        self,
        job_id: str,
        *,
        image_bytes: bytes,
        prompt: str,
        image_name: str = "",
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.image_bytes = image_bytes
            job.prompt = prompt
            job.image_name = image_name or "photo"
            try:
                from output_resolution import image_pixel_size

                w, h = image_pixel_size(image_bytes)
                job.output_max_side = max(w, h)
            except Exception:
                job.output_max_side = 0
            job.updated_at = time.time()

    def enqueue_pending(self, job_id: str) -> int:
        """Add job to tail of queue. Returns 1-based position."""
        with self._lock:
            if job_id not in self._jobs:
                return 0
            if job_id not in self._pending_ids:
                self._pending_ids.append(job_id)
            job = self._jobs[job_id]
            job.status = "queued"
            pos = self._pending_ids.index(job_id) + 1
            ahead = pos - 1
            if ahead:
                job.message = f"В очереди — позиция {pos}"
            else:
                job.message = "В очереди — скоро начнётся"
            job.updated_at = time.time()
            return pos

    def pending_position(self, job_id: str) -> int:
        with self._lock:
            if job_id in self._pending_ids:
                return self._pending_ids.index(job_id) + 1
            return 0

    def claim_next_for_prep(self) -> Job | None:
        with self._lock:
            if self._prep_active_id:
                return None
            for job_id in self._pending_ids:
                job = self._jobs.get(job_id)
                if job and job.status == "queued" and job.image_bytes:
                    self._prep_active_id = job_id
                    return job
            return None

    def claim_next_for_gpu(self) -> Job | None:
        with self._lock:
            if self._active_job_id:
                return None
            if self.prep_still_running_unlocked():
                return None
            if any(
                (j := self._jobs.get(jid)) and j.status == PAUSED_STATUS
                for jid in self._pending_ids
            ):
                return None
            for job_id in self._pending_ids:
                job = self._jobs.get(job_id)
                if (
                    job
                    and job.status == "queued_gpu"
                    and job.image_bytes
                    and job.prompt_used
                ):
                    self._active_job_id = job_id
                    return job
            return None

    def prep_still_running_unlocked(self) -> bool:
        if self._prep_active_id:
            job = self._jobs.get(self._prep_active_id)
            if job and job.status not in TERMINAL_STATUSES:
                return True
        for job_id in self._pending_ids:
            job = self._jobs.get(job_id)
            if job and job.status in ("queued", "translate"):
                return True
        return False

    def finish_job(self, job_id: str) -> None:
        with self._lock:
            if job_id in self._pending_ids:
                self._pending_ids.remove(job_id)
            if self._active_job_id == job_id:
                self._active_job_id = None
            if self._prep_active_id == job_id:
                self._prep_active_id = None
            self._sync_gpu_session_unlocked()

    def _release_slots_unlocked(self, job_id: str) -> None:
        if self._active_job_id == job_id:
            self._active_job_id = None
        if self._prep_active_id == job_id:
            self._prep_active_id = None

    def _update_queue_positions_unlocked(self) -> None:
        pos = 0
        for job_id in self._pending_ids:
            job = self._jobs.get(job_id)
            if not job or job.status in TERMINAL_STATUSES:
                continue
            if job_id == self._active_job_id:
                continue
            if job.status not in PREP_WAIT_STATUSES | GPU_WAIT_STATUSES | APPROVED_STATUSES | DRAFT_STATUSES:
                continue
            if job.status == DRAFT_STATUS:
                continue
            pos += 1
            tag = "Отменялась · " if job.was_cancelled else ""
            if job.status == "queued":
                job.message = (
                    f"{tag}В очереди — позиция {pos}"
                    if pos > 1
                    else f"{tag}В очереди — скоро начнётся"
                )
            elif job.status == "approve":
                job.message = f"{tag}Ждёт подтверждения промпта · #{pos}"
            elif job.status == "approved":
                job.message = f"{tag}Промпт подтверждён · #{pos}"
            elif job.status == "queued_gpu":
                job.message = f"{tag}В очереди на GPU · #{pos}"
            elif job.status == PAUSED_STATUS:
                job.message = f"{tag}На паузе · #{pos}"
            job.updated_at = time.time()

    def _queue_item(self, job: Job, *, position: int = 0) -> dict[str, Any]:
        data = job.to_dict()
        data["position"] = position
        reorderable = job.status in REORDERABLE_STATUSES
        data["reorderable"] = reorderable
        data["deletable"] = job.status in PENDING_DELETABLE_STATUSES
        data["gpu_queued"] = job.status == "queued_gpu"
        data["can_demote_gpu"] = job.status == "queued_gpu"
        data["can_edit_prompt"] = job.was_cancelled and job.status in (
            "approve",
            "approved",
            "queued",
        )
        data["prompt_preview"] = (
            job.prompt[:100] + "…" if len(job.prompt) > 100 else job.prompt
        )
        return data

    def queue_snapshot(self, *, finished_page: int = 1) -> dict[str, Any]:
        with self._lock:
            running: dict[str, Any] | None = None
            running_id = self._active_job_id or self._prep_active_id
            if running_id:
                job = self._jobs.get(running_id)
                if job:
                    running = self._queue_item(job, position=0)

            drafts: list[dict[str, Any]] = []
            pending: list[dict[str, Any]] = []
            pos = 0
            for job_id in self._pending_ids:
                job = self._jobs.get(job_id)
                if not job or job.status in TERMINAL_STATUSES:
                    continue
                if job_id == self._active_job_id:
                    continue
                if job.status == DRAFT_STATUS:
                    drafts.append(self._queue_item(job, position=0))
                    continue
                if job.status in PREP_WAIT_STATUSES | GPU_WAIT_STATUSES | APPROVED_STATUSES | {PAUSED_STATUS}:
                    pos += 1
                    pending.append(self._queue_item(job, position=pos))

            finished: list[dict[str, Any]] = []
            active_or_pending = set(self._pending_ids)
            if self._active_job_id:
                active_or_pending.add(self._active_job_id)
            candidates = [
                j
                for j in self._jobs.values()
                if j.id not in active_or_pending and j.status in TERMINAL_STATUSES
            ]
            candidates.sort(key=lambda j: j.updated_at, reverse=True)
            finished_total = len(candidates)
            finished_pages = max(
                1, (finished_total + FINISHED_PAGE_SIZE - 1) // FINISHED_PAGE_SIZE
            )
            page = max(1, min(int(finished_page or 1), finished_pages))
            start = (page - 1) * FINISHED_PAGE_SIZE
            for job in candidates[start : start + FINISHED_PAGE_SIZE]:
                finished.append(self._queue_item(job, position=0))

            draft_ready = sum(1 for d in drafts if d.get("draft_ready"))
            approved_count = sum(1 for p in pending if p.get("approved"))
            queued_gpu_count = sum(1 for p in pending if p.get("queued_gpu"))
            paused_gpu_count = sum(1 for p in pending if p.get("paused"))
            gpu_session_active = self.gpu_session_active_unlocked()
            reorderable_ids = [
                jid
                for jid in self._pending_ids
                if (job := self._jobs.get(jid))
                and job.status in REORDERABLE_STATUSES
                and jid != self._active_job_id
            ]
            return {
                "running": running,
                "drafts": drafts,
                "pending": pending,
                "finished": finished,
                "finished_page": page,
                "finished_page_size": FINISHED_PAGE_SIZE,
                "finished_total": finished_total,
                "finished_pages": finished_pages,
                "reorderable_ids": reorderable_ids,
                "draft_count": len(drafts),
                "draft_ready_count": draft_ready,
                "approved_count": approved_count,
                "queued_gpu_count": queued_gpu_count,
                "paused_gpu_count": paused_gpu_count,
                "gpu_session_active": gpu_session_active,
                "gpu_session_model": self._gpu_session_model,
                "gpu_session_steps": self._gpu_session_steps,
                "gpu_confirm_only": self._gpu_session_model is not None,
                "can_prepare": draft_ready > 0,
                "can_skip_prep": draft_ready > 0,
                "can_start_generation": approved_count > 0 and paused_gpu_count == 0,
                "prep_in_progress": self.prep_still_running_unlocked(),
                "pending_count": len(pending),
                "has_work": bool(running or pending or drafts),
            }

    def prep_still_running(self) -> bool:
        with self._lock:
            return self.prep_still_running_unlocked()

    def reorder_pending(self, ordered_ids: list[str]) -> tuple[bool, str]:
        with self._lock:
            reorderable = [
                jid
                for jid in self._pending_ids
                if (job := self._jobs.get(jid))
                and job.status in REORDERABLE_STATUSES
                and jid != self._active_job_id
            ]
            if set(ordered_ids) != set(reorderable):
                return False, "Список задач не совпадает с очередью"
            if len(ordered_ids) != len(reorderable):
                return False, "Список задач не совпадает с очередью"
            reorder_iter = iter(ordered_ids)
            new_pending: list[str] = []
            for job_id in self._pending_ids:
                job = self._jobs.get(job_id)
                if not job:
                    continue
                if job_id in reorderable:
                    new_pending.append(next(reorder_iter))
                else:
                    new_pending.append(job_id)
            self._pending_ids = new_pending
            self._update_queue_positions_unlocked()
            return True, ""

    def _drop_job_unlocked(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)
        self._approval_events.pop(job_id, None)
        self._approval_outcome.pop(job_id, None)
        self._interrupt_hooks.pop(job_id, None)

    def remove_queued(self, job_id: str) -> tuple[bool, str]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False, "Задача не найдена"
            if job.status in (DRAFT_STATUS, "queued", "approve", "approved", "queued_gpu", PAUSED_STATUS):
                if job_id in self._pending_ids:
                    self._pending_ids.remove(job_id)
                self._release_slots_unlocked(job_id)
                self._drop_job_unlocked(job_id)
                return True, ""
            if job.status in TERMINAL_STATUSES:
                self._drop_job_unlocked(job_id)
                return True, ""
            return False, "Сначала отмените выполняющуюся задачу"

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(
        self,
        job_id: str,
        *,
        status: str | None = None,
        progress: int | None = None,
        phase: int | None = None,
        phase_total: int | None = None,
        step: int | None = None,
        total_steps: int | None = None,
        message: str | None = None,
        prompt_used: str | None = None,
        prompt_translation: str | None = None,
        note: str | None = None,
        result: bytes | None = None,
        error: str | None = None,
        cancel_requested: bool | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            if status is not None:
                if status == "generate" and job.generation_started_at is None:
                    job.generation_started_at = time.time()
                if status == "done" and job.generation_finished_at is None:
                    job.generation_finished_at = time.time()
                job.status = status
            if cancel_requested is not None:
                job.cancel_requested = cancel_requested
            if progress is not None:
                job.progress = max(0, min(100, progress))
            if phase is not None:
                job.phase = phase
            if phase_total is not None:
                job.phase_total = phase_total
            if step is not None:
                job.step = step
            if total_steps is not None:
                job.total_steps = total_steps
            if message is not None:
                job.message = message
            if prompt_used is not None:
                job.prompt_used = prompt_used
            if prompt_translation is not None:
                job.prompt_translation = prompt_translation
            if note is not None:
                job.note = note
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error
            job.updated_at = time.time()

    def set_preview(
        self,
        job_id: str,
        preview: bytes,
        *,
        step: int,
        total_steps: int,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.preview = preview
            job.preview_step = step
            job.preview_total_steps = total_steps
            job.updated_at = time.time()

    def log_activity(
        self,
        job_id: str,
        text: str,
        *,
        phase: int | None = None,
        status: str | None = None,
        step: int = 0,
        total_steps: int = 0,
        force: bool = False,
        replace_in_progress: bool = False,
        checkpoint: bool = False,
    ) -> None:
        text = text.strip()
        if not text:
            return
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            if replace_in_progress and job.activities:
                last = job.activities[-1]
                new_status = status if status is not None else job.status
                last_status = last.get("status") or job.status
                # Never replace a finalized checkpointed step with a wait pulse.
                if last.get("checkpoint") and not checkpoint:
                    pass
                elif last_status == new_status and new_status in ("generate", "load_model"):
                    same_substep = new_status != "generate" or (
                        total_steps > 0
                        and last.get("step") == step
                        and last.get("total_steps") == total_steps
                    )
                    if same_substep:
                        # Wait-pulse rows must refresh timestamp/text so the UI
                        # shows the step is still alive (elapsed time changes).
                        keep_time = bool(last.get("checkpoint")) and not checkpoint
                        entry: dict[str, Any] = {
                            "t": last.get("t", time.time()) if keep_time else time.time(),
                            "text": text,
                            "phase": phase if phase is not None else job.phase,
                            "status": new_status,
                        }
                        if total_steps > 0:
                            entry["step"] = step
                            entry["total_steps"] = total_steps
                        if checkpoint or last.get("checkpoint"):
                            entry["checkpoint"] = True
                        job.activities[-1] = entry
                        job.updated_at = time.time()
                        return
            if not force and job.activities and job.activities[-1]["text"] == text:
                return
            entry = {
                "t": time.time(),
                "text": text,
                "phase": phase if phase is not None else job.phase,
                "status": status if status is not None else job.status,
            }
            if total_steps > 0:
                entry["step"] = step
                entry["total_steps"] = total_steps
            if checkpoint:
                entry["checkpoint"] = True
            job.activities.append(entry)
            if len(job.activities) > MAX_ACTIVITY_LOG:
                job.activities = job.activities[-MAX_ACTIVITY_LOG:]
            job.updated_at = time.time()

    def gpu_lock(self) -> threading.Lock:
        return self._gpu_lock

    def should_abort(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            return bool(job and job.cancel_requested)

    def should_cancel_generation(self, job_id: str) -> bool:
        return self.should_abort(job_id)

    def _wake_approval_rejected(self, job_id: str) -> None:
        self._approval_outcome[job_id] = (False, None)
        event = self._approval_events.get(job_id)
        if event:
            event.set()

    def should_pause(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            return bool(job and job.pause_requested)

    def request_pause(self, job_id: str) -> tuple[bool, str]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False, "Задача не найдена"
            if job.status != "generate":
                return False, "Пауза доступна только во время генерации"
            job.pause_requested = True
            job.updated_at = time.time()
        # Kill GPU subprocess only — do NOT punch JobCancelled into the worker thread.
        self.kill_gpu_process()
        return True, ""

    def ensure_paused(self, job_id: str) -> None:
        """Finalize pause if the worker thread did not do it yet."""
        job = self.get(job_id)
        if not job:
            return
        if job.status == "generate" or job.pause_requested:
            self.finish_pause(job_id)

    def finish_pause(self, job_id: str) -> None:
        from job_checkpoint import has_checkpoint, load_checkpoint

        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            if job.status == PAUSED_STATUS:
                return
            was_requested = bool(job.pause_requested)
            job.pause_requested = False

            paused_step = int(job.step or 0)
            total = int(job.total_steps or 0)
            if has_checkpoint(job_id):
                loaded = load_checkpoint(job_id)
                if loaded:
                    meta = loaded[0]
                    paused_step = int(meta.get("completed_step") or paused_step)
                    total = int(meta.get("total_steps") or total)
                    if total:
                        job.total_steps = total

            job.paused_at_step = paused_step
            if paused_step:
                job.step = paused_step
            job.status = PAUSED_STATUS
            job.progress = 0
            if paused_step and total:
                if was_requested:
                    job.message = f"На паузе — продолжение с шага {paused_step}/{total}"
                else:
                    job.message = (
                        f"Экстренная пауза — откат к чекпоинту шага {paused_step}/{total}"
                    )
            elif has_checkpoint(job_id):
                job.message = "На паузе — можно продолжить"
            else:
                job.message = "На паузе — продолжить с начала"
            job.updated_at = time.time()
            self._release_slots_unlocked(job_id)
            self._gpu_session_model = None
            self._gpu_session_steps = 0
            self._gpu_session_lora = ""
            self._update_queue_positions_unlocked()
            self._sync_gpu_session_unlocked()
        try:
            from gpu_supervisor import clear_step_timers

            clear_step_timers(job_id)
        except Exception:
            pass

    def resume_paused_job(self, job_id: str) -> tuple[bool, str]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False, "Задача не найдена"
            if job.status != PAUSED_STATUS:
                return False, "Задача не на паузе"
            step = job.paused_at_step or job.step
            total = job.total_steps
            job.status = "queued_gpu"
            if step and total:
                job.message = f"В очереди — возобновление с шага {step}/{total}"
            else:
                job.message = "В очереди на GPU — генерация с начала"
            job.pause_requested = False
            if job.model_id:
                self._gpu_session_model = job.model_id
            if job.inference_steps:
                self._gpu_session_steps = job.inference_steps
            if job.lora_id:
                self._gpu_session_lora = job.lora_id
            job.updated_at = time.time()
            self._update_queue_positions_unlocked()
        return True, ""

    def demote_from_gpu(self, job_id: str) -> tuple[bool, str]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False, "Задача не найдена"
            if job.status != "queued_gpu":
                return False, "Элемент не в очереди на генерацию"
            job.status = "approved"
            job.message = "Промпт подтверждён — убрано из очереди генерации"
            job.updated_at = time.time()
            self._sync_gpu_session_unlocked()
            self._update_queue_positions_unlocked()
        return True, ""

    def _apply_requeue_after_cancel_unlocked(self, job: Job, old_status: str) -> None:
        job.cancel_requested = False
        job.pause_requested = False
        job.was_cancelled = True
        job.progress = 0
        job.preview = None
        job.preview_step = 0
        job.preview_total_steps = 0
        job.step = 0
        job.total_steps = 0
        job.paused_at_step = 0
        job.generation_started_at = None
        job.generation_finished_at = None
        job.error = None
        if old_status in (
            "generate",
            "translate",
            PAUSED_STATUS,
            "queued_gpu",
            "approve",
            "approved",
            "queued",
        ):
            job.status = "approve"
        else:
            job.status = "queued"
        if job.id not in self._pending_ids:
            self._pending_ids.append(job.id)
        job.updated_at = time.time()
        self._sync_gpu_session_unlocked()
        self._update_queue_positions_unlocked()

    def requeue_after_cancel(self, job_id: str, old_status: str | None = None) -> None:
        from job_checkpoint import delete_checkpoint

        delete_checkpoint(job_id)
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status in (DRAFT_STATUS, "done", "error"):
                return
            if job.was_cancelled and job.status in (
                "queued",
                "approve",
                "approved",
                "queued_gpu",
                PAUSED_STATUS,
            ):
                return
            if old_status is None:
                if job.status in ("generate", "translate", PAUSED_STATUS, "queued_gpu", "approve"):
                    old_status = job.status
                elif job.status == "cancelled":
                    old_status = "generate"
                else:
                    return
            self._apply_requeue_after_cancel_unlocked(job, old_status)

    def request_cancel(self, job_id: str) -> tuple[bool, str]:
        old_status: str | None = None
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False, "Задача не найдена"
            if job.status == DRAFT_STATUS:
                if job_id in self._pending_ids:
                    self._pending_ids.remove(job_id)
                self._release_slots_unlocked(job_id)
                self._drop_job_unlocked(job_id)
                return True, ""
            if job.status == "queued":
                self._apply_requeue_after_cancel_unlocked(job, "queued")
                return True, ""
            if job.status in TERMINAL_STATUSES:
                return False, "Задача уже завершена"
            old_status = job.status
            job.cancel_requested = True
            job.updated_at = time.time()
            self._release_slots_unlocked(job_id)
            if old_status == "approve":
                self._wake_approval_rejected(job_id)
        self.trigger_interrupt(job_id)
        self.kill_gpu_process()
        if old_status in ("generate", "translate", "queued_gpu", PAUSED_STATUS):
            self.punch_gpu_workers()
        self.requeue_after_cancel(job_id, old_status)
        try:
            from job_logger import log_job

            log_job(job_id, f"CANCEL requested (requeued, was {old_status})")
        except Exception:
            pass
        return True, ""

    def wait_for_approval(
        self,
        job_id: str,
        *,
        timeout: float = APPROVAL_TIMEOUT_SEC,
    ) -> tuple[bool, str | None]:
        with self._lock:
            self._approval_outcome.pop(job_id, None)
        event = self._approval_events.get(job_id)
        if event is None:
            return False, None
        event.clear()
        if not event.wait(timeout=timeout):
            return False, None
        with self._lock:
            return self._approval_outcome.get(job_id, (False, None))

    def auto_approve_prompt(self, job_id: str) -> tuple[bool, str]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False, "Задача не найдена"
            if job.cancel_requested:
                return False, "Задача отменена"
            if not job.prompt_used:
                return False, "Пустой промпт"
            job.status = "approved"
            job.phase = 2
            job.progress = 100
            job.message = "Промпт подтверждён — ждёт запуска генерации"
            job.updated_at = time.time()
            self._update_queue_positions_unlocked()
        return True, ""

    def approve_prompt(self, job_id: str, prompt_used: str | None = None) -> tuple[bool, str]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False, "Задача не найдена"
            if job.status != "approve":
                return False, "Сейчас не требуется подтверждение"
            if job.cancel_requested:
                return False, "Задача отменена"
            if prompt_used is not None:
                prompt_used = prompt_used.strip()
                if prompt_used:
                    job.prompt_used = prompt_used
            if not job.prompt_used:
                return False, "Пустой промпт"
            job.status = "approved"
            job.phase = 2
            job.progress = 100
            job.message = "Промпт подтверждён — ждёт запуска генерации"
            job.updated_at = time.time()
            self._approval_outcome[job_id] = (True, job.prompt_used)
            event = self._approval_events.get(job_id)
            self._update_queue_positions_unlocked()
        if event:
            event.set()
        return True, ""

    def edit_requeued_prompt(self, job_id: str, prompt: str) -> tuple[bool, str]:
        prompt = prompt.strip()
        if not prompt:
            return False, "Введите промпт"
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False, "Задача не найдена"
            if not job.was_cancelled:
                return False, "Редактирование доступно только после отмены"
            if job.status not in ("approve", "approved", "queued"):
                return False, "Сейчас промпт изменить нельзя"
            job.prompt = prompt
            job.prompt_used = None
            job.prompt_translation = None
            job.note = None
            job.status = "queued"
            job.phase = 0
            job.progress = 0
            job.updated_at = time.time()
            self._update_queue_positions_unlocked()
        return True, ""

    def reprep_prompt(self, job_id: str, prompt: str) -> tuple[bool, str]:
        prompt = prompt.strip()
        if not prompt:
            return False, "Введите промпт"
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False, "Задача не найдена"
            if job.status != "approve":
                return False, "Перегенерация доступна только на этапе подтверждения"
            job.prompt = prompt
            job.prompt_used = None
            job.prompt_translation = None
            job.note = None
            job.status = "queued"
            job.phase = 0
            job.progress = 0
            job.message = "В очереди — повторная подготовка промпта"
            job.updated_at = time.time()
            self._update_queue_positions_unlocked()
        return True, ""

    def promote_approved_to_gpu(
        self, model_id: str, inference_steps: int = 0, lora_id: str = ""
    ) -> tuple[int, list[str]]:
        from image_editor import inference_steps_for_model

        with self._lock:
            if not self._gpu_session_model:
                self._gpu_session_model = model_id
                self._gpu_session_steps = inference_steps_for_model(
                    model_id, inference_steps or None
                )
                self._gpu_session_lora = lora_id or ""
            elif self._gpu_session_model != model_id:
                model_id = self._gpu_session_model
            steps = self._gpu_session_steps
            session_lora = getattr(self, "_gpu_session_lora", "") or lora_id
            promoted: list[str] = []
            for job_id in self._pending_ids:
                job = self._jobs.get(job_id)
                if job and job.status == "approved":
                    job.model_id = model_id
                    job.inference_steps = steps
                    job.lora_id = session_lora
                    job.status = "queued_gpu"
                    job.message = f"В очереди на GPU — {steps} шагов"
                    job.updated_at = time.time()
                    promoted.append(job_id)
            if promoted:
                self._update_queue_positions_unlocked()
            return len(promoted), promoted

    def prune_old(self, max_age_sec: int = 3600) -> None:
        cutoff = time.time() - max_age_sec
        with self._lock:
            stale = [
                jid
                for jid, j in self._jobs.items()
                if j.updated_at < cutoff
                and j.status in TERMINAL_STATUSES
                and jid != self._active_job_id
                and jid not in self._pending_ids
            ]
            for jid in stale:
                del self._jobs[jid]
                self._approval_events.pop(jid, None)
                self._approval_outcome.pop(jid, None)
                self._interrupt_hooks.pop(jid, None)


JOBS = JobStore()
