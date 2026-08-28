"""Parent-side supervision of the GPU worker subprocess."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from job_errors import JobCancelled, JobPaused
from job_logger import log_job, log_job_exception, log_generation_error

ROOT = Path(__file__).resolve().parent
_WAIT_LOG_TS: dict[str, float] = {}
_STEP_STARTED_AT: dict[str, float] = {}


def kill_process(proc: subprocess.Popen | None, *, grace_sec: float = 3.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
    except Exception:
        pass
    deadline = time.monotonic() + grace_sec
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.15)
    try:
        proc.kill()
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def _python_exe() -> str:
    venv = ROOT / ".venv" / "Scripts" / "python.exe"
    if venv.is_file():
        return str(venv)
    return sys.executable


def _pause_exit_pending(
    job_id: str,
    work_dir: Path,
    proc: subprocess.Popen,
    pause_checker,
) -> bool:
    if pause_checker and pause_checker():
        return True
    if (work_dir / "pause.flag").is_file():
        return True
    if proc.returncode == 2:
        return True
    from job_store import JOBS, PAUSED_STATUS

    job = JOBS.get(job_id)
    if job and (job.pause_requested or job.status == PAUSED_STATUS):
        return True
    return False


def clear_step_timers(job_id: str) -> None:
    """Drop elapsed clocks for a job (pause / new GPU session). Pause time must not count."""
    prefix = f"{job_id}:"
    for key in [k for k in _STEP_STARTED_AT if k.startswith(prefix)]:
        _STEP_STARTED_AT.pop(key, None)
    for key in [k for k in _WAIT_LOG_TS if k.startswith(prefix)]:
        _WAIT_LOG_TS.pop(key, None)


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    if total < 60:
        return f"{total} с"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes} мин {secs} с" if secs else f"{minutes} мин"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} ч {minutes} мин"


def _enrich_wait_message(job_id: str, data: dict, msg: str) -> str:
    """Add live elapsed for wait pulses. Never count paused time across GPU sessions."""
    status = data.get("status")
    step = int(data.get("step") or 0)
    if status != "generate" or step <= 0:
        return msg
    if bool(data.get("checkpoint")):
        _STEP_STARTED_AT.pop(f"{job_id}:{step}", None)
        return msg
    progress = int(data.get("progress") or 0)
    # Completed-looking updates should not be treated as wait pulses.
    if progress >= 99 and "завершён" in msg:
        _STEP_STARTED_AT.pop(f"{job_id}:{step}", None)
        return msg

    # Worker already reports elapsed from its own step clock (resets on resume).
    # Do not overwrite it with a parent clock that may still include a long pause.
    if "прошло" in msg:
        return msg

    key = f"{job_id}:{step}"
    now = time.time()
    low = msg.lower()
    force_reset = (
        progress <= 5
        or "возобновление" in low
        or "первая итерация" in low
    )
    started = None if force_reset else _STEP_STARTED_AT.get(key)
    if started is None:
        stale = [k for k in _STEP_STARTED_AT if k.startswith(f"{job_id}:") and k != key]
        for k in stale:
            _STEP_STARTED_AT.pop(k, None)
        _STEP_STARTED_AT[key] = now
        started = now
    elapsed = max(0.0, now - started)
    elapsed_txt = _format_elapsed(elapsed)
    if "среднее" in msg:
        return msg.replace("среднее", f"прошло {elapsed_txt}, среднее", 1)
    return f"{msg} · прошло {elapsed_txt}"


def _apply_progress(job_id: str, pipeline, data: dict) -> None:
    from job_store import JOBS

    fields = {
        k: data[k]
        for k in (
            "status",
            "phase",
            "phase_total",
            "progress",
            "message",
            "step",
            "total_steps",
        )
        if k in data
    }
    msg = data.get("message")
    if msg:
        msg = _enrich_wait_message(job_id, data, msg)
        fields["message"] = msg
        data = dict(data)
        data["message"] = msg
    if fields:
        JOBS.update(job_id, **fields)
    if msg:
        if "journal_replace" in data:
            journal_replace = bool(data.get("journal_replace"))
        else:
            # Fallback for older progress payloads: replace only in-progress generate rows.
            journal_replace = data.get("status") == "generate" and bool(data.get("step"))
        JOBS.log_activity(
            job_id,
            msg,
            phase=fields.get("phase"),
            status=fields.get("status"),
            step=fields.get("step") or 0,
            total_steps=fields.get("total_steps") or 0,
            replace_in_progress=journal_replace,
            checkpoint=bool(data.get("checkpoint")),
        )
        # File log: always for phase changes / completed steps; throttle wait pulses.
        status = fields.get("status")
        if status in ("load_model", "done", "error", "cancelled"):
            log_job(job_id, msg)
        elif status == "generate":
            checkpoint = bool(data.get("checkpoint"))
            progress = int(fields.get("progress") or 0)
            # Completed step (~100%) or start of step — always log.
            if checkpoint or progress >= 99 or progress == 0 or not journal_replace:
                log_job(job_id, msg)
            else:
                # Wait pulse: log at most once per ~30s so the file stays readable.
                now = time.time()
                key = f"{job_id}:{fields.get('step') or 0}"
                last = _WAIT_LOG_TS.get(key, 0.0)
                if now - last >= 30:
                    _WAIT_LOG_TS[key] = now
                    log_job(job_id, msg)


def run_gpu_subprocess(
    job_id: str,
    raw: bytes,
    model_id: str,
    english_prompt: str,
    *,
    flux_prompt: str | None,
    pipeline,
    cancel_checker,
    on_process,
    resume: bool = False,
    pause_checker=None,
    inference_steps: int | None = None,
    reference_raw: list[bytes] | None = None,
) -> bytes:
    clear_step_timers(job_id)
    work_dir = Path(tempfile.mkdtemp(prefix="photoedit-gpu-", dir=ROOT / "logs"))
    progress_path = work_dir / "progress.json"
    last_progress = ""

    refs = [data for data in (reference_raw or []) if data]
    manifest = {
        "job_id": job_id,
        "model_id": model_id,
        "prompt": english_prompt,
        "flux_prompt": flux_prompt,
        "resume": resume,
        "inference_steps": inference_steps or 0,
        "has_reference": bool(refs),
        "reference_count": len(refs),
    }
    (work_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (work_dir / "input.bin").write_bytes(raw)
    for index, data in enumerate(refs):
        (work_dir / f"reference-{index}.bin").write_bytes(data)
        if index == 0:
            (work_dir / "reference.bin").write_bytes(data)

    proc = subprocess.Popen(
        [_python_exe(), str(ROOT / "gpu_worker.py"), str(work_dir)],
        cwd=str(ROOT),
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    on_process(proc)
    log_job(job_id, f"GPU subprocess pid={proc.pid} resume={resume}")

    def _should_stop() -> str | None:
        if cancel_checker and cancel_checker():
            return "cancel"
        if pause_checker and pause_checker():
            return "pause"
        from job_store import JOBS, PAUSED_STATUS

        job = JOBS.get(job_id)
        if job and job.status == PAUSED_STATUS:
            return "pause"
        return None

    try:
        while proc.poll() is None:
            reason = _should_stop()
            if reason == "cancel":
                log_job(job_id, "CANCEL killing GPU subprocess")
                kill_process(proc)
                raise JobCancelled()
            if reason == "pause":
                log_job(job_id, "PAUSE killing GPU subprocess")
                (work_dir / "pause.flag").write_text("1", encoding="utf-8")
                kill_process(proc)
                raise JobPaused()
            if progress_path.is_file():
                try:
                    text = progress_path.read_text(encoding="utf-8")
                    if text and text != last_progress:
                        last_progress = text
                        _apply_progress(job_id, pipeline, json.loads(text))
                except Exception:
                    pass
            time.sleep(0.35)

        reason = _should_stop()
        if reason == "cancel":
            raise JobCancelled()
        if reason == "pause":
            raise JobPaused()

        done_path = work_dir / "done.json"
        if not done_path.is_file():
            if _pause_exit_pending(job_id, work_dir, proc, pause_checker):
                log_job(job_id, f"PAUSE subprocess exit code={proc.returncode}")
                raise JobPaused()
            # Emergency interrupt (driver reset, game stole GPU, OOM kill, etc.):
            # keep progress via last checkpoint instead of failing the job.
            from job_checkpoint import has_checkpoint

            if has_checkpoint(job_id):
                log_job(
                    job_id,
                    f"EMERGENCY PAUSE — GPU exit code={proc.returncode}, rollback to checkpoint",
                )
                raise JobPaused()
            if proc.returncode not in (0, None):
                exc = RuntimeError(f"GPU-процесс завершился с кодом {proc.returncode}")
                log_generation_error(
                    job_id,
                    exc,
                    model_id=model_id,
                    phase="gpu_subprocess",
                    resumed=resume,
                    returncode=proc.returncode,
                )
                raise exc
            raise JobCancelled()

        done = json.loads(done_path.read_text(encoding="utf-8"))
        if not done.get("ok"):
            err = done.get("error") or "Ошибка GPU-процесса"
            kind = done.get("kind")
            if kind == "JobCancelled":
                raise JobCancelled()
            if kind == "JobPaused":
                raise JobPaused()
            from job_checkpoint import has_checkpoint

            if has_checkpoint(job_id):
                log_job(
                    job_id,
                    f"EMERGENCY PAUSE after GPU error ({kind}): {err}",
                )
                raise JobPaused()
            exc = RuntimeError(err)
            log_generation_error(
                job_id,
                exc,
                model_id=model_id,
                phase="gpu_subprocess",
                resumed=resume,
                subprocess_kind=kind,
                returncode=proc.returncode,
            )
            raise exc

        return (work_dir / "result.png").read_bytes()
    finally:
        kill_process(proc)
        on_process(None)
        try:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass
