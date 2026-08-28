"""Persistent file logging for job diagnostics (survives server restarts)."""

from __future__ import annotations

import json
import logging
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / os.environ.get("PHOTO_EDIT_LOG", "photo-edit.log")
GEN_ERRORS_FILE = LOG_DIR / "generation-errors.jsonl"

_logger: logging.Logger | None = None


def get_logger() -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _logger = logging.getLogger("photo_edit")
    _logger.setLevel(logging.DEBUG)
    if not _logger.handlers:
        handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        _logger.addHandler(handler)
    _logger.propagate = False
    return _logger


def log_line(message: str, level: int = logging.INFO) -> None:
    get_logger().log(level, message)


def log_job(job_id: str, message: str, level: int = logging.INFO) -> None:
    get_logger().log(level, "[job:%s] %s", job_id[:8], message)


def log_job_exception(job_id: str, context: str, exc: BaseException) -> None:
    get_logger().error(
        "[job:%s] %s: %s\n%s",
        job_id[:8],
        context,
        exc,
        traceback.format_exc(),
    )


def log_generation_info(job_id: str, event: str, **fields: Any) -> None:
    """Structured info for generation monitoring (JSONL + text log)."""
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": "info",
        "job_id": job_id,
        "event": event,
        **fields,
    }
    _append_generation_jsonl(payload)
    get_logger().info("[job:%s] GEN %s %s", job_id[:8], event, _compact_fields(fields))


def log_generation_error(
    job_id: str,
    exc: BaseException,
    *,
    model_id: str = "",
    phase: str = "generate",
    resumed: bool = False,
    step: int = 0,
    total_steps: int = 0,
    **extra: Any,
) -> None:
    """Structured generation error for monitoring (JSONL + text log with traceback)."""
    tb = traceback.format_exc()
    if not tb or tb.strip() == "NoneType: None\n":
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": "error",
        "job_id": job_id,
        "event": "generation_error",
        "model_id": model_id,
        "phase": phase,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "resumed": resumed,
        "step": step,
        "total_steps": total_steps,
        "traceback": tb,
        **extra,
    }
    _append_generation_jsonl(payload)
    get_logger().error(
        "[job:%s] GEN_ERROR model=%s phase=%s resumed=%s step=%s/%s %s: %s\n%s",
        job_id[:8],
        model_id or "?",
        phase,
        resumed,
        step or "?",
        total_steps or "?",
        type(exc).__name__,
        exc,
        tb,
    )


def _compact_fields(fields: dict[str, Any]) -> str:
    parts = []
    for key, value in fields.items():
        if value is None or value == "":
            continue
        text = str(value)
        if len(text) > 120:
            text = text[:117] + "..."
        parts.append(f"{key}={text}")
    return " ".join(parts)


def _append_generation_jsonl(payload: dict[str, Any]) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with GEN_ERRORS_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        get_logger().exception("failed to write generation-errors.jsonl")


def open_log_in_system_editor() -> Path:
    """Open the main log file in the OS default text editor."""
    import subprocess
    import sys

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_FILE.is_file():
        LOG_FILE.touch()
    path = LOG_FILE.resolve()
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)], start_new_session=True)
    else:
        subprocess.Popen(["xdg-open", str(path)], start_new_session=True)
    return path
