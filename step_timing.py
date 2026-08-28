"""Diffusion step timing: per-job tracker + persistent stats for ETA."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATS_PATH = ROOT / "logs" / "diffusion-step-stats.json"
MAX_SAMPLES_PER_MODEL = 80

_lock = threading.Lock()


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    if total < 60:
        return f"{total} с"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes} мин {secs} с" if secs else f"{minutes} мин"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} ч {minutes} мин"


def format_eta_per_step(seconds: float | None) -> str:
    if seconds is None:
        return "неизвестно"
    return f"~{format_duration(seconds)}/шаг"


def _load_stats() -> dict[str, list[float]]:
    try:
        if STATS_PATH.is_file():
            data = json.loads(STATS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {k: list(v) for k, v in data.items() if isinstance(v, list)}
    except Exception:
        pass
    return {}


def _save_stats(data: dict[str, list[float]]) -> None:
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def record_global_step(model_id: str, duration_sec: float) -> None:
    if duration_sec <= 0:
        return
    with _lock:
        stats = _load_stats()
        samples = stats.setdefault(model_id, [])
        samples.append(round(duration_sec, 2))
        if len(samples) > MAX_SAMPLES_PER_MODEL:
            samples = samples[-MAX_SAMPLES_PER_MODEL:]
        stats[model_id] = samples
        _save_stats(stats)


def global_avg_step_sec(model_id: str) -> float | None:
    with _lock:
        samples = _load_stats().get(model_id) or []
    if not samples:
        return None
    return sum(samples) / len(samples)


class DiffusionStepTracker:
    """Tracks in-flight diffusion step and completed step durations."""

    def __init__(self, total_steps: int, model_id: str = "unknown") -> None:
        self.total_steps = total_steps
        self.model_id = model_id
        self._completed: list[float] = []
        self._working_step = 1
        self._step_started = time.monotonic()
        self._step_avg_hint: float | None = None
        self._last_completed_avg_hint: float | None = None
        self._lock = threading.RLock()

    def begin_step(self, step: int) -> None:
        with self._lock:
            self._working_step = max(1, min(step, self.total_steps))
            self._step_started = time.monotonic()
            self._step_avg_hint = self.estimate_step_sec()

    def complete_step(self, step: int) -> float:
        with self._lock:
            avg_hint = self._step_avg_hint
            elapsed = max(0.0, time.monotonic() - self._step_started)
            self._completed.append(elapsed)
            record_global_step(self.model_id, elapsed)
            if step < self.total_steps:
                self._working_step = step + 1
                self._step_started = time.monotonic()
                self._step_avg_hint = self.estimate_step_sec()
            self._last_completed_avg_hint = avg_hint
            return elapsed

    def avg_hint_for_display(self) -> float | None:
        with self._lock:
            if self._step_avg_hint is not None:
                return self._step_avg_hint
            return global_avg_step_sec(self.model_id)

    def last_completed_avg_hint(self) -> float | None:
        with self._lock:
            return getattr(self, "_last_completed_avg_hint", None)

    def working_step(self) -> int:
        with self._lock:
            return self._working_step

    def completed_count(self) -> int:
        with self._lock:
            return len(self._completed)

    def _job_avg_sec(self) -> float | None:
        with self._lock:
            if not self._completed:
                return None
            return sum(self._completed) / len(self._completed)

    def estimate_step_sec(self) -> float | None:
        job_avg = self._job_avg_sec()
        if job_avg is not None:
            return job_avg
        return global_avg_step_sec(self.model_id)

    def detail_waiting(self) -> str:
        step = self.working_step()
        elapsed = self.elapsed_on_current_step()
        avg = self.avg_hint_for_display()
        parts = [
            f"шаг {step} из {self.total_steps}",
            f"прошло {format_duration(elapsed)}",
        ]
        if avg is not None:
            parts.append(f"среднее ~{format_duration(avg)}")
            remaining = max(0.0, avg - elapsed)
            if remaining > 0:
                parts.append(f"ещё ~{format_duration(remaining)}")
        else:
            parts.append("оценка пока неизвестна")
        return ", ".join(parts)

    def detail_step_completed(self, step: int, duration_sec: float) -> str:
        avg = self.last_completed_avg_hint()
        if avg is None:
            avg = self.avg_hint_for_display()
        parts = [f"шаг {step} из {self.total_steps}"]
        if avg is not None:
            parts.append(f"среднее время {format_duration(avg)}")
        parts.append(f"завершён за {format_duration(duration_sec)}")
        return ", ".join(parts)

    def detail_start(self) -> str:
        step = self.working_step()
        avg = self.avg_hint_for_display()
        avg_text = format_duration(avg) if avg is not None else "неизвестно"
        return f"шаг {step} из {self.total_steps}, среднее время {avg_text}"

    def elapsed_on_current_step(self) -> float:
        with self._lock:
            return max(0.0, time.monotonic() - self._step_started)
