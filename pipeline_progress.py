"""Numbered pipeline steps; progress bar shows 0–100% within the current step only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

UpdateFn = Callable[..., None]

PHASES: list[tuple[str, str]] = [
    ("translate", "Перевод и подготовка промпта"),
    ("approve", "Подтверждение промпта"),
    ("load_model", "Загрузка модели в GPU"),
    ("prepare", "Подготовка изображения"),
    ("generate", "Генерация на GPU"),
    ("done", "Готово"),
]

PHASE_INDEX = {phase_id: idx + 1 for idx, (phase_id, _) in enumerate(PHASES)}
PHASE_TOTAL = len(PHASES)


@dataclass(frozen=True)
class ProgressSnapshot:
    status: str
    phase: int
    phase_total: int
    progress: int
    message: str
    sub_step: int = 0
    sub_total: int = 0


class PipelineProgress:
    def __init__(self, on_update: UpdateFn | None = None) -> None:
        self._on_update = on_update

    def report(
        self,
        phase_id: str,
        fraction: float = 0.0,
        *,
        detail: str = "",
        sub_step: int = 0,
        sub_total: int = 0,
        log_journal: bool = True,
        journal_replace: bool = False,
        checkpoint: bool = False,
    ) -> None:
        if phase_id not in PHASE_INDEX:
            raise ValueError(f"Неизвестный этап: {phase_id}")

        _, title = PHASES[PHASE_INDEX[phase_id] - 1]
        fraction = max(0.0, min(1.0, fraction))
        step_progress = int(round(fraction * 100))
        message = f"{PHASE_INDEX[phase_id]}. {title}"
        if detail:
            message = f"{message} — {detail}"

        if self._on_update:
            self._on_update(
                status=phase_id,
                phase=PHASE_INDEX[phase_id],
                phase_total=PHASE_TOTAL,
                progress=step_progress,
                message=message,
                step=sub_step,
                total_steps=sub_total,
                log_journal=log_journal,
                journal_replace=journal_replace,
                checkpoint=checkpoint,
            )
