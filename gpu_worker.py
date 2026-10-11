"""Run image generation in an isolated process (killable on cancel/pause)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import portable_env  # noqa: F401 — sets HF cache under project root

ROOT = Path(__file__).resolve().parent


def _file_pipeline(progress_path: Path):
    from pipeline_progress import PipelineProgress

    def on_update(**fields) -> None:
        fields["t"] = __import__("time").time()
        progress_path.write_text(json.dumps(fields, ensure_ascii=False), encoding="utf-8")

    return PipelineProgress(on_update)


def main(work_dir: str) -> int:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    work = Path(work_dir)
    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    job_id = manifest["job_id"]
    model_id = manifest["model_id"]
    prompt = manifest["prompt"]
    approved_prompt = manifest.get("approved_prompt") or manifest.get("flux_prompt")
    resume = bool(manifest.get("resume"))
    inference_steps = int(manifest.get("inference_steps") or 0) or None
    lora_id = str(manifest.get("lora_id") or "")

    from image_editor import edit_image, resolve_lora_path, uses_joycaption_flow
    from job_errors import JobCancelled, JobPaused

    raw = (work / "input.bin").read_bytes()
    reference_list: list[bytes] = []
    ref_count = int(manifest.get("reference_count") or 0)
    if manifest.get("has_reference"):
        if ref_count <= 0:
            ref_count = 1
        for index in range(ref_count):
            path = work / f"reference-{index}.bin"
            if not path.is_file() and index == 0:
                path = work / "reference.bin"
            if path.is_file():
                reference_list.append(path.read_bytes())
    pipeline = _file_pipeline(work / "progress.json")
    pause_flag = work / "pause.flag"

    def pause_checker() -> bool:
        return pause_flag.is_file()

    try:
        result = edit_image(
            raw,
            prompt,
            model_id=model_id,
            pipeline=pipeline,
            cancel_checker=None,
            pause_checker=pause_checker,
            approved_prompt=approved_prompt if uses_joycaption_flow(model_id) else None,
            job_id=job_id,
            resume_job_id=job_id if resume else None,
            inference_steps=inference_steps,
            reference_images_bytes=reference_list or None,
            lora_path=resolve_lora_path(lora_id) if lora_id else None,
        )
        (work / "result.png").write_bytes(result)
        (work / "done.json").write_text(
            json.dumps({"ok": True, "bytes": len(result)}),
            encoding="utf-8",
        )
        return 0
    except JobPaused:
        (work / "done.json").write_text(
            json.dumps({"ok": False, "error": "paused", "kind": "JobPaused"}),
            encoding="utf-8",
        )
        return 2
    except JobCancelled:
        (work / "done.json").write_text(
            json.dumps({"ok": False, "error": "cancelled", "kind": "JobCancelled"}),
            encoding="utf-8",
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        from job_logger import log_generation_error

        log_generation_error(
            job_id,
            exc,
            model_id=model_id,
            phase="gpu_worker",
            resumed=resume,
            kind=type(exc).__name__,
        )
        (work / "done.json").write_text(
            json.dumps({"ok": False, "error": str(exc), "kind": type(exc).__name__}),
            encoding="utf-8",
        )
        return 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: gpu_worker.py <work_dir>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
