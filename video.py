"""Local text-to-video and image-to-video. One GPU job at a time."""

from __future__ import annotations

import gc
import os
import threading
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import portable_env  # noqa: F401
from portable_env import ROOT

VIDEO_ROOT = ROOT / "Video"

MODELS = (
    {
        "id": "wan22",
        "name": "Wan 2.2",
        "vendor": "Alibaba",
        "repo_t2v": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "repo_i2v": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "fps": 24,
        "size": "около 10–15 ГБ",
        "hint": "Текст или фото. 5B, 720p на 4090. Картинка и движение самые крепкие из трёх.",
    },
    {
        "id": "ltx23",
        "name": "LTX-2.3",
        "vendor": "Lightricks",
        "repo": "diffusers/LTX-2.3-Diffusers",
        "fps": 24,
        "size": "около 20–40 ГБ",
        "hint": "Текст или фото. Быстрее остальных, может сразу положить звук в ролик.",
    },
    {
        "id": "hunyuan15",
        "name": "HunyuanVideo 1.5",
        "vendor": "Tencent",
        "repo_t2v": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        "repo_i2v": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v",
        "fps": 24,
        "size": "около 20–30 ГБ",
        "hint": "Текст или фото. Сильнее по лицам. На 24 ГБ идёт в 480p, иначе не влезает.",
    },
)

_pipe: Any = None
_pipe_key: str | None = None


class VideoJob:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.phase = "idle"
        self.message = ""
        self.error: str | None = None
        self.progress = 0
        self.model_id = "wan22"
        self.prompt = ""
        self.result_path: str | None = None
        self.cancel = False
        self.phase_started = 0.0
        self.seconds: float | None = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            busy = self.phase in ("load", "generate")
            elapsed = int(time.monotonic() - self.phase_started) if busy and self.phase_started else 0
            return {
                "phase": self.phase,
                "busy": busy,
                "message": self.message,
                "error": self.error,
                "progress": self.progress,
                "model_id": self.model_id,
                "prompt": self.prompt,
                "elapsed_sec": elapsed,
                "seconds": self.seconds,
                "has_result": bool(self.result_path and Path(self.result_path).is_file()),
                "result_name": Path(self.result_path).name if self.result_path else None,
                "library_root": str(VIDEO_ROOT),
                "models": [dict(item) for item in MODELS],
            }


VIDEO = VideoJob()


def release_video_gpu() -> None:
    global _pipe, _pipe_key
    _pipe = None
    _pipe_key = None
    _cuda_empty()


def _cuda_empty() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _aborted() -> bool:
    with VIDEO._lock:
        return VIDEO.cancel


def _status(phase: str, progress: int, message: str) -> None:
    with VIDEO._lock:
        VIDEO.phase = phase
        VIDEO.progress = progress
        VIDEO.message = message


def _spec(model_id: str) -> dict[str, Any]:
    for item in MODELS:
        if item["id"] == model_id:
            return item
    raise ValueError(f"Неизвестная модель: {model_id}")


def _weights_ready(repo: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        found = try_to_load_from_cache(repo, "model_index.json")
    except Exception:
        return False
    return isinstance(found, str) and Path(found).is_file()


def catalog() -> dict[str, Any]:
    models = []
    for item in MODELS:
        row = dict(item)
        repos = [item.get("repo") or item.get("repo_t2v"), item.get("repo_i2v")]
        row["ready"] = any(_weights_ready(repo) for repo in repos if repo)
        row["i2v"] = True
        models.append(row)
    return {"models": models, "library_root": str(VIDEO_ROOT)}


def _hf_relax() -> None:
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    try:
        from music import _allow_intercepted_ssl

        _allow_intercepted_ssl()
    except Exception:
        pass
    import requests
    import urllib3
    from huggingface_hub import configure_http_backend

    urllib3.disable_warnings()

    def backend() -> requests.Session:
        session = requests.Session()
        session.verify = False
        return session

    configure_http_backend(backend)


def _frames(model_id: str, seconds: float) -> int:
    seconds = max(2.0, min(5.0, float(seconds)))
    if model_id == "ltx23":
        count = max(17, int(round(seconds * 24)))
        k = max(2, round((count - 1) / 8))
        return 8 * k + 1
    count = max(17, int(round(seconds * 24)))
    k = max(4, round((count - 1) / 4))
    return 4 * k + 1


def _size(quality: str) -> tuple[int, int]:
    if quality == "720":
        return 1280, 704
    return 832, 480


def _load_image(path: Path, width: int, height: int):
    from PIL import Image

    image = Image.open(path).convert("RGB")
    scale = max(width / image.width, height / image.height)
    resized = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
    left = max(0, (resized.width - width) // 2)
    top = max(0, (resized.height - height) // 2)
    return resized.crop((left, top, left + width, top + height))


def _pipeline_class(model_id: str, has_image: bool):
    import diffusers

    if model_id == "wan22":
        name = "WanImageToVideoPipeline" if has_image else "WanPipeline"
    elif model_id == "ltx23":
        name = "LTX2ImageToVideoPipeline" if has_image else "LTX2Pipeline"
    else:
        name = "HunyuanVideo15ImageToVideoPipeline" if has_image else "HunyuanVideo15Pipeline"
    if not hasattr(diffusers, name):
        raise RuntimeError(
            f"В установленном diffusers нет {name}. Нужна свежая версия библиотеки, torch при этом не меняется."
        )
    return getattr(diffusers, name), name


def _repo(spec: dict[str, Any], has_image: bool) -> str:
    if has_image and spec.get("repo_i2v"):
        return spec["repo_i2v"]
    return spec.get("repo_t2v") or spec["repo"]


def _load_pipe(model_id: str, has_image: bool):
    global _pipe, _pipe_key
    spec = _spec(model_id)
    repo = _repo(spec, has_image)
    key = f"{model_id}:{repo}"
    if _pipe is not None and _pipe_key == key:
        return _pipe
    import torch

    _hf_relax()
    from image_editor import release_gpu_vram
    from music import release_music_models

    _status("load", 8, "Снимаю фото, музыку и речь с GPU…")
    release_gpu_vram()
    release_music_models()
    try:
        from speech import release_speech_gpu

        release_speech_gpu()
    except Exception:
        pass
    release_video_gpu()
    cls, cls_name = _pipeline_class(model_id, has_image)
    _status("load", 18, f"Читаю {spec['name']} ({repo})…")
    dtype = torch.bfloat16
    kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if model_id == "wan22":
        try:
            from diffusers import AutoencoderKLWan

            kwargs["vae"] = AutoencoderKLWan.from_pretrained(repo, subfolder="vae", torch_dtype=torch.float32)
        except Exception:
            pass
    try:
        pipe = cls.from_pretrained(repo, **kwargs)
    except TypeError:
        kwargs.pop("torch_dtype", None)
        pipe = cls.from_pretrained(repo, dtype=dtype, **kwargs)
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    if hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    _pipe = pipe
    _pipe_key = key
    _pipe._photone_class = cls_name  # type: ignore[attr-defined]
    return pipe


def _export(frames, dest: Path, fps: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    from diffusers.utils import export_to_video

    export_to_video(frames, str(dest), fps=fps)


def _run(model_id: str, prompt: str, image_path: Path | None, seconds: float, seed: int, quality: str):
    import inspect

    import torch

    spec = _spec(model_id)
    has_image = image_path is not None
    width, height = _size("480" if model_id == "hunyuan15" else quality)
    frames_n = _frames(model_id, seconds)
    pipe = _load_pipe(model_id, has_image)
    image = _load_image(image_path, width, height) if image_path is not None else None
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    negative = "worst quality, blurry, jittery, distorted, watermark, text"
    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative,
        "height": height,
        "width": width,
        "num_frames": frames_n,
        "num_inference_steps": 30 if model_id != "ltx23" else 20,
        "frame_rate": float(spec["fps"]),
        "guidance_scale": 5.0 if model_id == "wan22" else 4.0,
        "generator": generator,
    }
    if image is not None:
        kwargs["image"] = image
    accepted = set(inspect.signature(pipe.__call__).parameters)
    kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    if model_id == "hunyuan15" and hasattr(pipe, "guider") and "guidance_scale" not in accepted:
        try:
            pipe.guider = pipe.guider.new(guidance_scale=6.0)
        except Exception:
            pass
    _status("generate", 45, f"{spec['name']} · {frames_n} кадров…")
    result = pipe(**kwargs)
    frames = result.frames[0]
    return frames, int(spec["fps"])


def start_video(
    *,
    model_id: str,
    prompt: str,
    image: Path | None,
    seconds: float,
    seed: int,
    quality: str,
    gpu_busy: bool,
    acquire_gpu,
    release_gpu,
) -> dict[str, Any]:
    _spec(model_id)
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Опишите, что должно произойти в кадре")
    if len(prompt) > 2000:
        raise ValueError("Описание длиннее 2000 символов")
    if gpu_busy:
        raise RuntimeError("GPU занят очередью, сортировкой, картинкой, музыкой или речью")
    quality = "720" if quality == "720" else "480"
    seconds = max(2.0, min(5.0, float(seconds or 3)))
    with VIDEO._lock:
        if VIDEO.phase in ("load", "generate"):
            raise RuntimeError("Уже идёт генерация видео")
        VIDEO.phase = "load"
        VIDEO.phase_started = time.monotonic()
        VIDEO.model_id = model_id
        VIDEO.prompt = prompt
        VIDEO.error = None
        VIDEO.cancel = False
        VIDEO.progress = 2
        VIDEO.message = "Запуск…"
        VIDEO.result_path = None
        VIDEO.seconds = None

    def worker() -> None:
        acquired = False
        try:
            if _aborted():
                raise RuntimeError("Отменено")
            if acquire_gpu is not None:
                acquire_gpu()
                acquired = True
            frames, fps = _run(model_id, prompt, image, seconds, seed, quality)
            if _aborted():
                raise RuntimeError("Отменено")
            stamp = time.strftime("%Y%m%d-%H%M%S")
            dest = VIDEO_ROOT / f"{stamp}-{model_id}.mp4"
            _status("generate", 92, "Сохраняю mp4…")
            _export(frames, dest, fps)
            duration = round(len(frames) / float(fps), 2) if frames is not None else None
            with VIDEO._lock:
                VIDEO.result_path = str(dest)
                VIDEO.seconds = duration
                VIDEO.phase = "done"
                VIDEO.progress = 100
                VIDEO.message = f"Готово · {duration:.1f} с · {dest.name}" if duration else f"Готово · {dest.name}"
        except Exception as exc:  # noqa: BLE001
            _cuda_empty()
            with VIDEO._lock:
                if VIDEO.cancel:
                    VIDEO.phase = "idle"
                    VIDEO.message = "Отменено"
                    VIDEO.error = None
                else:
                    VIDEO.phase = "error"
                    VIDEO.error = str(exc)
                    VIDEO.message = f"Ошибка: {exc}"
        finally:
            if acquired and release_gpu is not None:
                try:
                    release_gpu()
                except Exception:
                    pass

    threading.Thread(target=worker, daemon=True, name="video-gen").start()
    return VIDEO.snapshot()


def request_video_cancel() -> dict[str, Any]:
    with VIDEO._lock:
        VIDEO.cancel = True
        VIDEO.message = "Остановка…"
    return VIDEO.snapshot()


def result_file() -> Path | None:
    with VIDEO._lock:
        path = VIDEO.result_path
    if not path:
        return None
    file = Path(path)
    return file if file.is_file() else None
