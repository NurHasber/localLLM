"""Local YuE2-3B text-to-music jobs. Shares the photo GPU lock; unloads image models first."""

from __future__ import annotations

import gc
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from audio_library import AudioLibrary
from job_logger import get_logger
from portable_env import HF_HOME, ROOT

MUSIC_JOB_ID = "music"
YUE2_ID = "yue2-3b"
ACESTEP_ID = "acestep-1.5"
MOSS_ID = "moss-sfx-v2"
MODEL_ID = YUE2_ID
MODEL_REPO = "m-a-p/YuE2-3B"
VAE_REPO = "m-a-p/YuE2-Vae"
MOSS_REPO = "OpenMOSS-Team/MOSS-SoundEffect-v2.0"
ACE_SRC = ROOT / "vendor" / "ACE-Step-1.5"
ACE_CHECKPOINTS = ROOT / "models" / "acestep"
MOSS_SRC = ROOT / "vendor" / "MOSS-TTS"
LIBRARY_ROOT = ROOT / "Music"
OUT_DIR = LIBRARY_ROOT
STATS_PATH = ROOT / "logs" / "music-stats.jsonl"
COT_CHOICES = ("full", "melody", "off")
AUDIO_EXTS = {".flac", ".wav", ".mp3", ".ogg", ".m4a", ".aac"}
MUSIC_MODELS = (
    {
        "id": YUE2_ID,
        "name": "YuE2-3B",
        "kind": "yue2",
        "hint": "Полная песня с вокалом, 48 кГц. CUDA-graph + FlashAttention. CoT планирует партитуру.",
    },
    {
        "id": ACESTEP_ID,
        "name": "ACE-Step 1.5",
        "kind": "acestep",
        "hint": "Turbo DiT, короткие песни за секунды–минуты. Стиль — caption, текст — lyrics. 48 кГц.",
    },
    {
        "id": MOSS_ID,
        "name": "MOSS-SoundEffect v2.0",
        "kind": "sfx",
        "hint": "Игровые SFX: удар, ветер, гонг. Текст-описание, до 30 с, 48 кГц. Apache-2.0.",
    },
)

DEFAULT_STYLE = (
    "English, acoustic folk, warm female vocal, nylon guitar, light percussion, 90 bpm"
)
DEFAULT_LYRICS = """[Verse]
Hello morning light
A quiet simple song
[Chorus]
Sing it once and let it go
"""
DEFAULT_SFX = (
    "A sharp metal sword hit, short clang, dry game Foley, no music, no voice"
)

_PIPE = None
_ACE = None
_MOSS = None
_PIPE_LOCK = threading.Lock()
_YUE2_NAR_PATCHED = False


class MusicStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.phase = "idle"
        self.message = ""
        self.error: str | None = None
        self.progress = 0
        self.style = DEFAULT_STYLE
        self.lyrics = DEFAULT_LYRICS.strip() + "\n"
        self.edit = ""
        self.source_rel: str | None = None
        self.cot = "off"
        self.seed = 42
        self.lang = "en"
        self.model_id = MODEL_ID
        self.result_path: str | None = None
        self.cancel = False
        self.phase_started = 0.0
        self.load_sec: float | None = None
        self.generate_sec: float | None = None
        self.audio_sec: float | None = None
        self.vram_gb: float | None = None
        self.timing: dict[str, Any] = {}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            busy = self.phase in ("load_model", "generate")
            elapsed = int(time.monotonic() - self.phase_started) if busy and self.phase_started else 0
            rtf = None
            if self.generate_sec and self.audio_sec and self.audio_sec > 0:
                rtf = round(self.generate_sec / self.audio_sec, 3)
            return {
                "phase": self.phase,
                "busy": busy,
                "message": self.message,
                "error": self.error,
                "progress": self.progress,
                "style": self.style,
                "lyrics": self.lyrics,
                "edit": getattr(self, "edit", ""),
                "source_rel": getattr(self, "source_rel", None),
                "cot": self.cot,
                "seed": self.seed,
                "lang": getattr(self, "lang", "en"),
                "model_id": self.model_id,
                "has_result": bool(self.result_path and Path(self.result_path).is_file()),
                "result_url": "/api/music/result" if self.result_path else None,
                "library_root": str(LIBRARY_ROOT),
                "result_rel": _library_rel(self.result_path) if self.result_path else None,
                "elapsed_sec": elapsed,
                "load_sec": self.load_sec,
                "generate_sec": self.generate_sec,
                "audio_sec": self.audio_sec,
                "rtf": rtf,
                "vram_gb": self.vram_gb,
                "timing": self.timing,
                "cot_choices": list(COT_CHOICES),
                "models": list(MUSIC_MODELS),
            }


MUSIC = MusicStore()


def _peak_vram_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.max_memory_allocated() / (1024**3), 2)
    except Exception:
        return None


def load_stats() -> list[dict[str, Any]]:
    if not STATS_PATH.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in STATS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def stats_boards() -> dict[str, Any]:
    rows = [r for r in load_stats() if r.get("ok")]
    last = rows[-1] if rows else None
    gen_vals = [float(r["generate_sec"]) for r in rows if r.get("generate_sec")]
    audio_vals = [float(r["audio_sec"]) for r in rows if r.get("audio_sec")]
    rtf_vals = [float(r["rtf"]) for r in rows if r.get("rtf")]
    load_vals = [float(r["load_sec"]) for r in rows if r.get("load_sec")]

    def avg(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 2) if xs else None

    last_rtf = None
    if last and last.get("generate_sec") and last.get("audio_sec"):
        last_rtf = round(float(last["generate_sec"]) / float(last["audio_sec"]), 3)
    return {
        "count": len(rows),
        "last_generate_sec": last.get("generate_sec") if last else None,
        "avg_generate_sec": avg(gen_vals),
        "last_audio_sec": last.get("audio_sec") if last else None,
        "avg_audio_sec": avg(audio_vals),
        "last_rtf": last_rtf,
        "avg_rtf": avg(rtf_vals),
        "last_load_sec": last.get("load_sec") if last else None,
        "avg_load_sec": avg(load_vals),
        "last_vram_gb": last.get("vram_gb") if last else None,
        "last_cot": last.get("cot") if last else None,
        "last_model_id": last.get("model_id") if last else None,
        "history": rows[-12:],
    }


def _append_stat(row: dict[str, Any]) -> None:
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _hub_cache_dir() -> Path:
    user_hub = Path.home() / ".cache" / "huggingface" / "hub"
    if (user_hub / "models--m-a-p--YuE2-3B").exists():
        return user_hub
    return Path(os.environ.get("HUGGINGFACE_HUB_CACHE") or (HF_HOME / "hub"))


def _allow_intercepted_ssl() -> None:
    """Windows AV/proxy often injects a self-signed cert into the chain."""
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001
    os.environ.setdefault("CURL_CA_BUNDLE", "")
    os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
    os.environ.setdefault("SSL_CERT_FILE", "")
    try:
        import urllib3

        urllib3.disable_warnings()
    except Exception:
        pass


def _cuda_empty() -> None:
    try:
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _relax_cuda_budget() -> None:
    """YuE2 caps the allocator at budget-2 GiB (20 of 24), then NAR OOMs."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(1.0)
            torch.cuda.empty_cache()
    except Exception:
        pass


def _unload_yue2() -> None:
    global _PIPE
    if _PIPE is None:
        return
    try:
        _PIPE.close()
    except Exception:
        pass
    _PIPE = None
    _cuda_empty()


def _unload_ace() -> None:
    global _ACE
    if _ACE is None:
        return
    try:
        for name in ("model", "vae", "text_encoder", "text_tokenizer"):
            setattr(_ACE, name, None)
    except Exception:
        pass
    _ACE = None
    _cuda_empty()


def release_music_models() -> None:
    """Drop YuE2, ACE-Step and MOSS-SFX so speech can use the GPU."""
    _unload_yue2()
    _unload_ace()
    _unload_moss()


def _unload_moss() -> None:
    global _MOSS
    if _MOSS is None:
        return
    try:
        _MOSS.to("cpu")
    except Exception:
        pass
    _MOSS = None
    _cuda_empty()


def _yue2_graph_attention() -> str:
    """CUDA-graph decode. Official Linux FA kernel is missing from Win cu124 wheels."""
    return "sdpa"


def _patch_yue2_fast() -> None:
    global _YUE2_NAR_PATCHED
    if _YUE2_NAR_PATCHED:
        return
    from yue2 import cuda_graph

    attn = _yue2_graph_attention()
    orig_init = cuda_graph.GraphAR.__init__

    def __init__(self, model, prefixes, max_tokens, *, capture=True, attention_backend="auto", fuse_projections=False):
        if attention_backend == "auto":
            attention_backend = attn
        orig_init(
            self,
            model,
            prefixes,
            max_tokens,
            capture=capture,
            attention_backend=attention_backend,
            fuse_projections=fuse_projections,
        )

    cuda_graph.GraphAR.__init__ = __init__
    from yue2 import sampling

    orig_gen = sampling.generate_tokens

    def generate_tokens(*args: Any, **kwargs: Any):
        kwargs.setdefault("use_cuda_graph", True)
        try:
            return orig_gen(*args, **kwargs)
        except RuntimeError as exc:
            text = str(exc)
            if kwargs.get("use_cuda_graph") and (
                "FLASH" in text or "kernel" in text.lower() or "Aborting execution" in text
            ):
                kwargs["use_cuda_graph"] = False
                return orig_gen(*args, **kwargs)
            raise

    sampling.generate_tokens = generate_tokens
    from yue2 import nar

    orig_nar = nar.synthesize

    def synthesize(*args: Any, **kwargs: Any):
        import torch

        kwargs.setdefault("query_chunk_size", 256)
        kwargs.setdefault("offload_ar", True)
        _cuda_empty()
        try:
            return orig_nar(*args, **kwargs)
        except torch.cuda.OutOfMemoryError:
            _cuda_empty()
            kwargs["query_chunk_size"] = 128
            kwargs["offload_ar"] = True
            return orig_nar(*args, **kwargs)

    nar.synthesize = synthesize
    _YUE2_NAR_PATCHED = True


def _ensure_yue2(on_status) -> Any:
    global _PIPE
    with _PIPE_LOCK:
        if _PIPE is not None:
            _PIPE.offload_ar = True
            _relax_cuda_budget()
            return _PIPE
        _unload_ace()
        _unload_moss()
        on_status("load_model", 8, "Скачиваю/читаю YuE2-3B (CUDA-graph + FlashAttention)…")
        _allow_intercepted_ssl()
        cache_dir = str(_hub_cache_dir())
        os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
        os.environ["HF_HOME"] = str(Path(cache_dir).parent)
        from yue2 import YuE2Pipeline

        _patch_yue2_fast()
        t0 = time.perf_counter()
        pipe = YuE2Pipeline.from_pretrained(
            MODEL_REPO,
            vae=VAE_REPO,
            backend="torch",
            device="cuda",
            cache_dir=cache_dir,
            progress=True,
            memory_budget_gib=24,
            offload_ar=True,
        )
        _relax_cuda_budget()
        MUSIC.load_sec = round(time.perf_counter() - t0, 2)
        _PIPE = pipe
        return pipe


def _ensure_ace(on_status) -> Any:
    global _ACE
    with _PIPE_LOCK:
        if _ACE is not None:
            return _ACE
        if not ACE_SRC.is_dir():
            raise RuntimeError("Нет vendor/ACE-Step-1.5 — склонируйте ACE-Step 1.5 в vendor")
        _unload_yue2()
        _unload_moss()
        on_status("load_model", 8, "Скачиваю/читаю ACE-Step 1.5 turbo…")
        _allow_intercepted_ssl()
        src = str(ACE_SRC)
        if src not in sys.path:
            sys.path.insert(0, src)
        os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(ACE_CHECKPOINTS)
        os.environ.setdefault("ACESTEP_DOWNLOAD_SOURCE", "huggingface")
        os.environ.setdefault("ACESTEP_INIT_LLM", "false")
        os.environ.setdefault("ACESTEP_LM_BACKEND", "pt")
        ACE_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        from acestep.gpu_config import get_gpu_config, set_global_gpu_config
        from acestep.handler import AceStepHandler

        set_global_gpu_config(get_gpu_config())
        handler = AceStepHandler()
        t0 = time.perf_counter()
        status, ok = handler.initialize_service(
            project_root=str(ACE_SRC),
            config_path="acestep-v15-turbo",
            device="cuda",
            use_flash_attention=False,
            compile_model=False,
            offload_to_cpu=False,
            prefer_source="huggingface",
        )
        MUSIC.load_sec = round(time.perf_counter() - t0, 2)
        if not ok:
            raise RuntimeError(status)
        _ACE = handler
        return handler


def _ensure_moss(on_status) -> Any:
    global _MOSS
    with _PIPE_LOCK:
        if _MOSS is not None:
            return _MOSS
        if not (MOSS_SRC / "moss_soundeffect_v2").is_dir():
            raise RuntimeError("Нет vendor/MOSS-TTS/moss_soundeffect_v2")
        _unload_yue2()
        _unload_ace()
        on_status("load_model", 8, "Скачиваю/читаю MOSS-SoundEffect v2.0…")
        _allow_intercepted_ssl()
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        src = str(MOSS_SRC)
        if src not in sys.path:
            sys.path.insert(0, src)
        cache_dir = str(_hub_cache_dir())
        os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
        import torch
        from moss_soundeffect_v2 import MossSoundEffectPipeline

        t0 = time.perf_counter()
        pipe = MossSoundEffectPipeline.from_pretrained(
            MOSS_REPO,
            torch_dtype=torch.bfloat16,
            device="cuda",
            cache_dir=cache_dir,
        )
        MUSIC.load_sec = round(time.perf_counter() - t0, 2)
        _MOSS = pipe
        return pipe


def _save_moss_audio(audio: Any, sample_rate: int, dest: Path) -> float | None:
    import numpy as np
    import soundfile as sf

    arr = audio.detach().float().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 2 and arr.shape[0] <= 8:
        arr = np.transpose(arr, (1, 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dest, arr, int(sample_rate), subtype="PCM_24")
    return round(len(arr) / float(sample_rate), 2)


def _save_ace_audio(payload: dict[str, Any], dest: Path) -> float | None:
    import numpy as np
    import soundfile as sf

    audios = payload.get("audios") or []
    if not audios:
        raise RuntimeError(payload.get("error") or "ACE-Step не вернул аудио")
    item = audios[0]
    tensor = item["tensor"]
    sr = int(item.get("sample_rate") or 48000)
    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 2 and arr.shape[0] <= 8:
        arr = np.transpose(arr, (1, 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dest, arr, sr, subtype="PCM_24")
    return round(len(arr) / float(sr), 2)


def _probe_audio_sec(path: Path) -> float | None:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.samplerate:
            return round(info.frames / float(info.samplerate), 2)
    except Exception:
        return None
    return None


def _cover_caption(style: str, edit: str) -> str:
    style = (style or "").strip()
    edit = (edit or "").strip()
    if style and edit:
        return f"{style}. Changes: {edit}"
    if edit:
        return f"Keep the same arrangement and melody. Changes: {edit}"
    return style


def start_music(
    *,
    style: str,
    lyrics: str,
    cot: str,
    seed: int,
    gpu_busy: bool,
    acquire_gpu,
    release_gpu,
    model_id: str = YUE2_ID,
    duration_sec: float | None = None,
    source_rel: str | None = None,
    edit: str = "",
    lang: str = "en",
) -> dict[str, Any]:
    style = (style or "").strip()
    lyrics = (lyrics or "").strip()
    edit = (edit or "").strip()
    lang = "ru" if (lang or "").strip().lower() in ("ru", "rus", "russian") else "en"
    source_rel = (source_rel or "").strip() or None
    model_id = (model_id or YUE2_ID).strip()
    source_path: Path | None = None
    if source_rel:
        source_path = resolve_library_path(source_rel)
        if not source_path.is_file() or source_path.suffix.lower() not in AUDIO_EXTS:
            raise ValueError("Исходный трек не найден")
        model_id = ACESTEP_ID
    known = {m["id"] for m in MUSIC_MODELS}
    if model_id not in known:
        raise ValueError(f"Неизвестная модель: {model_id}")
    if source_path and not style and not edit:
        raise ValueError("Укажите стиль или что изменить в треке")
    if not style and not source_path:
        raise ValueError(
            "Укажите описание звука" if model_id == MOSS_ID else "Укажите стиль (genre, instruments, vocal)"
        )
    if model_id != MOSS_ID and not lyrics:
        lyrics = "" if source_path else "[Instrumental]"
    if model_id == YUE2_ID and cot not in COT_CHOICES:
        raise ValueError(f"cot: {COT_CHOICES}")
    if gpu_busy:
        raise RuntimeError("GPU занят очередью правки, сортировкой или генерацией картинок")
    with MUSIC._lock:
        if MUSIC.phase in ("load_model", "generate"):
            raise RuntimeError("Уже идёт генерация музыки")
        MUSIC.phase = "load_model"
        MUSIC.phase_started = time.monotonic()
        MUSIC.style = style
        MUSIC.lyrics = lyrics
        MUSIC.edit = edit
        MUSIC.source_rel = source_rel
        MUSIC.cot = cot if model_id == YUE2_ID else "off"
        MUSIC.seed = int(seed)
        MUSIC.lang = lang
        MUSIC.model_id = model_id
        MUSIC.error = None
        MUSIC.cancel = False
        MUSIC.progress = 2
        MUSIC.message = "Освобождаю GPU от фото-моделей…"
        MUSIC.result_path = None
        MUSIC.generate_sec = None
        MUSIC.audio_sec = None
        MUSIC.vram_gb = None
        MUSIC.timing = {}

    def worker() -> None:
        log = get_logger()
        acquired = False
        t_all = time.perf_counter()
        try:
            def aborted() -> bool:
                with MUSIC._lock:
                    return MUSIC.cancel

            def on_status(phase: str, progress: int, message: str) -> None:
                with MUSIC._lock:
                    MUSIC.phase = phase
                    MUSIC.progress = progress
                    MUSIC.message = message

            if aborted():
                raise RuntimeError("Отменено")
            from image_editor import release_gpu_vram

            on_status("load_model", 5, "Снимаю фото-модели с GPU…")
            release_gpu_vram()
            try:
                from speech import release_speech_gpu

                release_speech_gpu()
            except Exception:
                pass
            acquire_gpu()
            acquired = True
            if aborted():
                raise RuntimeError("Отменено")
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
            LIBRARY_ROOT.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            t_gen = time.perf_counter()
            if model_id == MOSS_ID:
                pipe = _ensure_moss(on_status)
                if aborted():
                    raise RuntimeError("Отменено")
                seconds = float(duration_sec) if duration_sec else 4.0
                seconds = max(0.5, min(30.0, seconds))
                on_status("generate", 25, f"MOSS-SoundEffect ~{seconds:.1f} с…")
                steps_total = 50

                def moss_bar(iterable=None, total=None, **_kw):
                    seq = list(iterable) if iterable is not None else list(range(int(total or steps_total)))
                    n = len(seq) or 1
                    for i, item in enumerate(seq, 1):
                        if aborted():
                            raise RuntimeError("Отменено")
                        on_status("generate", 25 + int(70 * i / n), f"MOSS-SoundEffect {i}/{n}")
                        yield item

                audio = pipe(
                    prompt=style,
                    seconds=seconds,
                    num_inference_steps=steps_total,
                    cfg_scale=4.0,
                    seed=int(seed),
                    progress_bar_cmd=moss_bar,
                )
                dest = LIBRARY_ROOT / f"{stamp}-moss.flac"
                audio_sec = _save_moss_audio(audio, getattr(pipe, "sample_rate", 48000), dest)
                timing = {"steps": steps_total, "seconds": seconds}
            elif model_id == ACESTEP_ID:
                handler = _ensure_ace(on_status)
                if aborted():
                    raise RuntimeError("Отменено")
                cover = source_path is not None
                if cover:
                    src_sec = _probe_audio_sec(source_path)
                    seconds = src_sec if src_sec else (float(duration_sec) if duration_sec else 30.0)
                    caption = _cover_caption(style, edit)
                    on_status("generate", 25, f"ACE-Step cover «{source_path.name}» ~{seconds:.0f} с…")
                else:
                    seconds = float(duration_sec) if duration_sec else 30.0
                    caption = style
                    on_status("generate", 25, f"ACE-Step 1.5 генерирует ~{seconds:.0f} с…")

                def ace_progress(ratio, desc=None, **_kwargs):
                    try:
                        pct = 25 + int(max(0.0, min(1.0, float(ratio))) * 70)
                    except Exception:
                        pct = 50
                    on_status("generate", pct, desc or "ACE-Step 1.5…")

                payload = handler.generate_music(
                    captions=caption,
                    lyrics=lyrics,
                    seed=int(seed),
                    use_random_seed=False,
                    inference_steps=8,
                    guidance_scale=1.0,
                    audio_duration=seconds,
                    batch_size=1,
                    vocal_language=lang,
                    dcw_enabled=False,
                    src_audio=str(source_path) if cover else None,
                    task_type="cover" if cover else "text2music",
                    audio_cover_strength=0.72 if cover else 1.0,
                    progress=ace_progress,
                )
                if not payload.get("success"):
                    raise RuntimeError(payload.get("error") or payload.get("status_message") or "ACE-Step ошибка")
                if cover:
                    dest = source_path.parent / f"{source_path.stem}-edit.flac"
                    if dest.exists() or dest.resolve() == source_path.resolve():
                        dest = source_path.parent / f"{source_path.stem}-edit-{stamp}.flac"
                else:
                    dest = LIBRARY_ROOT / f"{stamp}-acestep.flac"
                audio_sec = _save_ace_audio(payload, dest)
                extra = payload.get("extra_outputs") or {}
                timing = extra.get("time_costs") or {}
            else:
                pipe = _ensure_yue2(on_status)
                if aborted():
                    raise RuntimeError("Отменено")
                on_status("generate", 25, f"YuE2-3B CUDA-graph (cot={cot})…")
                _relax_cuda_budget()
                song = pipe(style=style, lyrics=lyrics, cot=cot, seed=int(seed), cancelled=aborted)
                dest = LIBRARY_ROOT / f"{stamp}-yue2.flac"
                song.save(dest)
                audio_sec = None
                try:
                    audio_sec = round(len(song.audio) / float(song.sample_rate), 2)
                except Exception:
                    pass
                timing = getattr(song, "timing", {}) or {}
            generate_sec = round(time.perf_counter() - t_gen, 2)
            vram = _peak_vram_gb()
            rtf = round(generate_sec / audio_sec, 3) if audio_sec else None
            row = {
                "ok": True,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "model_id": model_id,
                "cot": cot if model_id == YUE2_ID else None,
                "seed": int(seed),
                "style": style[:240],
                "lyrics_chars": len(lyrics),
                "load_sec": MUSIC.load_sec,
                "generate_sec": generate_sec,
                "total_sec": round(time.perf_counter() - t_all, 2),
                "audio_sec": audio_sec,
                "rtf": rtf,
                "vram_gb": vram,
                "path": str(dest),
                "timing": timing,
            }
            _append_stat(row)
            with MUSIC._lock:
                MUSIC.result_path = str(dest)
                MUSIC.generate_sec = generate_sec
                MUSIC.audio_sec = audio_sec
                MUSIC.vram_gb = vram
                MUSIC.timing = timing
                MUSIC.phase = "done"
                MUSIC.progress = 100
                MUSIC.message = (
                    f"Готово · генерация {generate_sec:.0f} с"
                    + (f" · трек {audio_sec:.0f} с" if audio_sec else "")
                )
        except Exception as exc:  # noqa: BLE001
            log.exception("music")
            _cuda_empty()
            with MUSIC._lock:
                if MUSIC.cancel:
                    MUSIC.phase = "idle"
                    MUSIC.message = "Отменено"
                    MUSIC.error = None
                else:
                    MUSIC.phase = "error"
                    MUSIC.error = str(exc)
                    MUSIC.message = f"Ошибка: {exc}"
                    _append_stat(
                        {
                            "ok": False,
                            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "model_id": model_id,
                            "cot": cot,
                            "error": str(exc),
                            "total_sec": round(time.perf_counter() - t_all, 2),
                        }
                    )
        finally:
            if acquired:
                try:
                    release_gpu()
                except Exception:
                    pass

    threading.Thread(target=worker, daemon=True, name="yue2-generate").start()
    return MUSIC.snapshot()


def request_music_cancel() -> dict[str, Any]:
    with MUSIC._lock:
        MUSIC.cancel = True
        MUSIC.message = "Остановка…"
    return MUSIC.snapshot()


def result_bytes() -> bytes | None:
    with MUSIC._lock:
        path = MUSIC.result_path
    if not path:
        return None
    file = Path(path)
    if not file.is_file():
        return None
    return file.read_bytes()


_MUSIC_LIB = AudioLibrary(LIBRARY_ROOT, crumb="Music", outside_label="Music")


def _library_root() -> Path:
    return _MUSIC_LIB.root_path()


def _library_rel(path: str | Path | None) -> str | None:
    return _MUSIC_LIB.rel(path)


def resolve_library_path(rel: str | None) -> Path:
    """Resolve a path inside the project Music folder. Empty = root."""
    return _MUSIC_LIB.resolve(rel)


def _retarget_music_result(src: Path, dest: Path | None) -> None:
    with MUSIC._lock:
        if MUSIC.result_path and Path(MUSIC.result_path).resolve() == src.resolve():
            MUSIC.result_path = None if dest is None else str(dest)


def list_library(rel: str | None = None) -> dict[str, Any]:
    return _MUSIC_LIB.list(rel)


def create_library_folder(rel: str | None, name: str) -> dict[str, Any]:
    return _MUSIC_LIB.mkdir(rel, name)


def rename_library_file(src_rel: str, stem: str, list_rel: str | None = None) -> dict[str, Any]:
    return _MUSIC_LIB.rename(src_rel, stem, list_rel, on_retarget=_retarget_music_result)


def delete_library_file(src_rel: str, list_rel: str | None = None) -> dict[str, Any]:
    return _MUSIC_LIB.delete(src_rel, list_rel, on_retarget=_retarget_music_result)


def move_library_item(src_rel: str, dest_dir_rel: str | None, list_rel: str | None = None) -> dict[str, Any]:
    return _MUSIC_LIB.move(src_rel, dest_dir_rel, list_rel, on_retarget=_retarget_music_result)


def library_file_bytes(rel: str) -> tuple[Path, str]:
    return _MUSIC_LIB.file(rel)

