"""Local speech: Silero, Piper, Kokoro, MOSS-TTS Nano, MOSS-TTS Local v1.5."""

from __future__ import annotations

import gc
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import portable_env  # noqa: F401
from audio_library import AudioLibrary
from portable_env import ROOT

SPEECH_ROOT = ROOT / "Speech"
MODEL_ROOT = ROOT / "models" / "speech"
NANO_SRC = ROOT / "vendor" / "MOSS-TTS-Nano"
COSY_SRC = ROOT / "vendor" / "CosyVoice"
COSY_MODEL = COSY_SRC / "pretrained_models" / "Fun-CosyVoice3-0.5B"
COSY_PROMPT = COSY_SRC / "asset" / "zero_shot_prompt.wav"
NANO_DIR = MODEL_ROOT / "moss-nano"
KOKORO_ONNX = MODEL_ROOT / "kokoro" / "kokoro-v1.0.onnx"
KOKORO_VOICES = MODEL_ROOT / "kokoro" / "voices-v1.0.bin"
PIPER_ROOT = MODEL_ROOT / "piper"
SILERO_DIR = MODEL_ROOT / "silero"

MOSS_LOCAL_REPO = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
MOSS_CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"

SILERO_RU = ("aidar", "baya", "kseniya", "xenia", "eugene")
KOKORO_VOICES_FALLBACK = (
    ("af_heart", "en-us", "Heart, US"),
    ("af_bella", "en-us", "Bella, US"),
    ("af_sarah", "en-us", "Sarah, US"),
    ("af_nicole", "en-us", "Nicole, US"),
    ("am_adam", "en-us", "Adam, US"),
    ("am_michael", "en-us", "Michael, US"),
    ("am_echo", "en-us", "Echo, US"),
    ("bf_emma", "en-gb", "Emma, GB"),
    ("bf_alice", "en-gb", "Alice, GB"),
    ("bm_george", "en-gb", "George, GB"),
    ("bm_daniel", "en-gb", "Daniel, GB"),
)
PIPER_VOICES = (
    {
        "id": "en_US-lessac-high",
        "lang": "en",
        "label": "Lessac, US",
        "rel": "en/en_US/lessac/high/en_US-lessac-high.onnx",
    },
    {
        "id": "en_US-amy-medium",
        "lang": "en",
        "label": "Amy, US",
        "rel": "en/en_US/amy/medium/en_US-amy-medium.onnx",
    },
    {
        "id": "ru_RU-irina-medium",
        "lang": "ru",
        "label": "Ирина",
        "rel": "ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx",
    },
    {
        "id": "ru_RU-dmitri-medium",
        "lang": "ru",
        "label": "Дмитрий",
        "rel": "ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium.onnx",
    },
)

SPEECH_LIB = AudioLibrary(
    SPEECH_ROOT,
    crumb="Speech",
    outside_label="Speech",
    skip_prefixes=(".", "_"),
)


def speech_library_rel(path: str | Path | None) -> str | None:
    return SPEECH_LIB.rel(path)


def resolve_speech_library_path(rel: str | None) -> Path:
    return SPEECH_LIB.resolve(rel)


ENGINES = (
    {
        "id": "silero",
        "name": "Silero",
        "langs": ["ru", "en"],
        "clone": False,
        "gpu": False,
        "hint": "Лёгкий CPU. Русский v5 — ударения и «ё». Английский — отдельный пакет Silero.",
    },
    {
        "id": "piper",
        "name": "Piper",
        "langs": ["ru", "en"],
        "clone": False,
        "gpu": False,
        "hint": "ONNX, CPU, почти сразу. EN: Lessac и Amy. RU: Ирина и Дмитрий.",
    },
    {
        "id": "kokoro",
        "name": "Kokoro-82M",
        "langs": ["en"],
        "clone": False,
        "gpu": False,
        "hint": "Самый живой английский из лёгких. Русского нет — только en-US и en-GB.",
    },
    {
        "id": "moss-nano",
        "name": "MOSS-TTS Nano",
        "langs": ["ru", "en"],
        "clone": True,
        "gpu": False,
        "hint": "100M, ONNX, CPU, 48 кГц. Готовый голос или клон с короткого референса (3–10 с).",
    },
    {
        "id": "moss-local",
        "name": "MOSS-TTS Local v1.5",
        "langs": ["ru", "en"],
        "clone": True,
        "gpu": True,
        "hint": "4B на GPU, 48 кГц. Первый запуск качает ~16 ГБ. Не вместе с YuE2 и фото.",
    },
    {
        "id": "cosyvoice3",
        "name": "CosyVoice 3",
        "langs": ["ru", "en"],
        "clone": True,
        "gpu": True,
        "hint": "Fun-CosyVoice3 0.5B на GPU. Русский и английский, клон с референса. Без файла берётся встроенный пример. Не вместе с YuE2 и фото.",
    },
)

_lock = threading.Lock()
_silero: dict[str, Any] = {}
_piper: dict[str, Any] = {}
_kokoro: Any = None
_nano: Any = None
_moss_model: Any = None
_moss_processor: Any = None
_cosy: Any = None
_silero_speakers: dict[str, list[str]] = {}


class SpeechJob:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.phase = "idle"
        self.message = ""
        self.error: str | None = None
        self.progress = 0
        self.engine = "silero"
        self.lang = "ru"
        self.voice = ""
        self.text = ""
        self.result_path: str | None = None
        self.cancel = False
        self.phase_started = 0.0
        self.audio_sec: float | None = None

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
                "engine": self.engine,
                "lang": self.lang,
                "voice": self.voice,
                "text": self.text,
                "elapsed_sec": elapsed,
                "audio_sec": self.audio_sec,
                "has_result": bool(self.result_path and Path(self.result_path).is_file()),
                "result_url": "/api/speech/result" if self.result_path else None,
                "result_name": Path(self.result_path).name if self.result_path else None,
                "result_rel": speech_library_rel(self.result_path) if self.result_path else None,
                "library_root": str(SPEECH_ROOT),
                "engines": [dict(item) for item in ENGINES],
            }


SPEECH = SpeechJob()


def release_speech_gpu() -> None:
    """Unload GPU speech models so music or photos can take the card."""
    global _moss_model, _moss_processor, _cosy
    _moss_model = None
    _moss_processor = None
    _cosy = None
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
    with SPEECH._lock:
        return SPEECH.cancel


def _status(phase: str, progress: int, message: str) -> None:
    with SPEECH._lock:
        SPEECH.phase = phase
        SPEECH.progress = progress
        SPEECH.message = message


def _write_wav(audio: Any, sample_rate: int, dest: Path) -> float:
    import numpy as np
    import soundfile as sf

    arr = audio.detach().float().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 2 and arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1]:
        arr = np.transpose(arr, (1, 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dest, arr, int(sample_rate), subtype="PCM_16")
    frames = arr.shape[0] if arr.ndim else len(arr)
    return round(frames / float(sample_rate), 2)


def _silero_path(lang: str) -> Path:
    if lang == "ru":
        return SILERO_DIR / "v5_5_ru.pt"
    for name in ("v5_en.pt", "v3_en.pt"):
        path = SILERO_DIR / name
        if path.is_file() and path.stat().st_size > 100_000:
            return path
    return SILERO_DIR / "v3_en.pt"


def _load_silero(lang: str) -> Any:
    path = _silero_path(lang)
    if not path.is_file():
        raise FileNotFoundError(f"Нет весов Silero: {path.name}")
    cached = _silero.get(lang)
    if cached is not None and cached[0] == path:
        return cached[1]
    import torch

    torch.set_num_threads(4)
    model = torch.package.PackageImporter(str(path)).load_pickle("tts_models", "model")
    model.to(torch.device("cpu"))
    _silero[lang] = (path, model)
    speakers = getattr(model, "speakers", None)
    if isinstance(speakers, (list, tuple)) and speakers:
        _silero_speakers[lang] = [str(item) for item in speakers]
    return model


def _silero_voice_list(lang: str) -> list[dict[str, str]]:
    if lang not in _silero_speakers and _silero_path(lang).is_file():
        try:
            _load_silero(lang)
        except Exception:
            pass
    if lang in _silero_speakers:
        names = _silero_speakers[lang]
    elif lang == "ru":
        names = list(SILERO_RU)
    else:
        names = [f"en_{i}" for i in range(8)]
    if lang == "en" and len(names) > 8:
        names = names[:8]
    return [{"id": name, "lang": lang, "label": name} for name in names]


def _piper_voice_list() -> list[dict[str, str]]:
    rows = []
    for item in PIPER_VOICES:
        path = PIPER_ROOT / item["rel"]
        rows.append(
            {
                "id": item["id"],
                "lang": item["lang"],
                "label": item["label"] + ("" if path.is_file() else " (нет файла)"),
            }
        )
    return rows


def _kokoro_voice_list() -> list[dict[str, str]]:
    names: list[tuple[str, str, str]] = []
    if KOKORO_VOICES.is_file():
        try:
            import numpy as np

            blob = np.load(KOKORO_VOICES, allow_pickle=True)
            keys = list(blob.keys()) if hasattr(blob, "keys") else []
            for key in sorted(str(item) for item in keys):
                if key.startswith(("af_", "am_")):
                    accent = "en-us"
                elif key.startswith(("bf_", "bm_")):
                    accent = "en-gb"
                else:
                    continue
                names.append((key, accent, key))
        except Exception:
            names = []
    if not names:
        names = list(KOKORO_VOICES_FALLBACK)
    return [{"id": key, "lang": "en", "label": label, "accent": accent} for key, accent, label in names]


def _hf_relax() -> None:
    """Antivirus TLS intercept breaks the Xet CDN. Talk to Hugging Face without that check."""
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


def _nano_weights_ready() -> bool:
    folder = NANO_DIR / "MOSS-TTS-Nano-100M-ONNX"
    codec = NANO_DIR / "MOSS-Audio-Tokenizer-Nano-ONNX"
    if not folder.is_dir() or not codec.is_dir():
        return False

    def bulk(path: Path) -> int:
        return sum(item.stat().st_size for item in path.glob("*") if item.suffix in {".onnx", ".data"})

    return bulk(folder) > 50_000_000 and bulk(codec) > 5_000_000


def _nano_manifest() -> dict[str, Any] | None:
    for rel in (
        "browser_poc_manifest.json",
        "MOSS-TTS-Nano-100M-ONNX/browser_poc_manifest.json",
        "MOSS-TTS-Nano-ONNX-CPU/browser_poc_manifest.json",
    ):
        path = NANO_DIR / rel
        if path.is_file():
            import json

            return json.loads(path.read_text(encoding="utf-8"))
    return None


def _nano_voice_list() -> list[dict[str, str]]:
    manifest = _nano_manifest() or {}
    raw = manifest.get("builtin_voices") or []
    names: list[str] = []
    for item in raw:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            name = item.get("name") or item.get("id") or item.get("voice")
            if name:
                names.append(str(name))
    if not names:
        names = ["Junhao"]
    rows = []
    for name in names:
        rows.append({"id": name, "lang": "ru", "label": name})
        rows.append({"id": name, "lang": "en", "label": name})
    return rows


def _weights_ready(repo: str, filename: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        found = try_to_load_from_cache(repo, filename)
    except Exception:
        return False
    return isinstance(found, str) and Path(found).is_file()


def catalog() -> dict[str, Any]:
    engines = []
    for item in ENGINES:
        engine = dict(item)
        engine_id = item["id"]
        if engine_id == "silero":
            voices = _silero_voice_list("ru") + _silero_voice_list("en")
            ready = _silero_path("ru").is_file() or _silero_path("en").is_file()
        elif engine_id == "piper":
            voices = _piper_voice_list()
            ready = any((PIPER_ROOT / row["rel"]).is_file() for row in PIPER_VOICES)
        elif engine_id == "kokoro":
            voices = _kokoro_voice_list()
            ready = KOKORO_ONNX.is_file() and KOKORO_VOICES.is_file()
        elif engine_id == "moss-nano":
            voices = _nano_voice_list()
            ready = _nano_weights_ready()
        elif engine_id == "cosyvoice3":
            voices = [
                {"id": "builtin", "lang": "ru", "label": "встроенный пример"},
                {"id": "builtin", "lang": "en", "label": "встроенный пример"},
            ]
            ready = (COSY_MODEL / "cosyvoice3.yaml").is_file() and (COSY_MODEL / "llm.pt").is_file()
        else:
            voices = [
                {"id": "default", "lang": "ru", "label": "без референса"},
                {"id": "default", "lang": "en", "label": "без референса"},
            ]
            ready = _weights_ready(MOSS_LOCAL_REPO, "model.safetensors") and _weights_ready(
                MOSS_CODEC_REPO, "model-00001-of-00003.safetensors"
            )
        engine["voices"] = voices
        engine["ready"] = ready
        engines.append(engine)
    return {"engines": engines, "library_root": str(SPEECH_ROOT)}


def _synth_silero(text: str, lang: str, voice: str) -> tuple[Any, int]:
    model = _load_silero(lang)
    speakers = _silero_speakers.get(lang) or list(SILERO_RU if lang == "ru" else ["en_0"])
    speaker = voice if voice in speakers else speakers[0]
    kwargs: dict[str, Any] = {"text": text, "speaker": speaker, "sample_rate": 48000}
    if lang == "ru":
        kwargs["put_accent"] = True
        kwargs["put_yo"] = True
    audio = model.apply_tts(**kwargs)
    return audio, 48000


def _synth_piper(text: str, lang: str, voice: str) -> tuple[Any, int]:
    import numpy as np
    from piper import PiperVoice
    from piper.config import SynthesisConfig

    pool = [item for item in PIPER_VOICES if item["lang"] == lang] or list(PIPER_VOICES)
    chosen = next((item for item in pool if item["id"] == voice), pool[0])
    path = PIPER_ROOT / chosen["rel"]
    if not path.is_file():
        raise FileNotFoundError(f"Нет голоса Piper: {chosen['id']}")
    cached = _piper.get(chosen["id"])
    if cached is None:
        cached = PiperVoice.load(path)
        _piper[chosen["id"]] = cached
    chunks = [chunk.audio_float_array for chunk in cached.synthesize(text, SynthesisConfig())]
    if not chunks:
        raise RuntimeError("Piper не вернул звук")
    return np.concatenate(chunks), int(cached.config.sample_rate)


def _synth_kokoro(text: str, voice: str) -> tuple[Any, int]:
    global _kokoro
    if not KOKORO_ONNX.is_file() or not KOKORO_VOICES.is_file():
        raise FileNotFoundError("Нет весов Kokoro")
    if _kokoro is None:
        from kokoro_onnx import Kokoro

        _kokoro = Kokoro(str(KOKORO_ONNX), str(KOKORO_VOICES))
    known = _kokoro.get_voices()
    speaker = voice if voice in known else (known[0] if known else "af_heart")
    lang = "en-gb" if speaker.startswith(("bf_", "bm_")) else "en-us"
    audio, sample_rate = _kokoro.create(text, voice=speaker, speed=1.0, lang=lang)
    return audio, int(sample_rate)


def _load_nano() -> Any:
    global _nano
    if _nano is not None:
        return _nano
    if not NANO_SRC.is_dir():
        raise RuntimeError("Нет vendor/MOSS-TTS-Nano")
    _hf_relax()
    src = str(NANO_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    from onnx_tts_runtime import (  # type: ignore
        _download_default_browser_onnx_assets,
        _find_manifest_path,
    )
    from onnx_tts_runtime import OnnxTtsRuntime

    if _find_manifest_path(NANO_DIR) is None:
        _status("load", 12, "Скачиваю MOSS-TTS Nano ONNX…")
        _download_default_browser_onnx_assets(NANO_DIR)
    _nano = OnnxTtsRuntime(NANO_DIR, thread_count=4, execution_provider="cpu", output_dir=SPEECH_ROOT)
    return _nano


def _synth_nano(text: str, voice: str, reference: Path | None) -> tuple[Any, int]:
    runtime = _load_nano()
    payload = runtime.synthesize(
        text=text,
        voice=None if reference else (voice or None),
        prompt_audio_path=str(reference) if reference else None,
        sample_mode="fixed",
        do_sample=True,
        enable_wetext=False,
        enable_normalize_tts_text=False,
    )
    return payload["waveform"], int(payload["sample_rate"])


def _load_moss_local() -> tuple[Any, Any]:
    global _moss_model, _moss_processor
    if _moss_model is not None and _moss_processor is not None:
        return _moss_model, _moss_processor
    import torch
    from transformers import AutoModel, AutoProcessor

    _hf_relax()
    from image_editor import release_gpu_vram
    from music import release_music_models

    _status("load", 8, "Снимаю фото и музыку с GPU…")
    release_gpu_vram()
    release_music_models()
    _cuda_empty()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if device == "cuda":
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    _status("load", 18, "Читаю MOSS-TTS Local v1.5 (4B + токенайзер)…")
    processor = AutoProcessor.from_pretrained(MOSS_LOCAL_REPO, trust_remote_code=True)
    # The codec decode path casts activations to float32 and expects float weights.
    try:
        processor.audio_tokenizer = processor.audio_tokenizer.to(device=device, dtype=torch.float32)
    except Exception:
        processor.audio_tokenizer = processor.audio_tokenizer.to(device="cpu", dtype=torch.float32)
    model = AutoModel.from_pretrained(
        MOSS_LOCAL_REPO,
        trust_remote_code=True,
        attn_implementation="sdpa" if device == "cuda" else "eager",
        torch_dtype=dtype,
    )
    try:
        model = model.to(device)
    except torch.cuda.OutOfMemoryError:
        _cuda_empty()
        processor.audio_tokenizer = processor.audio_tokenizer.to(device="cpu")
        model = model.to(device)
    model.eval()
    _moss_model = model
    _moss_processor = processor
    return model, processor


def _synth_moss_local(text: str, lang: str, reference: Path | None) -> tuple[Any, int]:
    import torch

    model, processor = _load_moss_local()
    language = "Russian" if lang == "ru" else "English"
    kwargs: dict[str, Any] = {"text": text, "language": language}
    if reference is not None:
        kwargs["reference"] = [str(reference)]
    conversation = [processor.build_user_message(**kwargs)]
    device = next(model.parameters()).device
    batch = processor([conversation], mode="generation")
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    limit = min(4096, max(384, len(text) * 12))
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=limit,
            do_sample=True,
            audio_temperature=1.7,
            audio_top_p=0.8,
            audio_top_k=25,
            audio_repetition_penalty=1.0,
        )
    messages = list(processor.decode(outputs))
    if not messages or messages[0] is None or not getattr(messages[0], "audio_codes_list", None):
        raise RuntimeError("MOSS Local не вернул аудио")
    audio = messages[0].audio_codes_list[0]
    sample_rate = int(getattr(processor.model_config, "sampling_rate", 48000))
    return audio, sample_rate


def _load_cosy() -> Any:
    global _cosy
    if _cosy is not None:
        return _cosy
    if not (COSY_MODEL / "cosyvoice3.yaml").is_file():
        raise FileNotFoundError("Нет Fun-CosyVoice3 в vendor/CosyVoice")
    from image_editor import release_gpu_vram
    from music import release_music_models

    _status("load", 8, "Снимаю фото и музыку с GPU…")
    release_gpu_vram()
    release_music_models()
    release_speech_gpu()
    src = str(COSY_SRC)
    matcha = str(COSY_SRC / "third_party" / "Matcha-TTS")
    for path in (src, matcha):
        if path not in sys.path:
            sys.path.insert(0, path)
    _status("load", 18, "Читаю CosyVoice 3…")
    from cosyvoice.cli.cosyvoice import AutoModel

    _cosy = AutoModel(model_dir=str(COSY_MODEL), fp16=True)
    return _cosy


def _synth_cosy(text: str, lang: str, reference: Path | None) -> tuple[Any, int]:
    import torch

    model = _load_cosy()
    prompt = reference if reference is not None and reference.is_file() else COSY_PROMPT
    if not prompt.is_file():
        raise FileNotFoundError("Нет референса CosyVoice и встроенного примера")
    style = "请用俄语表达。" if lang == "ru" else "请用英语表达。"
    instruct = f"You are a helpful assistant. {style}<|endofprompt|>"
    pieces = []
    for chunk in model.inference_instruct2(
        text, instruct, str(prompt), stream=False, speed=1.0, text_frontend=False
    ):
        speech = chunk.get("tts_speech")
        if speech is not None:
            pieces.append(speech)
    if not pieces:
        raise RuntimeError("CosyVoice не вернул звук")
    audio = torch.cat(pieces, dim=1)
    return audio, int(model.sample_rate)


def start_speech(
    *,
    engine: str,
    lang: str,
    voice: str,
    text: str,
    reference: Path | None,
    gpu_busy: bool,
    acquire_gpu,
    release_gpu,
) -> dict[str, Any]:
    known = {item["id"] for item in ENGINES}
    engine = (engine or "").strip()
    if engine not in known:
        raise ValueError(f"Неизвестный синтезатор: {engine}")
    spec = next(item for item in ENGINES if item["id"] == engine)
    lang = "en" if (lang or "").strip().lower() in ("en", "eng", "english") else "ru"
    if lang not in spec["langs"]:
        lang = spec["langs"][0]
    text = (text or "").strip()
    if not text:
        raise ValueError("Введите текст")
    if len(text) > 4000:
        raise ValueError("Текст длиннее 4000 символов — укоротите для теста")
    if spec["gpu"] and gpu_busy:
        raise RuntimeError("GPU занят очередью, сортировкой, картинкой или музыкой")
    with SPEECH._lock:
        if SPEECH.phase in ("load", "generate"):
            raise RuntimeError("Уже идёт синтез речи")
        SPEECH.phase = "load"
        SPEECH.phase_started = time.monotonic()
        SPEECH.engine = engine
        SPEECH.lang = lang
        SPEECH.voice = voice
        SPEECH.text = text
        SPEECH.error = None
        SPEECH.cancel = False
        SPEECH.progress = 2
        SPEECH.message = "Запуск…"
        SPEECH.result_path = None
        SPEECH.audio_sec = None

    def worker() -> None:
        acquired = False
        try:
            if _aborted():
                raise RuntimeError("Отменено")
            if spec["gpu"] and acquire_gpu is not None:
                acquire_gpu()
                acquired = True
            _status("generate", 30, f"{spec['name']}…")
            if engine == "silero":
                audio, sample_rate = _synth_silero(text, lang, voice)
            elif engine == "piper":
                audio, sample_rate = _synth_piper(text, lang, voice)
            elif engine == "kokoro":
                audio, sample_rate = _synth_kokoro(text, voice)
            elif engine == "moss-nano":
                audio, sample_rate = _synth_nano(text, voice, reference)
            elif engine == "cosyvoice3":
                audio, sample_rate = _synth_cosy(text, lang, reference)
            else:
                audio, sample_rate = _synth_moss_local(text, lang, reference)
            if _aborted():
                raise RuntimeError("Отменено")
            stamp = time.strftime("%Y%m%d-%H%M%S")
            slug = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in (voice or "voice"))[:40]
            dest = SPEECH_ROOT / f"{stamp}-{engine}-{lang}-{slug}.wav"
            _status("generate", 90, "Сохраняю wav…")
            audio_sec = _write_wav(audio, sample_rate, dest)
            with SPEECH._lock:
                SPEECH.result_path = str(dest)
                SPEECH.audio_sec = audio_sec
                SPEECH.phase = "done"
                SPEECH.progress = 100
                SPEECH.message = f"Готово · {audio_sec:.1f} с · {dest.name}"
        except Exception as exc:  # noqa: BLE001
            _cuda_empty()
            with SPEECH._lock:
                if SPEECH.cancel:
                    SPEECH.phase = "idle"
                    SPEECH.message = "Отменено"
                    SPEECH.error = None
                else:
                    SPEECH.phase = "error"
                    SPEECH.error = str(exc)
                    SPEECH.message = f"Ошибка: {exc}"
        finally:
            if acquired and release_gpu is not None:
                try:
                    release_gpu()
                except Exception:
                    pass

    threading.Thread(target=worker, daemon=True, name="speech-tts").start()
    return SPEECH.snapshot()


def request_speech_cancel() -> dict[str, Any]:
    with SPEECH._lock:
        SPEECH.cancel = True
        SPEECH.message = "Остановка…"
    return SPEECH.snapshot()


def result_file() -> Path | None:
    with SPEECH._lock:
        path = SPEECH.result_path
    if not path:
        return None
    file = Path(path)
    return file if file.is_file() else None


def _retarget_speech_result(src: Path, dest: Path | None) -> None:
    with SPEECH._lock:
        if SPEECH.result_path and Path(SPEECH.result_path).resolve() == src.resolve():
            SPEECH.result_path = None if dest is None else str(dest)


def list_speech_library(rel: str | None = None) -> dict[str, Any]:
    return SPEECH_LIB.list(rel)


def create_speech_folder(rel: str | None, name: str) -> dict[str, Any]:
    return SPEECH_LIB.mkdir(rel, name)


def rename_speech_file(src_rel: str, stem: str, list_rel: str | None = None) -> dict[str, Any]:
    return SPEECH_LIB.rename(src_rel, stem, list_rel, on_retarget=_retarget_speech_result)


def delete_speech_file(src_rel: str, list_rel: str | None = None) -> dict[str, Any]:
    return SPEECH_LIB.delete(src_rel, list_rel, on_retarget=_retarget_speech_result)


def move_speech_item(src_rel: str, dest_dir_rel: str | None, list_rel: str | None = None) -> dict[str, Any]:
    return SPEECH_LIB.move(src_rel, dest_dir_rel, list_rel, on_retarget=_retarget_speech_result)


def speech_file_bytes(rel: str) -> tuple[Path, str]:
    return SPEECH_LIB.file(rel)


def save_speech_recording(raw: bytes) -> dict[str, Any]:
    """Store a microphone take as PCM wav in the project Speech folder."""
    import io

    import numpy as np
    import soundfile as sf

    if not raw or len(raw) < 44:
        raise ValueError("Пустая запись")
    if len(raw) > 1_000_000_000:
        raise ValueError("Запись больше 1 ГБ")
    try:
        audio, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception as exc:
        raise ValueError("Нужен файл wav") from exc
    sample_rate = int(sample_rate)
    if sample_rate < 8000 or sample_rate > 96000:
        raise ValueError("Неподдерживаемая частота записи")
    frames = int(audio.shape[0])
    seconds = frames / float(sample_rate)
    if seconds < 0.4:
        raise ValueError("Запись короче 0,4 с")
    mono = np.asarray(audio, dtype=np.float32).mean(axis=1)
    root = SPEECH_LIB.root_path()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = root / f"{stamp}-mic.wav"
    n = 2
    while dest.exists():
        dest = root / f"{stamp}-mic-{n}.wav"
        n += 1
    sf.write(dest, mono, sample_rate, subtype="PCM_16")
    listing = SPEECH_LIB.list("")
    rel = dest.relative_to(root).as_posix()
    listing["saved"] = rel
    listing["name"] = dest.name
    listing["audio_sec"] = round(seconds, 2)
    return listing
