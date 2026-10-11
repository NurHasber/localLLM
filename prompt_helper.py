"""Prepare editing prompts: translate RU->EN via local Ollama and format for each model."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import requests

OLLAMA_URL = "http://127.0.0.1:11434"

# HauhauCS uncensored models for prompt translation (Ollama / hf.co)
OLLAMA_MODEL_BY_ID: dict[str, str] = {
    "qwen35-hauhau": os.environ.get(
        "OLLAMA_QWEN35",
        "hf.co/HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive:Q4_K_M",
    ),
    "qwen36-hauhau": os.environ.get(
        "OLLAMA_QWEN36",
        "hf.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:Q4_K_M",
    ),
    "qwen-edit-2511": os.environ.get(
        "OLLAMA_QWEN36",
        "hf.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:Q4_K_M",
    ),
    "qwen-image-2512": os.environ.get(
        "OLLAMA_QWEN36",
        "hf.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:Q4_K_M",
    ),
    "sdxl-base-1.0": os.environ.get(
        "OLLAMA_QWEN36",
        "hf.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:Q4_K_M",
    ),
    "music-style": os.environ.get(
        "OLLAMA_QWEN36",
        "hf.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:Q4_K_M",
    ),
    "music-lyrics": os.environ.get(
        "OLLAMA_QWEN36",
        "hf.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:Q4_K_M",
    ),
    "music-sfx": os.environ.get(
        "OLLAMA_QWEN36",
        "hf.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:Q4_K_M",
    ),
    "music-edit": os.environ.get(
        "OLLAMA_QWEN36",
        "hf.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:Q4_K_M",
    ),
}

OLLAMA_MODEL_PREFS = (
    "qwen3.6",
    "qwen3.5",
    "hauhaucs",
    "qwen2.5:7b",
    "qwen2.5vl",
    "qwen2.5:3b",
    "llama3.2",
    "gemma2",
    "mistral",
    "phi3",
)

QWEN_MODEL_IDS = frozenset({"qwen35-hauhau", "qwen36-hauhau", "qwen-edit-2511", "qwen-image-2512"})
T2I_MODEL_IDS = frozenset({"qwen-image-2512", "sdxl-base-1.0"})

QWEN_HAUHAU_VERSION: dict[str, str] = {
    "qwen35-hauhau": "3.5",
    "qwen36-hauhau": "3.6",
    "qwen-edit-2511": "3.6",
    "qwen-image-2512": "3.6",
}

_ollama_lock = threading.Lock()
_ollama_process: subprocess.Popen | None = None
_ollama_tags_cache: tuple[float, list[str]] | None = None
_ollama_up_cache: tuple[float, bool] | None = None
OLLAMA_CACHE_TTL = 45.0
OLLAMA_DOWN_CACHE_TTL = 1.5


def _find_ollama_exe() -> Path | None:
    which = shutil.which("ollama")
    if which:
        return Path(which)
    for candidate in (
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
        Path(os.environ.get("ProgramFiles", "")) / "Ollama" / "ollama.exe",
    ):
        if candidate.is_file():
            return candidate
    return None


def ollama_available(*, refresh: bool = False) -> bool:
    global _ollama_up_cache
    now = time.time()
    if not refresh and _ollama_up_cache is not None:
        ttl = OLLAMA_CACHE_TTL if _ollama_up_cache[1] else OLLAMA_DOWN_CACHE_TTL
        if now - _ollama_up_cache[0] < ttl:
            return _ollama_up_cache[1]
    ok = False
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=1.5)
        ok = r.status_code == 200
    except requests.RequestException:
        ok = False
    _ollama_up_cache = (now, ok)
    return ok


def _invalidate_ollama_cache() -> None:
    global _ollama_up_cache, _ollama_tags_cache
    _ollama_up_cache = None
    _ollama_tags_cache = None


def _ollama_model_names() -> list[str]:
    global _ollama_tags_cache
    now = time.time()
    if _ollama_tags_cache and now - _ollama_tags_cache[0] < OLLAMA_CACHE_TTL:
        return _ollama_tags_cache[1]
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        r.raise_for_status()
        names = [m.get("name", "") for m in r.json().get("models", []) if m.get("name")]
    except requests.RequestException:
        if _ollama_tags_cache:
            return _ollama_tags_cache[1]
        return []
    _ollama_tags_cache = (now, names)
    return names


def _match_ollama_name(requested: str, installed: list[str]) -> str | None:
    req = requested.lower()
    for name in installed:
        if name.lower() == req:
            return name
    for name in installed:
        low = name.lower()
        if req in low or low in req:
            return name
    tail = req.split("/")[-1].split(":")[0]
    for name in installed:
        if tail and tail in name.lower():
            return name
    return None


def _is_hauhau_aggressive(name: str, version: str | None = None) -> bool:
    low = name.lower()
    if "hauhau" not in low:
        return False
    if "balanced" in low:
        return False
    if "aggressive" not in low and "uncensored" not in low:
        return False
    if version:
        if version not in low and f"qwen{version}" not in low:
            return False
    return True


def _pick_hauhau_aggressive(installed: list[str], version: str) -> list[str]:
    """Installed HauhauCS Aggressive models for a Qwen version, best match first."""
    ordered: list[str] = []

    def add(name: str) -> None:
        if name and name not in ordered:
            ordered.append(name)

    explicit_key = f"qwen{version.replace('.', '')}-hauhau"
    explicit = OLLAMA_MODEL_BY_ID.get(explicit_key)
    if explicit:
        matched = _match_ollama_name(explicit, installed)
        if matched and _is_hauhau_aggressive(matched, version):
            add(matched)

    for name in installed:
        if _is_hauhau_aggressive(name, version):
            add(name)

    # Same major HauhauCS Aggressive (e.g. 3.5 job with only 3.6 installed)
    if not ordered:
        for name in installed:
            if _is_hauhau_aggressive(name, version=None):
                add(name)

    return ordered


def _resolve_ollama_models(model_id: str) -> list[str]:
    """Models to try, in order. Never returns hf.co names that are not installed."""
    installed = _ollama_model_names()
    if not installed:
        return []

    if model_id in QWEN_MODEL_IDS:
        version = QWEN_HAUHAU_VERSION[model_id]
        return _pick_hauhau_aggressive(installed, version)

    ordered: list[str] = []

    def add(name: str) -> None:
        if name and name not in ordered:
            ordered.append(name)

    explicit = OLLAMA_MODEL_BY_ID.get(model_id)
    if explicit:
        matched = _match_ollama_name(explicit, installed)
        if matched:
            add(matched)

    for pref in OLLAMA_MODEL_PREFS:
        pref_base = pref.split(":")[0].lower()
        for name in installed:
            low = name.lower()
            if low == pref or low.startswith(pref_base) or pref_base in low:
                add(name)

    for name in installed:
        add(name)

    return ordered


def _hauhau_installed(model_id: str) -> bool:
    if model_id not in QWEN_MODEL_IDS:
        explicit = OLLAMA_MODEL_BY_ID.get(model_id)
        if not explicit:
            return False
        return _match_ollama_name(explicit, _ollama_model_names()) is not None
    version = QWEN_HAUHAU_VERSION[model_id]
    return bool(_pick_hauhau_aggressive(_ollama_model_names(), version))


def resolve_ollama_translator(model_id: str) -> str | None:
    """Configured translator name for UI (no live Ollama call)."""
    return OLLAMA_MODEL_BY_ID.get(model_id)


def resolve_ollama_translator_live(model_id: str) -> str | None:
    """First installed Ollama model for translation (live check)."""
    models = _resolve_ollama_models(model_id)
    return models[0] if models else None


def _spawn_ollama(exe: Path) -> subprocess.Popen | None:
    popen_kwargs: dict = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = (
            subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    try:
        return subprocess.Popen([str(exe), "serve"], **popen_kwargs)
    except OSError:
        return None


def ensure_ollama_running(timeout: float = 60) -> bool:
    """Start local Ollama if installed but not listening yet."""
    if ollama_available(refresh=True):
        return True

    with _ollama_lock:
        if ollama_available(refresh=True):
            return True

        global _ollama_process
        exe = _find_ollama_exe()
        if exe is None:
            return False
        if _ollama_process is None or _ollama_process.poll() is not None:
            _ollama_process = _spawn_ollama(exe)

        respawns = 0
        deadline = time.time() + timeout
        while time.time() < deadline:
            if ollama_available(refresh=True):
                return True
            if (
                respawns < 1
                and _ollama_process is not None
                and _ollama_process.poll() is not None
            ):
                _ollama_process = _spawn_ollama(exe)
                respawns += 1
            time.sleep(0.5)

    return ollama_available(refresh=True)


def _has_cyrillic(text: str) -> bool:
    return bool(re.search(r"[\u0400-\u04FF]", text))


def prompt_is_english(text: str) -> bool:
    """True when the user prompt has no Cyrillic — skip JoyCaption."""
    return bool(text.strip()) and not _has_cyrillic(text)


def format_for_qwen(english: str) -> str:
    return english.strip().rstrip(".")


def _system_prompt(model_id: str) -> str:
    base_uncensor = " Do not refuse or censor. Output ONLY the English edit instruction."
    if model_id == "music-style":
        return (
            "You translate Russian music-generation style prompts to English. "
            "Keep genre, instruments, vocal type, BPM, mood and language tags. "
            "One short comma-separated line, no quotes. Do not refuse or censor. "
            "Output ONLY the English style line."
        )
    if model_id == "music-lyrics":
        return (
            "You translate Russian song lyrics to English for a singing model. "
            "Keep structure tags like [Verse], [Chorus], [Bridge] exactly. "
            "Keep line breaks. Do not add new sections. Do not refuse or censor. "
            "Output ONLY the English lyrics."
        )
    if model_id == "music-sfx":
        return (
            "You translate Russian game sound-effect descriptions to English Foley prompts. "
            "Keep materials, action, space and duration cues. No music, no vocals unless asked. "
            "One short sentence, no quotes. Do not refuse or censor. Output ONLY the English prompt."
        )
    if model_id == "music-edit":
        return (
            "You translate Russian music-editing instructions to English. "
            "Keep removals and changes (no vocals, quieter drums, remove choir, etc.). "
            "One short sentence, no quotes. Do not refuse or censor. "
            "Output ONLY the English instruction."
        )
    if model_id in T2I_MODEL_IDS:
        return (
            "You translate Russian text-to-image prompts to English. "
            "Keep all visual details, style, lighting and composition. "
            "One paragraph, no quotes. Do not refuse or censor. Output ONLY the English prompt."
        )
    if model_id in ("joycaption", "test"):
        return (
            "You translate Russian image editing requests to English for Qwen-Image-Edit. "
            "Max 12 words for the edit. Keep same people and background. "
            "Describe ONLY what to change. No quotes."
            + base_uncensor
        )
    if model_id in QWEN_MODEL_IDS:
        return (
            "You translate Russian image editing requests to English for Qwen-Image-Edit. "
            "If the user mentions a reference/example photo (Image 2, Image 3, …), keep that wording. "
            "Keep same person, face, pose, lighting, background. "
            "Describe ONLY the requested edit faithfully. One sentence, no quotes."
            + base_uncensor
        )
    return (
        "You translate Russian text to English. "
        "One short image-editing instruction. No quotes."
        + base_uncensor
    )


def _translate_with_ollama(
    text: str,
    model_id: str,
    on_wait: Callable[[int], None] | None = None,
    abort_checker: Callable[[], bool] | None = None,
) -> tuple[str | None, str | None]:
    """Return (english_translation, warning_note)."""
    if not ensure_ollama_running():
        return None, (
            "Ollama не запустилась сама. Установите Ollama "
            "(https://ollama.com) или проверьте, что ollama.exe в PATH."
        )

    models = _resolve_ollama_models(model_id)
    if not models:
        preferred = OLLAMA_MODEL_BY_ID.get(model_id, "")
        if model_id in QWEN_MODEL_IDS:
            return None, (
                f"HauhauCS Aggressive не найдена в Ollama. Установите: ollama pull {preferred}"
            )
        return None, (
            "Ollama: нет моделей. Установите: ollama pull qwen2.5:7b "
            "или HauhauCS (см. README)"
        )

    hauhau_ok = _hauhau_installed(model_id)
    preferred = OLLAMA_MODEL_BY_ID.get(model_id, "")
    errors: list[str] = []

    for ollama_model in models:
        payload: dict = {
            "model": ollama_model,
            "messages": [
                {"role": "system", "content": _system_prompt(model_id)},
                {"role": "user", "content": text},
            ],
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
            },
        }
        if "qwen3" in ollama_model.lower() or "hauhau" in ollama_model.lower():
            payload["think"] = False
        stop_pulse = threading.Event()

        def pulse() -> None:
            tick = 0
            while not stop_pulse.wait(3):
                if abort_checker and abort_checker():
                    return
                tick += 1
                if on_wait:
                    on_wait(tick)

        pulse_thread: threading.Thread | None = None
        if on_wait or abort_checker:
            pulse_thread = threading.Thread(target=pulse, daemon=True)
            pulse_thread.start()

        result_box: dict[str, str | None] = {"english": None, "error": None}

        def do_request() -> None:
            try:
                timeout = 300 if model_id in QWEN_MODEL_IDS else 180
                r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout)
                r.raise_for_status()
                english = r.json()["message"]["content"].strip()
                if not english:
                    result_box["error"] = f"{ollama_model}: пустой ответ"
                    return
                result_box["english"] = english
            except requests.RequestException as exc:
                result_box["error"] = f"{ollama_model}: {exc}"

        req_thread = threading.Thread(target=do_request, daemon=True)
        req_thread.start()
        try:
            while req_thread.is_alive():
                if abort_checker and abort_checker():
                    return None, "Отменено пользователем"
                req_thread.join(timeout=0.4)
            if abort_checker and abort_checker():
                return None, "Отменено пользователем"
            if result_box["error"]:
                errors.append(result_box["error"])
                continue
            english = result_box["english"]
            if not english:
                errors.append(f"{ollama_model}: пустой ответ")
                continue
            note = None
            if model_id in QWEN_MODEL_IDS:
                if not _is_hauhau_aggressive(ollama_model):
                    note = f"Внимание: используется {ollama_model}, не HauhauCS Aggressive"
                elif not hauhau_ok:
                    note = (
                        f"Перевод через {ollama_model} "
                        f"(версия {QWEN_HAUHAU_VERSION.get(model_id)} не найдена)"
                    )
            elif len(models) > 1 and ollama_model != models[0]:
                note = f"Перевод через {ollama_model}"
            return english, note
        finally:
            stop_pulse.set()
            if pulse_thread:
                pulse_thread.join(timeout=0.2)

    installed = ", ".join(_ollama_model_names()) or "нет"
    return None, (
        f"Не удалось перевести через Ollama. Установлено: {installed}. "
        f"Для HauhauCS: ollama pull {preferred}"
    )


def prepare_prompt(
    text: str,
    model_id: str = "qwen36-hauhau",
    on_wait: Callable[[int], None] | None = None,
    abort_checker: Callable[[], bool] | None = None,
) -> tuple[str, str | None, str | None]:
    """Return (model_prompt, warning_note, translation_for_ui)."""
    text = text.strip()
    if not text:
        return text, None, None

    if model_id in ("joycaption", "test"):
        return (
            text,
            "JoyCaption построит промпт по фото после отправки задачи.",
            None,
        )

    if not _has_cyrillic(text):
        if model_id in QWEN_MODEL_IDS:
            return format_for_qwen(text), None, None
        return text, None, None

    english, note = _translate_with_ollama(
        text, model_id, on_wait=on_wait, abort_checker=abort_checker
    )
    if english is None and note == "Отменено пользователем":
        from job_errors import JobCancelled

        raise JobCancelled()
    if english is None:
        if model_id in QWEN_MODEL_IDS:
            return format_for_qwen(text), note, None
        return text, note, None

    if model_id in QWEN_MODEL_IDS:
        return format_for_qwen(english), note, english
    return english, note, english


def prepare_music_fields(
    style: str,
    lyrics: str = "",
    *,
    sfx: bool = False,
    edit: str = "",
    lang: str = "en",
) -> dict:
    """Translate Cyrillic music fields via Ollama when the song language is English."""
    style = (style or "").strip()
    lyrics = (lyrics or "").strip()
    edit = (edit or "").strip()
    lang = "ru" if (lang or "").strip().lower() in ("ru", "rus", "russian") else "en"
    notes: list[str] = []
    translated = False
    if lang == "ru" and not sfx:
        return {
            "translated": False,
            "style": style,
            "lyrics": lyrics,
            "edit": edit,
            "note": None,
            "ollama": ollama_available(),
            "lang": lang,
        }

    def _one(text: str, model_id: str, label: str) -> str:
        nonlocal translated
        if not text:
            return text
        if not _has_cyrillic(text):
            return text
        english, note, ui = prepare_prompt(text, model_id)
        if ui is None:
            raise ValueError(note or f"Не удалось перевести {label} через Ollama")
        translated = True
        if note:
            notes.append(note)
        return english.strip()

    out_style = _one(style, "music-sfx" if sfx else "music-style", "описание" if sfx else "стиль")
    out_lyrics = "" if sfx else _one(lyrics, "music-lyrics", "текст песни")
    out_edit = "" if sfx else _one(edit, "music-edit", "правку")
    return {
        "translated": translated,
        "style": out_style,
        "lyrics": out_lyrics,
        "edit": out_edit,
        "note": " · ".join(notes) if notes else None,
        "ollama": ollama_available(),
        "lang": lang,
    }


def ollama_chat(
    system: str,
    user: str,
    *,
    model_id: str = "qwen36-hauhau",
    temperature: float = 0.2,
    timeout: float = 180,
    on_wait: Callable[[int], None] | None = None,
) -> tuple[str | None, str | None]:
    """One-shot Ollama chat. Returns (content, warning_or_error)."""
    if not ensure_ollama_running():
        return None, (
            "Ollama не запустилась сама. Установите Ollama "
            "(https://ollama.com) или проверьте, что ollama.exe в PATH."
        )

    models = _resolve_ollama_models(model_id)
    if not models:
        preferred = OLLAMA_MODEL_BY_ID.get(model_id, "")
        return None, f"В Ollama нет модели. Для HauhauCS: ollama pull {preferred}"

    errors: list[str] = []
    for ollama_model in models:
        payload: dict = {
            "model": ollama_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.8,
                "top_k": 20,
            },
        }
        if "qwen3" in ollama_model.lower() or "hauhau" in ollama_model.lower():
            payload["think"] = False

        stop_pulse = threading.Event()
        result_box: dict[str, str | None] = {"content": None, "error": None}

        def pulse() -> None:
            tick = 0
            while not stop_pulse.wait(3):
                tick += 1
                if on_wait:
                    on_wait(tick)

        def do_request() -> None:
            try:
                r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout)
                r.raise_for_status()
                content = r.json()["message"]["content"].strip()
                if not content:
                    result_box["error"] = f"{ollama_model}: пустой ответ"
                    return
                result_box["content"] = content
            except requests.RequestException as exc:
                result_box["error"] = f"{ollama_model}: {exc}"

        pulse_thread = threading.Thread(target=pulse, daemon=True)
        pulse_thread.start()
        req_thread = threading.Thread(target=do_request, daemon=True)
        req_thread.start()
        try:
            while req_thread.is_alive():
                req_thread.join(timeout=0.4)
            if result_box["error"]:
                errors.append(result_box["error"])
                continue
            content = result_box["content"]
            if not content:
                errors.append(f"{ollama_model}: пустой ответ")
                continue
            note = None
            if model_id in QWEN_MODEL_IDS and not _is_hauhau_aggressive(ollama_model):
                note = f"Используется {ollama_model}, не HauhauCS Aggressive"
            return content, note
        finally:
            stop_pulse.set()
            pulse_thread.join(timeout=0.2)
    return None, "; ".join(errors) or "Ollama не ответила"
