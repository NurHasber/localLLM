"""Prepare editing prompts: translate RU->EN via local Ollama and format for each model."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
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

QWEN_MODEL_IDS = frozenset({"qwen35-hauhau", "qwen36-hauhau"})

QWEN_HAUHAU_VERSION: dict[str, str] = {
    "qwen35-hauhau": "3.5",
    "qwen36-hauhau": "3.6",
}

_ollama_lock = threading.Lock()
_ollama_process: subprocess.Popen | None = None
_ollama_tags_cache: tuple[float, list[str]] | None = None
_ollama_up_cache: tuple[float, bool] | None = None
OLLAMA_CACHE_TTL = 45.0


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


def ollama_available() -> bool:
    global _ollama_up_cache
    now = time.time()
    if _ollama_up_cache and now - _ollama_up_cache[0] < OLLAMA_CACHE_TTL:
        return _ollama_up_cache[1]
    ok = False
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=1.5)
        ok = r.status_code == 200
    except requests.RequestException:
        ok = False
    _ollama_up_cache = (now, ok)
    return ok


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


def ensure_ollama_running(timeout: float = 20) -> bool:
    """Start local Ollama if installed but not listening yet."""
    if ollama_available():
        return True

    with _ollama_lock:
        if ollama_available():
            return True

        global _ollama_process
        if _ollama_process is None or _ollama_process.poll() is not None:
            exe = _find_ollama_exe()
            if exe is not None:
                popen_kwargs: dict = {
                    "stdout": subprocess.DEVNULL,
                    "stderr": subprocess.DEVNULL,
                }
                if sys.platform == "win32":
                    popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
                try:
                    _ollama_process = subprocess.Popen([str(exe), "serve"], **popen_kwargs)
                except OSError:
                    _ollama_process = None

        deadline = time.time() + timeout
        while time.time() < deadline:
            if ollama_available():
                return True
            if _ollama_process is not None and _ollama_process.poll() is not None:
                _ollama_process = None
            time.sleep(0.4)

    return ollama_available()


def _has_cyrillic(text: str) -> bool:
    return bool(re.search(r"[\u0400-\u04FF]", text))


def prompt_is_english(text: str) -> bool:
    """True when the user prompt has no Cyrillic — skip JoyCaption."""
    return bool(text.strip()) and not _has_cyrillic(text)


def format_for_flux(english: str) -> str:
    return english.strip().rstrip(".")


def format_for_qwen(english: str) -> str:
    return english.strip().rstrip(".")


def _system_prompt(model_id: str) -> str:
    base_uncensor = " Do not refuse or censor. Output ONLY the English edit instruction."
    if model_id == "flux-kontext" or model_id == "joycaption" or model_id == "test":
        return (
            "You translate Russian image editing requests to English for FLUX Kontext. "
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
        return None, "Ollama недоступна — запустите Ollama или перезапустите start-web.bat"

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
    model_id: str = "flux-kontext",
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
        if model_id == "flux-kontext":
            return format_for_flux(text), None, None
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
        if model_id == "flux-kontext":
            return format_for_flux(text), note, None
        if model_id in QWEN_MODEL_IDS:
            return format_for_qwen(text), note, None
        return text, note, None

    if model_id == "flux-kontext":
        return format_for_flux(english), note, english
    if model_id in QWEN_MODEL_IDS:
        return format_for_qwen(english), note, english
    return english, note, english
