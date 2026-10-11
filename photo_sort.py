"""Folder photo sort: JoyCaption JSONL → cluster → copy into subfolders."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from job_logger import LOG_DIR, get_logger

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SKIP_DIR_NAMES = {"sorted", ".photo-sort"}
CLUSTER_BATCH = 45
BUSY_PHASES = frozenset({"captioning", "clustering", "copying"})
WORK_TITLE = {
    "captioning": "JoyCaption подписывает фото",
    "clustering": "Группировка подписей",
    "copying": "Копирование в папки",
}


class PhotoSortStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.folder: Path | None = None
        self.recursive = False
        self.dest: Path | None = None
        self.files: list[dict[str, str]] = []
        self.captions: dict[str, str] = {}
        self.errors: dict[str, str] = {}
        self.failed: set[str] = set()
        self.groups: list[dict[str, Any]] = []
        self.phase = "idle"
        self.message = ""
        self.error: str | None = None
        self.current = ""
        self.jsonl_path: Path | None = None
        self.paused = False
        self.cancel = False
        self.copied = 0
        self.cluster_note: str | None = None
        self.phase_started = 0.0
        self._thread: threading.Thread | None = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            total = len(self.files)
            captioned = len(self.captions)
            recent = []
            for item in reversed(self.files):
                rel = item["rel"]
                if rel in self.captions:
                    recent.append({"rel": rel, "name": item["name"], "caption": self.captions[rel]})
                if len(recent) >= 12:
                    break
            busy = self.phase in BUSY_PHASES
            elapsed = int(time.monotonic() - self.phase_started) if busy and self.phase_started else 0
            return {
                "folder": str(self.folder) if self.folder else "",
                "recursive": self.recursive,
                "dest": str(self.dest) if self.dest else "",
                "total": total,
                "captioned": captioned,
                "pending": max(0, total - captioned),
                "phase": self.phase,
                "busy": busy,
                "elapsed_sec": elapsed,
                "work_title": WORK_TITLE.get(self.phase, ""),
                "message": self.message,
                "error": self.error,
                "current": self.current,
                "paused": self.paused,
                "jsonl": str(self.jsonl_path) if self.jsonl_path else "",
                "groups": self.groups,
                "recent": recent,
                "copied": self.copied,
                "failed": len(self.failed),
                "cluster_note": self.cluster_note,
            }


SORT = PhotoSortStore()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _session_id(folder: Path) -> str:
    return hashlib.sha256(str(folder).encode("utf-8")).hexdigest()[:16]


def _jsonl_path(folder: Path) -> Path:
    d = LOG_DIR / "sort"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{_session_id(folder)}.jsonl"


def resolve_folder(raw: str) -> Path:
    text = (raw or "").strip().strip('"')
    if not text:
        raise ValueError("Укажите путь к папке")
    path = Path(text).expanduser()
    try:
        path = path.resolve()
    except OSError as exc:
        raise ValueError(f"Некорректный путь: {exc}") from exc
    if not path.exists():
        raise ValueError(f"Папка не найдена: {path}")
    if not path.is_dir():
        raise ValueError(f"Это не папка: {path}")
    return path


def scan_images(folder: Path, *, recursive: bool) -> list[dict[str, str]]:
    found: list[Path] = []
    if recursive:
        for p in folder.rglob("*"):
            if not p.is_file():
                continue
            if any(part.lower() in SKIP_DIR_NAMES for part in p.relative_to(folder).parts[:-1]):
                continue
            if p.suffix.lower() in IMAGE_EXTS:
                found.append(p)
    else:
        for p in folder.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                found.append(p)
    found.sort(key=lambda x: str(x).lower())
    items = []
    for p in found:
        rel = p.relative_to(folder).as_posix()
        items.append({"path": str(p), "name": p.name, "rel": rel})
    return items


def _load_jsonl(path: Path) -> dict[str, str]:
    captions: dict[str, str] = {}
    if not path.is_file():
        return captions
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rel = str(row.get("rel") or "")
            cap = str(row.get("caption") or "").strip()
            if rel and cap:
                captions[rel] = cap
    return captions


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def begin_scan(folder_raw: str, recursive: bool) -> dict[str, Any]:
    folder = resolve_folder(folder_raw)
    with SORT._lock:
        if SORT.phase in ("captioning", "copying"):
            raise RuntimeError("Сначала дождитесь окончания текущего прогона или нажмите Стоп")
        files = scan_images(folder, recursive=recursive)
        jsonl = _jsonl_path(folder)
        captions = _load_jsonl(jsonl)
        SORT.folder = folder
        SORT.recursive = recursive
        SORT.dest = folder / "sorted"
        SORT.files = files
        SORT.captions = {k: v for k, v in captions.items() if any(f["rel"] == k for f in files)}
        SORT.errors = {}
        SORT.failed = set()
        SORT.groups = []
        SORT.phase = "scanned"
        SORT.message = f"Найдено {len(files)} фото, подписей в JSONL: {len(SORT.captions)}"
        SORT.error = None
        SORT.current = ""
        SORT.jsonl_path = jsonl
        SORT.paused = False
        SORT.cancel = False
        SORT.copied = 0
        SORT.cluster_note = None
    return SORT.snapshot()


def _pending_files() -> list[dict[str, str]]:
    with SORT._lock:
        done = set(SORT.captions)
        return [f for f in SORT.files if f["rel"] not in done]


def start_caption(gpu_busy: bool, acquire_gpu, release_gpu) -> dict[str, Any]:
    if gpu_busy:
        raise RuntimeError("GPU занят очередью генерации — дождитесь окончания или сбросьте GPU")
    with SORT._lock:
        if not SORT.folder or not SORT.files:
            raise RuntimeError("Сначала укажите папку и нажмите «Сканировать»")
        if SORT.phase == "captioning":
            raise RuntimeError("Подписи уже идут")
        pending = [f for f in SORT.files if f["rel"] not in SORT.captions]
        if not pending:
            SORT.phase = "captioned"
            SORT.message = "Все фото уже подписаны"
            return SORT.snapshot()
        SORT.phase = "captioning"
        SORT.phase_started = time.monotonic()
        SORT.paused = False
        SORT.cancel = False
        SORT.error = None
        SORT.failed = set()
        SORT.message = f"Подписи: 0 / {len(pending)}"

    def worker() -> None:
        log = get_logger()
        acquired = False
        try:
            acquire_gpu()
            acquired = True
            from image_editor import caption_image_for_sort, ensure_joycaption_loaded, unload_joycaption

            def aborted() -> bool:
                with SORT._lock:
                    return SORT.cancel

            with SORT._lock:
                SORT.message = "Загрузка JoyCaption…"
            ensure_joycaption_loaded(cancel_checker=aborted)
            while True:
                with SORT._lock:
                    if SORT.cancel:
                        SORT.phase = "paused" if SORT.captions else "scanned"
                        SORT.message = "Остановлено"
                        break
                    if SORT.paused:
                        SORT.phase = "paused"
                        SORT.message = "Пауза — JSONL сохранён, можно продолжить"
                        break
                    nxt = None
                    for item in SORT.files:
                        rel = item["rel"]
                        if rel not in SORT.captions and rel not in SORT.failed:
                            nxt = item
                            break
                    if nxt is None:
                        SORT.phase = "captioned"
                        SORT.message = f"Готово: {len(SORT.captions)} подписей"
                        break
                    SORT.current = nxt["name"]
                    done_n = len(SORT.captions)
                    total = len(SORT.files)
                    SORT.message = f"JoyCaption {done_n + 1} / {total}: {nxt['name']}"
                try:
                    data = Path(nxt["path"]).read_bytes()
                    caption = caption_image_for_sort(data, cancel_checker=aborted)
                    if aborted():
                        break
                    row = {
                        "path": nxt["path"],
                        "name": nxt["name"],
                        "rel": nxt["rel"],
                        "caption": caption,
                        "ts": _utc_now(),
                    }
                    with SORT._lock:
                        jsonl = SORT.jsonl_path
                    if jsonl:
                        _append_jsonl(jsonl, row)
                    with SORT._lock:
                        SORT.captions[nxt["rel"]] = caption
                except Exception as exc:  # noqa: BLE001
                    log.exception("sort caption %s", nxt["rel"])
                    with SORT._lock:
                        SORT.errors[nxt["rel"]] = str(exc)
                        SORT.failed.add(nxt["rel"])
                    if SORT.jsonl_path:
                        _append_jsonl(
                            SORT.jsonl_path,
                            {
                                "path": nxt["path"],
                                "name": nxt["name"],
                                "rel": nxt["rel"],
                                "caption": "",
                                "error": str(exc),
                                "ts": _utc_now(),
                            },
                        )
            unload_joycaption()
        except Exception as exc:  # noqa: BLE001
            log.exception("sort caption worker")
            with SORT._lock:
                SORT.phase = "error"
                SORT.error = str(exc)
                SORT.message = f"Ошибка: {exc}"
            try:
                from image_editor import unload_joycaption

                unload_joycaption()
            except Exception:
                pass
        finally:
            with SORT._lock:
                SORT.current = ""
                SORT._thread = None
            if acquired:
                try:
                    release_gpu()
                except Exception:
                    pass

    t = threading.Thread(target=worker, daemon=True, name="photo-sort-caption")
    with SORT._lock:
        SORT._thread = t
    t.start()
    return SORT.snapshot()


def request_pause() -> dict[str, Any]:
    with SORT._lock:
        if SORT.phase == "captioning":
            SORT.paused = True
            SORT.message = "Пауза после текущего кадра…"
        return SORT.snapshot()


def request_cancel() -> dict[str, Any]:
    with SORT._lock:
        SORT.cancel = True
        SORT.paused = True
        if SORT.phase == "captioning":
            SORT.message = "Остановка…"
        return SORT.snapshot()


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return (s[:40] if s else "unlabeled") or "unlabeled"


def _caption_named_groups(entries: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Folder names come from JoyCaption text only — no preset taxonomy."""
    buckets: dict[str, dict[str, Any]] = {}
    for rel, caption in entries:
        slug = _slug(caption)
        if slug not in buckets:
            title = (caption or slug).strip()[:80] or slug
            buckets[slug] = {"slug": slug, "title": title, "files": []}
        buckets[slug]["files"].append(rel)
    groups = list(buckets.values())
    groups.sort(key=lambda g: (-len(g["files"]), g["slug"]))
    return groups


def _parse_cluster_json(text: str) -> list[dict[str, Any]] | None:
    blob = text.strip()
    if "```" in blob:
        blob = re.sub(r"^```(?:json)?\s*|\s*```$", "", blob, flags=re.I | re.M)
    match = re.search(r"\{.*\}", blob, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    raw = data.get("groups") or data.get("folders") or []
    if not isinstance(raw, list):
        return None
    groups: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        slug = _slug(str(item.get("slug") or item.get("name") or item.get("title") or "other"))
        title = str(item.get("title") or item.get("name") or slug).strip() or slug
        files = item.get("files") or item.get("names") or item.get("rel") or []
        if not isinstance(files, list):
            continue
        rels = [str(x).replace("\\", "/") for x in files if str(x).strip()]
        if rels:
            groups.append({"slug": slug, "title": title, "files": rels})
    return groups or None


def _normalize_groups(groups: list[dict[str, Any]], known: set[str]) -> list[dict[str, Any]]:
    assigned: set[str] = set()
    cleaned: list[dict[str, Any]] = []
    seen_slug: dict[str, dict[str, Any]] = {}
    for g in groups:
        slug = _slug(g.get("slug") or "unlabeled")
        files = [f for f in g.get("files") or [] if f in known and f not in assigned]
        if not files:
            continue
        assigned.update(files)
        if slug in seen_slug:
            seen_slug[slug]["files"].extend(files)
            continue
        row = {"slug": slug, "title": g.get("title") or slug, "files": files}
        seen_slug[slug] = row
        cleaned.append(row)
    leftover = [r for r in known if r not in assigned]
    if leftover:
        if "unlabeled" in seen_slug:
            seen_slug["unlabeled"]["files"].extend(leftover)
        else:
            cleaned.append({"slug": "unlabeled", "title": "Unlabeled", "files": leftover})
    cleaned.sort(key=lambda g: (-len(g["files"]), g["slug"]))
    return cleaned


def _cluster_batches(entries: list[tuple[str, str]], known: set[str]) -> tuple[list[dict[str, Any]], str, str]:
    from prompt_helper import ollama_chat

    notes: list[str] = []
    collected: list[dict[str, Any]] = []
    batches = [entries[i : i + CLUSTER_BATCH] for i in range(0, len(entries), CLUSTER_BATCH)]
    total_b = max(1, len(batches))

    def progress(msg: str) -> None:
        with SORT._lock:
            SORT.message = msg

    system = (
        "You group files using only their captions. "
        "Invent folder names from what the captions say. "
        "Do not use any predefined category list (no people/animals/anime/furry/etc. unless the captions themselves say that). "
        "Do not invent extra files. JSON only, no markdown."
    )
    used = "captions"
    for idx, batch in enumerate(batches, 1):
        progress(f"Ollama: пакет {idx}/{total_b} ({len(batch)} подписей) — запрос…")

        def on_wait(tick: int, packet=idx) -> None:
            progress(f"Ollama думает: пакет {packet}/{total_b} · {tick * 3} с")

        lines = [f"{rel}\t{cap}" for rel, cap in batch]
        user = (
            "Group these files into folders named from the captions. "
            "Merge only when captions clearly describe the same kind of image. "
            "Do not force a target number of folders. "
            "Do not dump leftovers into a generic 'other' bucket.\n"
            "Return JSON only: {\"groups\":[{\"slug\":\"folder-name\",\"title\":\"Folder name\",\"files\":[\"a.jpg\"]}]}\n"
            "files must be the exact relative paths from the list.\n\n"
            + "\n".join(lines)
        )
        text, note = ollama_chat(
            system,
            user,
            model_id="qwen36-hauhau",
            temperature=0.15,
            timeout=120,
            on_wait=on_wait,
        )
        if note:
            notes.append(note)
        parsed = _parse_cluster_json(text) if text else None
        if parsed:
            used = "ollama"
            collected.extend(parsed)
        else:
            collected.extend(_caption_named_groups(batch))
            notes.append(f"пакет {idx}: без JSON — папки из текста JoyCaption")
        progress(f"Ollama: пакет {idx}/{total_b} готов")

    groups = _normalize_groups(collected, known)
    return groups, used, "; ".join(n for n in notes if n)


def run_cluster() -> dict[str, Any]:
    with SORT._lock:
        if SORT.phase in ("captioning", "copying", "clustering"):
            raise RuntimeError("Дождитесь окончания текущего шага")
        if not SORT.captions:
            raise RuntimeError("Нет подписей — сначала JoyCaption")
        entries = [(f["rel"], SORT.captions[f["rel"]]) for f in SORT.files if f["rel"] in SORT.captions]
        known = {rel for rel, _ in entries}
        SORT.phase = "clustering"
        SORT.phase_started = time.monotonic()
        SORT.message = f"Запуск группировки ({len(entries)} подписей)…"
        SORT.error = None

    def worker() -> None:
        log = get_logger()
        try:
            groups, method, note = _cluster_batches(entries, known)
            with SORT._lock:
                SORT.groups = groups
                SORT.cluster_note = note or None
                SORT.phase = "clustered"
                SORT.message = f"Готово: {len(groups)} папок ({method})"
        except Exception as exc:  # noqa: BLE001
            log.exception("sort cluster")
            with SORT._lock:
                SORT.phase = "error"
                SORT.error = str(exc)
                SORT.message = f"Ошибка группировки: {exc}"

    threading.Thread(target=worker, daemon=True, name="photo-sort-cluster").start()
    return SORT.snapshot()


def start_copy(dest_raw: str | None = None) -> dict[str, Any]:
    with SORT._lock:
        if not SORT.groups:
            raise RuntimeError("Сначала сгруппируйте фото")
        if SORT.phase == "copying":
            raise RuntimeError("Копирование уже идёт")
        folder = SORT.folder
        if folder is None:
            raise RuntimeError("Папка не выбрана")
        dest = Path(dest_raw).expanduser().resolve() if dest_raw else (SORT.dest or folder / "sorted")
        if dest.exists() and dest.is_file():
            raise RuntimeError("Путь назначения — файл, нужна папка")
        SORT.dest = dest
        SORT.phase = "copying"
        SORT.phase_started = time.monotonic()
        SORT.copied = 0
        SORT.error = None
        SORT.message = "Копирование…"
        groups = [dict(g) for g in SORT.groups]
        files_by_rel = {f["rel"]: f for f in SORT.files}

    def worker() -> None:
        log = get_logger()
        copied = 0
        report: list[dict[str, str]] = []
        try:
            dest.mkdir(parents=True, exist_ok=True)
            for g in groups:
                slug = _slug(g["slug"])
                target_dir = dest / slug
                target_dir.mkdir(parents=True, exist_ok=True)
                for rel in g["files"]:
                    src_info = files_by_rel.get(rel)
                    if not src_info:
                        continue
                    src = Path(src_info["path"])
                    if not src.is_file():
                        continue
                    out = target_dir / src.name
                    if out.exists():
                        stem, suf = out.stem, out.suffix
                        n = 2
                        while True:
                            cand = target_dir / f"{stem}_{n}{suf}"
                            if not cand.exists():
                                out = cand
                                break
                            n += 1
                    shutil.copy2(src, out)
                    copied += 1
                    report.append({"src": str(src), "dest": str(out), "slug": slug})
                    with SORT._lock:
                        SORT.copied = copied
                        SORT.current = src.name
                        SORT.message = f"Копирование {copied}"
            report_path = dest / "sort-report.json"
            report_path.write_text(
                json.dumps({"copied": copied, "groups": groups, "files": report}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            with SORT._lock:
                SORT.phase = "done"
                SORT.message = f"Скопировано {copied} файлов → {dest}"
                SORT.current = ""
        except Exception as exc:  # noqa: BLE001
            log.exception("sort copy")
            with SORT._lock:
                SORT.phase = "error"
                SORT.error = str(exc)
                SORT.message = f"Ошибка копирования: {exc}"

    threading.Thread(target=worker, daemon=True, name="photo-sort-copy").start()
    return SORT.snapshot()
