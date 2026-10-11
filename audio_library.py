"""Folder browser for local audio libraries (music and speech)."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Callable

AUDIO_EXTS = {".flac", ".wav", ".mp3", ".ogg", ".m4a", ".aac"}
_NAME_BAD = frozenset('<>:"/\\|?*')
AUDIO_MIME = {
    ".flac": "audio/flac",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
}

Retarget = Callable[[Path, Path | None], None]


class AudioLibrary:
    def __init__(
        self,
        root: Path,
        *,
        crumb: str,
        outside_label: str,
        skip_prefixes: tuple[str, ...] = (".",),
    ) -> None:
        self._root = root
        self.crumb = crumb
        self.outside_label = outside_label
        self.skip_prefixes = skip_prefixes

    def root_path(self) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        return self._root.resolve()

    def rel(self, path: str | Path | None) -> str | None:
        if not path:
            return None
        try:
            rel = Path(path).resolve().relative_to(self.root_path())
        except ValueError:
            return None
        return rel.as_posix()

    def resolve(self, rel: str | None) -> Path:
        root = self.root_path()
        text = (rel or "").replace("\\", "/").strip()
        if text in ("", ".", "/"):
            return root
        if text.startswith("/"):
            text = text[1:]
        parts = [part for part in Path(text).parts if part not in ("", ".")]
        if any(part == ".." for part in parts):
            raise ValueError("Нельзя выйти выше библиотеки")
        target = root.joinpath(*parts).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"Путь вне {self.outside_label}")
        return target

    def _hidden(self, name: str) -> bool:
        return name.startswith(self.skip_prefixes)

    def list(self, rel: str | None = None) -> dict[str, Any]:
        folder = self.resolve(rel)
        if not folder.is_dir():
            raise FileNotFoundError("Папка не найдена")
        root = self.root_path()
        rel_posix = "" if folder == root else folder.relative_to(root).as_posix()
        crumbs = [{"name": self.crumb, "path": ""}]
        if rel_posix:
            acc: list[str] = []
            for part in rel_posix.split("/"):
                acc.append(part)
                crumbs.append({"name": part, "path": "/".join(acc)})
        parent = None
        if folder != root:
            parent_path = folder.parent
            parent = "" if parent_path == root else parent_path.relative_to(root).as_posix()
        dirs: list[dict[str, Any]] = []
        files: list[dict[str, Any]] = []
        for item in sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if self._hidden(item.name):
                continue
            item_rel = item.relative_to(root).as_posix()
            try:
                st = item.stat()
            except OSError:
                continue
            times = {"ctime": int(st.st_ctime), "mtime": int(st.st_mtime)}
            if item.is_dir():
                dirs.append({"name": item.name, "path": item_rel, "kind": "dir", **times})
                continue
            if item.suffix.lower() not in AUDIO_EXTS:
                continue
            files.append(
                {
                    "name": item.name,
                    "path": item_rel,
                    "kind": "file",
                    "size": st.st_size,
                    **times,
                }
            )
        return {
            "root": str(root),
            "path": rel_posix,
            "parent": parent,
            "crumbs": crumbs,
            "dirs": dirs,
            "files": files,
        }

    def mkdir(self, rel: str | None, name: str) -> dict[str, Any]:
        name = (name or "").strip()
        if not name or name in (".", "..") or "/" in name or "\\" in name:
            raise ValueError("Некорректное имя папки")
        if self._hidden(name):
            raise ValueError("Некорректное имя папки")
        folder = self.resolve(rel) / name
        if folder.exists():
            raise FileExistsError(f"Папка «{name}» уже есть")
        folder.mkdir(parents=False, exist_ok=False)
        return self.list(rel)

    def rename(
        self,
        src_rel: str,
        stem: str,
        list_rel: str | None = None,
        on_retarget: Retarget | None = None,
    ) -> dict[str, Any]:
        src = self.resolve(src_rel)
        if not src.is_file() or src.suffix.lower() not in AUDIO_EXTS:
            raise FileNotFoundError("Аудиофайл не найден")
        stem = _clean_file_stem(stem, src.suffix)
        dest = src.with_name(stem + src.suffix)
        stay = src.parent.relative_to(self.root_path()).as_posix()
        if stay == ".":
            stay = ""
        if dest.resolve() == src.resolve():
            return self.list(list_rel if list_rel is not None else stay)
        if dest.exists():
            raise FileExistsError(f"Уже есть файл «{dest.name}»")
        _retry_locked(lambda: src.rename(dest))
        if on_retarget is not None:
            on_retarget(src, dest)
        stay = dest.parent.relative_to(self.root_path()).as_posix()
        if stay == ".":
            stay = ""
        return self.list(list_rel if list_rel is not None else stay)

    def delete(
        self,
        src_rel: str,
        list_rel: str | None = None,
        on_retarget: Retarget | None = None,
    ) -> dict[str, Any]:
        src = self.resolve(src_rel)
        if not src.is_file() or src.suffix.lower() not in AUDIO_EXTS:
            raise FileNotFoundError("Аудиофайл не найден")
        _retry_locked(src.unlink)
        if on_retarget is not None:
            on_retarget(src, None)
        stay = src.parent.relative_to(self.root_path()).as_posix()
        if stay == ".":
            stay = ""
        return self.list(list_rel if list_rel is not None else stay)

    def move(
        self,
        src_rel: str,
        dest_dir_rel: str | None,
        list_rel: str | None = None,
        on_retarget: Retarget | None = None,
    ) -> dict[str, Any]:
        src = self.resolve(src_rel)
        dest_dir = self.resolve(dest_dir_rel)
        if not src.exists():
            raise FileNotFoundError("Файл не найден")
        if not dest_dir.is_dir():
            raise FileNotFoundError("Папка назначения не найдена")
        if src.parent.resolve() == dest_dir.resolve():
            return self.list(dest_dir_rel)
        dest = dest_dir / src.name
        if dest.exists():
            raise FileExistsError(f"Уже есть файл с именем {src.name}")
        shutil.move(str(src), str(dest))
        if on_retarget is not None:
            on_retarget(src, dest)
        stay = dest_dir_rel if list_rel is None else list_rel
        return self.list(stay)

    def file(self, rel: str) -> tuple[Path, str]:
        path = self.resolve(rel)
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
            raise FileNotFoundError("Аудиофайл не найден")
        mime = AUDIO_MIME.get(path.suffix.lower(), "application/octet-stream")
        return path, mime


def _clean_file_stem(stem: str, suffix: str) -> str:
    stem = (stem or "").strip()
    if suffix and stem.lower().endswith(suffix.lower()):
        stem = stem[: -len(suffix)].rstrip()
    if not stem or stem in (".", "..") or stem.endswith(".") or stem.endswith(" "):
        raise ValueError("Некорректное имя файла")
    if any(ch in _NAME_BAD or ord(ch) < 32 for ch in stem):
        raise ValueError('Имя не должно содержать \\ / : * ? " < > |')
    return stem


def _retry_locked(action) -> None:
    last_error: OSError | None = None
    for attempt in range(12):
        try:
            action()
            return
        except OSError as exc:
            locked = getattr(exc, "winerror", None) == 32 or exc.errno in (13, 16, 32)
            if not locked or attempt == 11:
                raise
            last_error = exc
            time.sleep(0.15)
    if last_error is not None:
        raise last_error
