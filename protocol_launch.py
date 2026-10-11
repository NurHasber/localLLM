"""Тихий обработчик localphotoai:// — без окна cmd / python."""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
import winreg
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PANEL = "http://127.0.0.1:7860"
PROTOCOL = "localphotoai"
PYTHONW = ROOT / ".venv" / "Scripts" / "pythonw.exe"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)
if not PYTHONW.exists():
    cand = Path(sys.executable).with_name("pythonw.exe")
    PYTHONW = cand if cand.exists() else PYTHON


def panel_up() -> bool:
    try:
        with urllib.request.urlopen(f"{PANEL}/api/status", timeout=1.2) as resp:
            return resp.status == 200
    except Exception:
        return False


def _popen_hidden(args: list[str]) -> subprocess.Popen:
    creationflags = 0
    startupinfo = None
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(
            subprocess, "CREATE_NO_WINDOW", 0x08000000
        )
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE

    log = ROOT / "logs" / "protocol_launch.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log, "ab", buffering=0)
    return subprocess.Popen(
        args,
        cwd=str(ROOT),
        stdin=subprocess.DEVNULL,
        stdout=fh,
        stderr=fh,
        startupinfo=startupinfo,
        creationflags=creationflags,
        close_fds=False,
    )


def ensure_panel() -> bool:
    if panel_up():
        return True
    exe = PYTHONW if PYTHONW.exists() else PYTHON
    _popen_hidden([str(exe), "-u", str(ROOT / "tray_app.py")])
    for _ in range(60):
        time.sleep(0.35)
        if panel_up():
            return True
    return False


def register_protocol() -> None:
    if sys.platform != "win32":
        return
    handler = ROOT / "protocol_launch.py"
    icon = ROOT / "static" / "favicon.ico"
    exe = str(PYTHONW if PYTHONW.exists() else PYTHON)
    cmd = f'"{exe}" -u "{handler}" "%1"'
    key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"Software\Classes\{PROTOCOL}")
    winreg.SetValueEx(key, None, 0, winreg.REG_SZ, "URL:Photone Protocol")
    winreg.SetValueEx(key, "URL Protocol", 0, winreg.REG_SZ, "")
    icon_key = winreg.CreateKey(key, "DefaultIcon")
    winreg.SetValueEx(icon_key, None, 0, winreg.REG_SZ, str(icon) if icon.exists() else "")
    open_key = winreg.CreateKey(key, r"shell\open\command")
    winreg.SetValueEx(open_key, None, 0, winreg.REG_SZ, cmd)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    raw = " ".join(argv).strip().lower()
    if "register" in raw:
        register_protocol()
        return 0
    ok = ensure_panel()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
