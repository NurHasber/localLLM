"""Photone: pythonw-вход в трей без окна cmd."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import portable_env  # noqa: E402

from protocol_launch import register_protocol  # noqa: E402
from win_tray import APP_USER_MODEL_ID, ensure_icon_file, set_app_id  # noqa: E402

LOG_PATH = ROOT / "logs" / "photo-edit.log"


def _redirect_logs() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        fh = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
        sys.stdout = fh  # type: ignore[assignment]
        sys.stderr = fh  # type: ignore[assignment]
    except OSError:
        pass
    logging.basicConfig(
        filename=str(LOG_PATH),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )


def main() -> int:
    if sys.platform != "win32":
        print("Tray только для Windows", file=sys.stderr)
        return 1
    set_app_id(APP_USER_MODEL_ID)
    ensure_icon_file()
    register_protocol()
    _redirect_logs()
    from web_app import main as run_web

    run_web()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
