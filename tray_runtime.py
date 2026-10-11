"""Иконка Photone в трее (pystray)."""

from __future__ import annotations

import os
import sys
import webbrowser
from pathlib import Path

from brand import APP_NAME

ROOT = Path(__file__).resolve().parent
ICON_PATH = ROOT / "static" / "favicon.ico"
PANEL_URL = "http://127.0.0.1:7860/"


def _icon_image():
    from PIL import Image, ImageDraw

    if ICON_PATH.is_file():
        img = Image.open(ICON_PATH)
        return img.convert("RGBA")
    image = Image.new("RGBA", (64, 64), (26, 29, 39, 255))
    draw = ImageDraw.Draw(image)
    fill = (108, 140, 255, 255)
    draw.rounded_rectangle((8, 20, 56, 52), radius=6, outline=fill, width=4)
    draw.ellipse((22, 26, 42, 46), outline=(138, 164, 255, 255), width=4)
    draw.ellipse((44, 24, 50, 30), fill=fill)
    return image


def run_tray_blocking(*, on_quit=None) -> None:
    """Блокирует поток (на Windows — главный). ЛКМ / пункт меню открывает сайт."""
    import pystray

    def open_page(icon=None, item=None) -> None:  # noqa: ARG001
        webbrowser.open(PANEL_URL)

    def quit_app(icon, item=None) -> None:  # noqa: ARG001
        try:
            icon.visible = False
        except Exception:
            pass
        try:
            icon.stop()
        except Exception:
            pass
        if on_quit:
            try:
                on_quit()
            except Exception:
                pass
        os._exit(0)

    menu = pystray.Menu(
        pystray.MenuItem(f"Открыть {APP_NAME}", open_page, default=True),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(f"Закрыть {APP_NAME}", quit_app),
    )
    def setup(icon) -> None:
        icon.visible = True
        try:
            icon.notify("Приложение в трее (стрелка ^ рядом с часами, если иконка спрятана).", APP_NAME)
        except Exception:
            pass

    icon = pystray.Icon(
        name="Photone",
        icon=_icon_image(),
        title=APP_NAME,
        menu=menu,
    )
    icon.run(setup=setup)


if __name__ == "__main__":
    run_tray_blocking()
