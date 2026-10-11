"""Общая Windows-иконка в трее: ЛКМ — открыть URL, ПКМ — меню."""

from __future__ import annotations

import ctypes
import subprocess
import sys
import webbrowser
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_ICON = ROOT / "static" / "favicon.ico"
ICON_DIR = ROOT / "static"
ICON_COLORS = {
    "panel": (108, 140, 255, 255),
}
PANEL_URL = "http://127.0.0.1:7860/"
APP_USER_MODEL_ID = "Photone"

user32 = ctypes.windll.user32
shell32 = ctypes.windll.shell32
# use_last_error: иначе GetLastError после CreateMutex врёт и второй экземпляр не отсекается
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
kernel32.CreateMutexW.restype = wintypes.HANDLE
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.CloseHandle.restype = wintypes.BOOL
kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.OpenProcess.restype = wintypes.HANDLE

LRESULT = ctypes.c_ssize_t
WPARAM = ctypes.c_size_t
LPARAM = ctypes.c_ssize_t

user32.DefWindowProcW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
user32.DefWindowProcW.restype = ctypes.c_ssize_t
user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
user32.GetMessageW.restype = ctypes.c_int

WM_DESTROY = 0x0002
WM_COMMAND = 0x0111
WM_USER = 0x0400
WM_LBUTTONUP = 0x0202
WM_RBUTTONUP = 0x0205
WM_LBUTTONDBLCLK = 0x0203

NIM_ADD = 0
NIM_DELETE = 2
NIF_MESSAGE = 0x00000001
NIF_ICON = 0x00000002
NIF_TIP = 0x00000004
NIF_SHOWTIP = 0x00000080

IDI_APPLICATION = 32512
IMAGE_ICON = 1
LR_LOADFROMFILE = 0x00000010
LR_DEFAULTSIZE = 0x00000040

MF_STRING = 0x00000000
MF_SEPARATOR = 0x00000800
TPM_RIGHTBUTTON = 0x0002
TPM_RETURNCMD = 0x0100

ID_OPEN = 1001
ID_QUIT = 1002

ERROR_ALREADY_EXISTS = 183


class NOTIFYICONDATA(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("hWnd", wintypes.HWND),
        ("uID", wintypes.UINT),
        ("uFlags", wintypes.UINT),
        ("uCallbackMessage", wintypes.UINT),
        ("hIcon", wintypes.HICON),
        ("szTip", wintypes.WCHAR * 128),
    ]


WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, WPARAM, LPARAM)


class WNDCLASS(ctypes.Structure):
    _fields_ = [
        ("style", wintypes.UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
    ]


def ensure_icon_file(path: Path | None = None, color: tuple[int, int, int, int] | None = None) -> Path:
    icon_path = path or DEFAULT_ICON
    if icon_path.exists():
        return icon_path
    icon_path.parent.mkdir(parents=True, exist_ok=True)
    fill = color or ICON_COLORS["panel"]
    try:
        from PIL import Image, ImageDraw

        img = Image.new("RGBA", (64, 64), (26, 29, 39, 255))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle((8, 20, 56, 52), radius=6, outline=fill, width=4)
        draw.ellipse((22, 26, 42, 46), outline=(138, 164, 255, 255), width=4)
        draw.ellipse((44, 24, 50, 30), fill=fill)
        draw.line((22, 20, 26, 12, 38, 12, 42, 20), fill=fill, width=4, joint="curve")
        img.save(icon_path, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (64, 64)])
        png = icon_path.with_suffix(".png")
        img.resize((32, 32), Image.Resampling.LANCZOS).save(png, format="PNG")
    except Exception:
        pass
    return icon_path


def icon_for(role: str) -> Path:
    """Отдельный .ico на роль: панель / watcher / puzzle / random."""
    color = ICON_COLORS.get(role, ICON_COLORS["panel"])
    path = ICON_DIR / f"tray_{role}.ico"
    if not path.exists():
        ensure_icon_file(path, color)
    return path


def load_icon(path: Path | None = None) -> wintypes.HICON:
    icon_path = ensure_icon_file(path)
    handle = user32.LoadImageW(
        None,
        str(icon_path),
        IMAGE_ICON,
        0,
        0,
        LR_LOADFROMFILE | LR_DEFAULTSIZE,
    )
    if handle:
        return handle
    return user32.LoadIconW(None, ctypes.c_void_p(IDI_APPLICATION))


def acquire_single_instance(mutex_name: str):
    """Named mutex: если уже есть — вернуть None."""
    if sys.platform != "win32":
        return object()
    ctypes.set_last_error(0)
    handle = kernel32.CreateMutexW(None, False, mutex_name)
    err = ctypes.get_last_error()
    if not handle:
        return None
    if err == ERROR_ALREADY_EXISTS:
        try:
            kernel32.CloseHandle(handle)
        except Exception:
            pass
        return None
    return handle


def release_mutex(handle) -> None:
    if sys.platform != "win32" or not handle:
        return
    try:
        kernel32.CloseHandle(handle)
    except Exception:
        pass


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return False
        kernel32.CloseHandle(handle)
        return True
    try:
        import os

        os.kill(pid, 0)
        return True
    except OSError:
        return False


def kill_self_tree() -> None:
    """Жёстко закрыть текущий процесс и дочерние (воркеры)."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.run(
        ["taskkill", "/PID", str(os_getpid()), "/F", "/T"],
        capture_output=True,
        check=False,
        creationflags=flags,
    )


def os_getpid() -> int:
    import os

    return os.getpid()


def patch_multiprocessing_no_window() -> None:
    """Чтобы mp-воркеры не мигали консолью."""
    if sys.platform != "win32":
        return
    try:
        import _winapi

        if getattr(_winapi, "_btc_no_window_patched", False):
            return
        _orig = _winapi.CreateProcess

        def _create(name, command_line, proc_attr, thread_attr, inherit, flags, env, cwd, startupinfo):
            flags = int(flags or 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
            return _orig(
                name, command_line, proc_attr, thread_attr, inherit, flags, env, cwd, startupinfo
            )

        _winapi.CreateProcess = _create  # type: ignore[assignment]
        _winapi._btc_no_window_patched = True
    except Exception:
        pass


class TrayIcon:
    """Блокирующий message loop. on_quit вызывается перед выходом из loop."""

    _uid_seq = 40

    def __init__(
        self,
        tip: str,
        *,
        open_url: str = PANEL_URL,
        quit_label: str = "Закрыть процесс",
        class_name: str | None = None,
        icon_path: Path | None = None,
        on_quit=None,
    ) -> None:
        TrayIcon._uid_seq += 1
        self.tip = tip[:127]
        self.open_url = open_url
        self.quit_label = quit_label
        self.class_name = class_name or f"LocalPhotoAiTray{TrayIcon._uid_seq}"
        self.icon_path = icon_path
        self.on_quit = on_quit
        self._wm_tray = WM_USER + TrayIcon._uid_seq
        self._uid = TrayIcon._uid_seq
        self._hwnd = None
        self._nid = None
        self._icon = None
        self._wndproc = None

    def _open_page(self) -> None:
        webbrowser.open(self.open_url)

    def _do_quit(self) -> None:
        if self.on_quit:
            try:
                self.on_quit()
            except Exception:
                pass
        if self._hwnd:
            user32.PostMessageW(self._hwnd, WM_DESTROY, 0, 0)

    def _show_menu(self) -> None:
        menu = user32.CreatePopupMenu()
        user32.AppendMenuW(menu, MF_STRING, ID_OPEN, "Открыть страницу")
        user32.AppendMenuW(menu, MF_SEPARATOR, 0, None)
        user32.AppendMenuW(menu, MF_STRING, ID_QUIT, self.quit_label)
        user32.SetForegroundWindow(self._hwnd)
        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        cmd = user32.TrackPopupMenu(
            menu,
            TPM_RIGHTBUTTON | TPM_RETURNCMD,
            pt.x,
            pt.y,
            0,
            self._hwnd,
            None,
        )
        user32.DestroyMenu(menu)
        user32.PostMessageW(self._hwnd, 0, 0, 0)
        if cmd == ID_OPEN:
            self._open_page()
        elif cmd == ID_QUIT:
            self._do_quit()

    def _wnd_proc(self, hwnd, msg, wparam, lparam):
        try:
            if msg == self._wm_tray:
                if lparam in (WM_LBUTTONUP, WM_LBUTTONDBLCLK):
                    self._open_page()
                    return 0
                if lparam == WM_RBUTTONUP:
                    self._show_menu()
                    return 0
            elif msg == WM_COMMAND:
                if wparam == ID_OPEN:
                    self._open_page()
                elif wparam == ID_QUIT:
                    self._do_quit()
                return 0
            elif msg == WM_DESTROY:
                self._remove_tray()
                user32.PostQuitMessage(0)
                return 0
            return user32.DefWindowProcW(hwnd, msg, wparam, lparam)
        except Exception:
            return 0

    def _add_tray(self) -> None:
        self._icon = load_icon(self.icon_path)
        nid = NOTIFYICONDATA()
        nid.cbSize = ctypes.sizeof(NOTIFYICONDATA)
        nid.hWnd = self._hwnd
        nid.uID = self._uid
        nid.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP | NIF_SHOWTIP
        nid.uCallbackMessage = self._wm_tray
        nid.hIcon = self._icon
        nid.szTip = self.tip
        self._nid = nid
        shell32.Shell_NotifyIconW(NIM_ADD, ctypes.byref(nid))

    def _remove_tray(self) -> None:
        if self._nid is not None:
            shell32.Shell_NotifyIconW(NIM_DELETE, ctypes.byref(self._nid))
            self._nid = None
        if self._icon:
            user32.DestroyIcon(self._icon)
            self._icon = None

    def run(self) -> int:
        if sys.platform != "win32":
            return 1
        self._wndproc = WNDPROC(self._wnd_proc)
        h_inst = kernel32.GetModuleHandleW(None)
        wc = WNDCLASS()
        wc.lpfnWndProc = self._wndproc
        wc.hInstance = h_inst
        wc.lpszClassName = self.class_name
        user32.RegisterClassW(ctypes.byref(wc))
        self._hwnd = user32.CreateWindowExW(
            0,
            self.class_name,
            self.tip,
            0,
            0,
            0,
            0,
            0,
            None,
            None,
            h_inst,
            None,
        )
        if not self._hwnd:
            return 1
        self._add_tray()
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        return 0


def set_app_id(app_id: str = APP_USER_MODEL_ID) -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass
