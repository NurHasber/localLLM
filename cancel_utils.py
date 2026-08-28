"""Inject an exception into a worker thread (same process) to unblock GPU work."""

from __future__ import annotations

import ctypes
import threading


def inject_thread_exception(thread_id: int | None, exc: type[BaseException]) -> bool:
    """Raise *exc* asynchronously in the target thread. Returns True if dispatched."""
    if not thread_id or thread_id == threading.get_ident():
        return False
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), ctypes.py_object(exc))
    if res == 0:
        return False
    if res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), None)
        return False
    return True
