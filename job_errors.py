"""Job control exceptions."""


class JobCancelled(Exception):
    """Raised when user cancels image generation (not model download)."""


class JobPaused(Exception):
    """Raised when user pauses generation (resume from last checkpoint)."""
