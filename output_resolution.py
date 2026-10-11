"""Image dimensions for diffusion pipelines (source resolution, model alignment)."""

from __future__ import annotations

import io

from PIL import Image

_MIN_SIDE = 64


def image_pixel_size(image_bytes: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(image_bytes)) as img:
        return img.size


def image_dimensions_label(image_bytes: bytes) -> str:
    w, h = image_pixel_size(image_bytes)
    return format_dimensions_label(w, h)


def fit_long_edge(width: int, height: int, max_side: int) -> tuple[int, int]:
    """Scale so the longer side equals max_side (up or down), keep aspect ratio."""
    w, h = max(1, int(width)), max(1, int(height))
    long_edge = max(w, h)
    scale = max_side / long_edge
    return max(_MIN_SIDE, int(round(w * scale))), max(_MIN_SIDE, int(round(h * scale)))


def align_dimensions(width: int, height: int, multiple: int) -> tuple[int, int]:
    m = max(1, int(multiple))
    w = max(m, (int(width) // m) * m)
    h = max(m, (int(height) // m) * m)
    return w, h


def pipeline_dimensions(
    width: int,
    height: int,
    *,
    pipe_class_name: str = "",
    vae_scale_factor: int = 8,
) -> tuple[int, int]:
    """Return (height, width) for diffusers at source size (aligned to model grid)."""
    w, h = max(1, int(width)), max(1, int(height))

    if "Qwen" in pipe_class_name:
        w, h = align_dimensions(w, h, 32)
        return h, w

    factor = max(2, int(vae_scale_factor))
    w = max(factor * 2, (w // factor) * factor)
    h = max(factor * 2, (h // factor) * factor)
    return h, w


def format_dimensions_label(width: int, height: int) -> str:
    return f"{width}×{height}"
