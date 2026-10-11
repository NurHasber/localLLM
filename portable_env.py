"""Project-local paths — import before huggingface/torch side effects."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HF_HOME = ROOT / "models" / "huggingface"
HF_HUB = HF_HOME / "hub"

HF_HOME.mkdir(parents=True, exist_ok=True)
HF_HUB.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HUB))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
