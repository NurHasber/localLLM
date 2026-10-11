"""Validate Qwen Edit LoRA pairs: source/ and target/ with the same file stem."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "training" / "qwen-edit-lora" / "source"
TARGET = ROOT / "training" / "qwen-edit-lora" / "target"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> int:
    SOURCE.mkdir(parents=True, exist_ok=True)
    TARGET.mkdir(parents=True, exist_ok=True)
    targets = {
        p.stem: p
        for p in TARGET.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXT
    }
    sources = {
        p.stem: p
        for p in SOURCE.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXT
    }
    missing_src = sorted(set(targets) - set(sources))
    missing_tgt = sorted(set(sources) - set(targets))
    paired = sorted(set(targets) & set(sources))
    no_caption = [stem for stem in paired if not (TARGET / f"{stem}.txt").is_file()]
    print(f"source: {SOURCE}")
    print(f"target: {TARGET}")
    print(f"pairs: {len(paired)}")
    if missing_src:
        print("no source for:", ", ".join(missing_src))
    if missing_tgt:
        print("no target for:", ", ".join(missing_tgt))
    if no_caption:
        print("no English .txt caption for:", ", ".join(no_caption))
    if len(paired) < 10:
        print("Need at least ~10–30 strong pairs before training is useful.")
        return 1
    print("Dataset looks usable. Copy the resulting .safetensors into loras/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
