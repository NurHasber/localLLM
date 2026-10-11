"""CLI for Fun-CosyVoice3: clone a voice from a short WAV and speak RU/EN."""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, "third_party", "Matcha-TTS"))
os.chdir(ROOT)

import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel

MODEL_DIR = os.path.join(ROOT, "pretrained_models", "Fun-CosyVoice3-0.5B")
DEFAULT_PROMPT = os.path.join(ROOT, "asset", "zero_shot_prompt.wav")
DEFAULT_PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"


def main():
    parser = argparse.ArgumentParser(description="CosyVoice3 local TTS (zero-shot clone / instruct)")
    parser.add_argument("--text", required=True, help="Text to speak (Russian or English)")
    parser.add_argument("--out", default="output.wav", help="Output WAV path")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Reference voice WAV, 3-15 seconds")
    parser.add_argument(
        "--prompt-text",
        default=DEFAULT_PROMPT_TEXT,
        help="Transcript of the prompt WAV, or instruct prefix + transcript",
    )
    parser.add_argument(
        "--instruct",
        default="",
        help='Optional style, e.g. "请用俄语表达" or "Speak in a calm male space-pilot voice."',
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "zero_shot", "cross_lingual", "instruct"],
        default="auto",
        help="auto: instruct if --instruct, else cross_lingual (avoids source-language accent at the start)",
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true", default=True, help="FP16 on GPU (default on)")
    parser.add_argument("--fp32", action="store_true", help="Disable FP16")
    args = parser.parse_args()

    fp16 = not args.fp32
    print("Loading", MODEL_DIR, "fp16=", fp16)
    cosyvoice = AutoModel(model_dir=MODEL_DIR, fp16=fp16)

    mode = args.mode
    if mode == "auto":
        mode = "instruct" if args.instruct else "cross_lingual"

    pieces = []
    if mode == "instruct":
        instruct = args.instruct or "请用俄语表达。"
        if "<|endofprompt|>" not in instruct:
            instruct = f"You are a helpful assistant. {instruct}<|endofprompt|>"
        print("Mode: instruct2")
        iterator = cosyvoice.inference_instruct2(
            args.text, instruct, args.prompt, stream=False, speed=args.speed
        )
    elif mode == "cross_lingual":
        print("Mode: cross_lingual")
        iterator = cosyvoice.inference_cross_lingual(
            args.text, args.prompt, stream=False, speed=args.speed
        )
    else:
        prompt_text = args.prompt_text
        if "<|endofprompt|>" not in prompt_text:
            prompt_text = f"You are a helpful assistant.<|endofprompt|>{prompt_text}"
        print("Mode: zero_shot")
        iterator = cosyvoice.inference_zero_shot(
            args.text, prompt_text, args.prompt, stream=False, speed=args.speed
        )

    for i, chunk in enumerate(iterator):
        pieces.append(chunk["tts_speech"])
        print("chunk", i, "samples", chunk["tts_speech"].shape)

    if not pieces:
        raise SystemExit("No audio generated")

    import torch

    audio = torch.cat(pieces, dim=1)
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torchaudio.save(out_path, audio, cosyvoice.sample_rate)
    print("Saved", out_path, "sr", cosyvoice.sample_rate)


if __name__ == "__main__":
    main()
