"""Download Qwen2.5 3B Instruct and convert it to a 4-bit MLX model."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
HF_MODEL = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR = PROJECT_DIR / "models" / "qwen2.5-3b-instruct-4bit"


def model_is_complete(path: Path) -> bool:
    return (path / "config.json").exists() and any(path.glob("*.safetensors"))


def main() -> None:
    if model_is_complete(OUTPUT_DIR):
        print(f"Qwen2.5 3B 4-bit MLX model already exists: {OUTPUT_DIR}")
        return

    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "mlx_lm",
        "convert",
        "--hf-path",
        HF_MODEL,
        "--mlx-path",
        str(OUTPUT_DIR),
        "--quantize",
        "--q-bits",
        "4",
    ]
    print("Preparing Qwen2.5 3B 4-bit MLX base:")
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_DIR, check=True)

    if not model_is_complete(OUTPUT_DIR):
        raise SystemExit(f"Conversion finished without model weights in {OUTPUT_DIR}")
    print(f"Qwen2.5 3B 4-bit MLX model ready: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
