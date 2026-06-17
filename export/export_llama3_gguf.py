"""Merge and export the Synapse Llama LoRA to a quantized GGUF."""

import os
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
BASE_MODEL_PATH = PROJECT_DIR / "models" / "llama-base"
ADAPTER_PATH = PROJECT_DIR / "models" / "llama3-synapse-finetuned-v1"
MERGED_PATH = PROJECT_DIR / "models" / "llama3-synapse-merged-v1"
OUTPUT_DIR = PROJECT_DIR / "models" / "gguf"
F16_GGUF_PATH = OUTPUT_DIR / "synapse-llama3-v1-f16.gguf"
Q4_GGUF_PATH = OUTPUT_DIR / "synapse-llama3-v1-q4_k_m.gguf"
LLAMA_CPP_DIR = PROJECT_DIR / "llama.cpp"
CONVERTER_PATH = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
GGUF_PY_PATH = LLAMA_CPP_DIR / "gguf-py"
QUANTIZER_PATH = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"


def run(command: list[str], label: str, *, env: dict[str, str] | None = None) -> None:
    print(f"\n{label}")
    print(" ".join(command))
    subprocess.run(command, check=True, env=env)


def validate() -> None:
    if not BASE_MODEL_PATH.exists():
        raise SystemExit(f"Base model not found: {BASE_MODEL_PATH}")
    if not (ADAPTER_PATH / "adapters.safetensors").exists():
        raise SystemExit(f"Llama adapters not found: {ADAPTER_PATH}")
    for path in (CONVERTER_PATH, GGUF_PY_PATH, QUANTIZER_PATH):
        if not path.exists():
            raise SystemExit(f"Required local llama.cpp path not found: {path}")


def converter_env() -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{GGUF_PY_PATH}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(GGUF_PY_PATH)
    )
    return env


def main() -> None:
    validate()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not (MERGED_PATH / "config.json").exists():
        run(
            [
                sys.executable,
                "-m",
                "mlx_lm",
                "fuse",
                "--model",
                str(BASE_MODEL_PATH),
                "--adapter-path",
                str(ADAPTER_PATH),
                "--save-path",
                str(MERGED_PATH),
                "--dequantize",
            ],
            "1/3 Fusing Llama LoRA adapters",
        )
    else:
        print(f"\n1/3 Merged Llama model already exists: {MERGED_PATH}")

    if not F16_GGUF_PATH.exists():
        run(
            [
                sys.executable,
                str(CONVERTER_PATH),
                str(MERGED_PATH),
                "--outfile",
                str(F16_GGUF_PATH),
                "--outtype",
                "f16",
            ],
            "2/3 Converting merged Llama model to F16 GGUF",
            env=converter_env(),
        )
    else:
        print(f"\n2/3 F16 GGUF already exists: {F16_GGUF_PATH}")

    if not Q4_GGUF_PATH.exists():
        run(
            [
                str(QUANTIZER_PATH),
                str(F16_GGUF_PATH),
                str(Q4_GGUF_PATH),
                "Q4_K_M",
            ],
            "3/3 Quantizing Llama GGUF to Q4_K_M",
        )
    else:
        print(f"\n3/3 Quantized GGUF already exists: {Q4_GGUF_PATH}")

    print(f"\nLlama GGUF ready: {Q4_GGUF_PATH}")


if __name__ == "__main__":
    main()
