"""Train an isolated Qwen2.5 3B QLoRA without touching 1.5B artifacts."""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "qwen2.5-3b-instruct-4bit"
DATASET_PATH = PROJECT_DIR / "data" / "mixed_dataset_v2.jsonl"
DATA_DIR = PROJECT_DIR / "data" / "qwen3b"
TRAIN_PATH = DATA_DIR / "train.jsonl"
VALID_PATH = DATA_DIR / "valid.jsonl"
ADAPTER_PATH = PROJECT_DIR / "models" / "qwen2.5-3b-synapse-lora-v1"
CONFIG_PATH = PROJECT_DIR / "train" / "configs" / "qwen3b_lora.yaml"
TRAIN_SPLIT = 0.9
SPLIT_SEED = 42
SYSTEM_PROMPT = (
    "You are a helpful AI assistant for Synapse Tech Inc. Answer questions "
    "accurately, stay grounded in company information when relevant, and "
    "maintain a natural conversational tone."
)


def model_is_complete(path: Path) -> bool:
    return (path / "config.json").exists() and any(path.glob("*.safetensors"))


def format_qwen_chat(pair: dict[str, Any]) -> dict[str, str]:
    messages = pair.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Each dataset row must contain a non-empty messages list")

    parts = [f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"]
    for message in messages:
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            raise ValueError(f"Invalid conversation message: {message!r}")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return {"text": "".join(parts)}


def prepare_dataset() -> tuple[int, int]:
    if not DATASET_PATH.exists():
        raise SystemExit(f"Dataset not found: {DATASET_PATH}")

    pairs: list[dict[str, Any]] = []
    with DATASET_PATH.open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"Invalid JSON at {DATASET_PATH}:{line_number}: {exc}"
                ) from exc

    formatted = [format_qwen_chat(pair) for pair in pairs]
    random.Random(SPLIT_SEED).shuffle(formatted)
    split_index = int(len(formatted) * TRAIN_SPLIT)
    train_rows = formatted[:split_index]
    valid_rows = formatted[split_index:]
    if not train_rows or not valid_rows:
        raise SystemExit("Dataset must produce non-empty train and validation splits")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for path, rows in ((TRAIN_PATH, train_rows), (VALID_PATH, valid_rows)):
        with path.open("w", encoding="utf-8") as output_file:
            for row in rows:
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    return len(train_rows), len(valid_rows)


def latest_checkpoint() -> Path | None:
    checkpoints = sorted(ADAPTER_PATH.glob("[0-9]*_adapters.safetensors"))
    return checkpoints[-1] if checkpoints else None


def train() -> None:
    if not model_is_complete(MODEL_PATH):
        raise SystemExit(
            f"Complete 4-bit MLX base not found: {MODEL_PATH}\n"
            "Run: ./venv/bin/python scripts/prepare_qwen3b_mlx.py"
        )
    if not CONFIG_PATH.exists():
        raise SystemExit(f"Training config not found: {CONFIG_PATH}")

    train_count, valid_count = prepare_dataset()
    ADAPTER_PATH.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--config",
        str(CONFIG_PATH),
    ]
    checkpoint = latest_checkpoint()
    if checkpoint is not None:
        command.extend(["--resume-adapter-file", str(checkpoint)])

    print("Qwen2.5 3B QLoRA configuration")
    print(f"Base: {MODEL_PATH}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Train/valid: {train_count}/{valid_count}")
    print(f"Adapters: {ADAPTER_PATH}")
    if checkpoint is not None:
        print(f"Resuming: {checkpoint}")
    print("Running:")
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_DIR, check=True)
    print(f"Qwen2.5 3B adapters ready: {ADAPTER_PATH}")


if __name__ == "__main__":
    train()
