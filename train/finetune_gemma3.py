"""Train a fresh Synapse LoRA on a local Gemma 3 base model.

The Ollama tag is for inference. LoRA training needs a local Hugging Face-style
model folder, expected here as `models/gemma3-4b-it`.
"""

import json
import os
import random
import subprocess
import sys
from pathlib import Path

from transformers import AutoTokenizer


MODEL_NAME = "models/gemma3-4b-it"
DATASET_PATH = "data/mixed_dataset_v2.jsonl"
OUTPUT_DIR = "models/gemma3-4b-synapse-finetuned-v1"
DATA_DIR = Path("data/gemma3-4b")
TRAIN_PATH = DATA_DIR / "train.jsonl"
VALID_PATH = DATA_DIR / "valid.jsonl"

LORA_RANK = 16
LORA_LAYERS = 8
LEARNING_RATE = 1e-5
ITERATIONS = 800
BATCH_SIZE = 1
STEPS_PER_EVAL = 200
STEPS_PER_SAVE = 200
TRAIN_SPLIT = 0.9
SPLIT_SEED = 42

SYSTEM_PROMPT = (
    "You are a helpful AI assistant for Synapse Tech Inc. Answer naturally, "
    "accurately, and stay grounded in company information when relevant."
)


def format_gemma_chat(pair: dict, tokenizer) -> dict:
    # Use Gemma's own tokenizer template so BOS/system/user/model tokens match
    # the exact format the base model expects.
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(pair["messages"])
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def convert_dataset() -> None:
    print("Converting dataset to Gemma chat format...")
    with open(DATASET_PATH, encoding="utf-8") as dataset_file:
        pairs = [json.loads(line) for line in dataset_file if line.strip()]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    formatted = [format_gemma_chat(pair, tokenizer) for pair in pairs]

    random.seed(SPLIT_SEED)
    random.shuffle(formatted)

    split_index = int(len(formatted) * TRAIN_SPLIT)
    train_data = formatted[:split_index]
    valid_data = formatted[split_index:]

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(TRAIN_PATH, "w", encoding="utf-8") as train_file:
        for item in train_data:
            train_file.write(json.dumps(item) + "\n")

    with open(VALID_PATH, "w", encoding="utf-8") as valid_file:
        for item in valid_data:
            valid_file.write(json.dumps(item) + "\n")

    print(f"  Train samples: {len(train_data)}")
    print(f"  Valid samples: {len(valid_data)}")
    print(f"  Saved to: {DATA_DIR}")


def find_checkpoint() -> str | None:
    if not os.path.exists(OUTPUT_DIR):
        return None

    adapter_files = [
        filename
        for filename in os.listdir(OUTPUT_DIR)
        if filename.endswith(".safetensors") or filename.endswith(".npz")
    ]
    if not adapter_files:
        return None

    numbered = sorted(filename for filename in adapter_files if filename[:6].isdigit())
    latest = numbered[-1] if numbered else sorted(adapter_files)[-1]
    return os.path.join(OUTPUT_DIR, latest)


def finetune() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        MODEL_NAME,
        "--train",
        "--data",
        str(DATA_DIR),
        "--adapter-path",
        OUTPUT_DIR,
        "--num-layers",
        str(LORA_LAYERS),
        "--batch-size",
        str(BATCH_SIZE),
        "--iters",
        str(ITERATIONS),
        "--learning-rate",
        str(LEARNING_RATE),
        "--steps-per-eval",
        str(STEPS_PER_EVAL),
        "--save-every",
        str(STEPS_PER_SAVE),
        "--val-batches",
        "10",
    ]

    checkpoint = find_checkpoint()
    if checkpoint:
        print(f"Resuming from checkpoint: {checkpoint}")
        cmd.extend(["--resume-adapter-file", checkpoint])
    else:
        print("Starting fresh Gemma LoRA training")

    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print("Running command:")
    print(" ".join(cmd))
    print("-" * 50)

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("Training failed. Check errors above.")
        raise SystemExit(result.returncode)

    print("-" * 50)
    print("Gemma fine-tuning complete.")
    print(f"Adapters saved to: {OUTPUT_DIR}")


def main() -> None:
    if not os.path.exists(MODEL_NAME):
        print(f"Base model not found: {MODEL_NAME}")
        print("Download the Hugging Face Gemma 3 4B model into that folder before training.")
        raise SystemExit(1)

    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found: {DATASET_PATH}")
        raise SystemExit(1)

    convert_dataset()
    finetune()


if __name__ == "__main__":
    main()
