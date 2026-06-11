"""Train a fresh Synapse V2 LoRA without touching the existing model."""

import json
import os

import finetune as trainer


MODEL_TAG = "synapse-1.5b-v2"

# Keep the same Qwen base model, but isolate every V2 artifact.
trainer.DATASET_PATH = "data/mixed_dataset_v2.jsonl"
trainer.OUTPUT_DIR = "models/qwen1.5b-finetuned-v2"
trainer.TRAIN_PATH = "data/train_v2.jsonl"
trainer.VALID_PATH = "data/valid_v2.jsonl"


def main() -> None:
    if not os.path.exists(trainer.DATASET_PATH):
        print(f"Dataset not found: {trainer.DATASET_PATH}")
        raise SystemExit(1)

    with open(trainer.DATASET_PATH, encoding="utf-8") as dataset_file:
        pairs = [json.loads(line) for line in dataset_file if line.strip()]

    print(f"Model tag: {MODEL_TAG}")
    print(f"Base model: {trainer.MODEL_NAME}")
    print(f"Dataset: {trainer.DATASET_PATH}")
    print(f"Samples: {len(pairs)}")

    if not pairs:
        print("Dataset is empty. Add JSONL conversation samples before training.")
        raise SystemExit(1)

    if len(pairs) < 50:
        print("Warning: very small dataset; results may be poor.")

    trainer.convert_dataset()
    trainer.finetune()


if __name__ == "__main__":
    main()
