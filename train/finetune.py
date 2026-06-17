import json
import os
import random
import subprocess
import sys

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME = "./models/qwen25-1.5b-instruct"                   # HuggingFace model ID
DATASET_PATH = "data/mixed_dataset.jsonl"           # blended website + chat pairs
OUTPUT_DIR = "models/qwen1.5b-finetuned"                # where adapters are saved
TRAIN_PATH = "data/train.jsonl"                     # MLX expects JSONL format
VALID_PATH = "data/valid.jsonl"                     # validation split

# ── LoRA Hyperparameters ─────────────────────────────────────────────────────
LORA_RANK = 16       # more capacity, still efficient on M2 Pro
LORA_LAYERS = 8     # more layers = deeper fine-tuning
LEARNING_RATE = 1e-4 # slightly higher for faster convergence
ITERATIONS = 800     # enough for 383 pairs to converge
BATCH_SIZE = 1       # keep this, safe for M2 Pro memory
STEPS_PER_EVAL = 200 # evaluate every 100 steps
STEPS_PER_SAVE = 200 # save checkpoint every 200 steps
TRAIN_SPLIT = 0.9    # 90% train, 10% validation
SPLIT_SEED = 42      # deterministic train/valid split

# ── Step 1: Convert dataset to MLX JSONL format ──────────────────────────────

def convert_dataset():
    print("📦 Converting dataset to MLX format...")

    pairs = []
    with open(DATASET_PATH, "r") as f:
        pairs = [json.loads(line) for line in f if line.strip()]

    # MLX expects pre-formatted chat text; this uses Qwen-style chat tokens.
    def format_pair(pair):
        parts = []
        parts.append("<|im_start|>system\nYou are a helpful AI assistant for Synapse Tech Inc. Answer questions accurately, stay grounded in company information when relevant, and maintain a natural conversational tone.<|im_end|>\n")
        for msg in pair["messages"]:
            role = msg["role"]  # user / assistant
            content = msg["content"].strip()
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        return {"text": "".join(parts)}

    formatted = [format_pair(p) for p in pairs]

    # Shuffle before split so validation is not biased by dataset ordering.
    random.seed(SPLIT_SEED)
    random.shuffle(formatted)

    # Split into train/validation
    split_idx = int(len(formatted) * TRAIN_SPLIT)
    train_data = formatted[:split_idx]
    valid_data = formatted[split_idx:]

    os.makedirs("data", exist_ok=True)

    # Save as JSONL (one JSON object per line)
    with open(TRAIN_PATH, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with open(VALID_PATH, "w") as f:
        for item in valid_data:
            f.write(json.dumps(item) + "\n")

    print(f"  ✅ Train samples: {len(train_data)}")
    print(f"  ✅ Valid samples: {len(valid_data)}")
    print(f"  📁 Saved to {TRAIN_PATH} and {VALID_PATH}\n")


# ── Step 2: Run MLX LoRA Fine-tuning ─────────────────────────────────────────

def finetune():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("🚀 Starting LoRA fine-tuning with MLX...")
    print(f"   Model:      {MODEL_NAME}")
    print(f"   Dataset:    {len(open(TRAIN_PATH).readlines())} train samples")
    print(f"   LoRA rank:  {LORA_RANK}")
    print(f"   Iterations: {ITERATIONS}")
    print(f"   Output:     {OUTPUT_DIR}")
    print()

    # Build base command
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", MODEL_NAME,
        "--train",
        "--data", "data",
        "--adapter-path", OUTPUT_DIR,
        "--num-layers", str(LORA_LAYERS),
        "--batch-size", str(BATCH_SIZE),
        "--iters", str(ITERATIONS),
        "--learning-rate", str(LEARNING_RATE),
        "--steps-per-eval", str(STEPS_PER_EVAL),
        "--save-every", str(STEPS_PER_SAVE),
        "--val-batches", "10",
    ]

    # Check if checkpoint exists to resume from (.safetensors or legacy .npz)
    checkpoint = None
    if os.path.exists(OUTPUT_DIR):
        adapter_files = [
            f for f in os.listdir(OUTPUT_DIR)
            if f.endswith(".safetensors") or f.endswith(".npz")
        ]
        if adapter_files:
            # Prefer numbered checkpoints if present (e.g. 0000200_adapters.safetensors).
            numbered = sorted(
                [f for f in adapter_files if f[:6].isdigit()]
            )
            latest = numbered[-1] if numbered else sorted(adapter_files)[-1]
            checkpoint = os.path.join(OUTPUT_DIR, latest)

    if checkpoint:
        print(f"  🔄 Resuming from checkpoint: {checkpoint}")
        cmd.extend(["--resume-adapter-file", checkpoint])
    else:
        print("  🆕 Starting fresh training\n")

    print(f"Running command:\n{' '.join(cmd)}\n")
    print("─" * 50)

    try:
        result = subprocess.run(cmd, capture_output=False)

        if result.returncode == 0:
            print("\n" + "─" * 50)
            print("✅ Fine-tuning complete!")
            print(f"📁 LoRA adapters saved to: {OUTPUT_DIR}")
            print("\nNext step: run export/export_gguf.py to merge and export")
        else:
            print("\n❌ Training failed. Check errors above.")
            print("💡 Tip: Run again to resume from last checkpoint")

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print("💡 Run again to resume from last checkpoint automatically")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Check dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found at {DATASET_PATH}")
        print("Run dataset/prepare.py first")
        exit(1)

    # Check dataset size
    pairs = []
    with open(DATASET_PATH) as f:
        pairs = [json.loads(line) for line in f if line.strip()]

    print(f"📊 Dataset: {len(pairs)} Q&A pairs")

    if len(pairs) < 50:
        print("⚠️  Warning: Very small dataset. Results may be poor.")

    print()

    # Run pipeline
    convert_dataset()
    finetune()
