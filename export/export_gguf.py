import os
import sys
import subprocess
import shutil
import time

# ── Config ──────────────────────────────────────────────────────────────────
BASE_MODEL_PATH = "./models/qwen2.5-1.5b-instruct"  # original HF model
ADAPTER_PATH = "models/qwen1.5b-finetuned"                      # LoRA adapters from training
MERGED_PATH = "models/qwen1.5b-merged"                          # merged model (intermediate)
OUTPUT_DIR = "models/gguf"                             # final GGUF output
GGUF_FILENAME = "synapse-qwen1.5b.gguf"                 # your custom model name
QUANTIZATION = "q4_k_m"                               # 4-bit quantization

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds between retries

# ── Helpers ──────────────────────────────────────────────────────────────────

def run_with_retry(cmd: list, step_name: str) -> bool:
    """Run a command with retry logic. Returns True if successful."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  🔄 Attempt {attempt}/{MAX_RETRIES}...")
            result = subprocess.run(
                cmd,
                capture_output=False,
                timeout=600  # 10 min timeout per step
            )

            if result.returncode == 0:
                print(f"  ✅ {step_name} succeeded")
                return True
            else:
                print(f"  ❌ {step_name} failed (exit code {result.returncode})")
                if attempt < MAX_RETRIES:
                    print(f"  ⏳ Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)

        except subprocess.TimeoutExpired:
            print(f"  ❌ {step_name} timed out on attempt {attempt}")
            if attempt < MAX_RETRIES:
                print(f"  ⏳ Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)

        except KeyboardInterrupt:
            print(f"\n⚠️  Interrupted during {step_name}")
            print("💡 Re-run export_gguf.py to retry from this step")
            sys.exit(1)

        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    print(f"  ❌ All {MAX_RETRIES} attempts failed for: {step_name}")
    return False


# ── Step 1: Validate inputs ──────────────────────────────────────────────────

def validate():
    print("🔍 Validating inputs...")

    errors = []

    if not os.path.exists(BASE_MODEL_PATH):
        errors.append(f"Base model not found: {BASE_MODEL_PATH}")

    if not os.path.exists(ADAPTER_PATH):
        errors.append(f"Adapter path not found: {ADAPTER_PATH}")
    else:
        adapters = [f for f in os.listdir(ADAPTER_PATH) if f.endswith(".npz") or f.endswith(".safetensors")]
        if not adapters:
            errors.append(f"No adapter files found in: {ADAPTER_PATH}")
        else:
            print(f"  ✅ Found adapters: {adapters}")

    if errors:
        print("\n❌ Validation failed:")
        for e in errors:
            print(f"   - {e}")
        return False

    print("  ✅ All inputs valid\n")
    return True


# ── Step 2: Merge LoRA adapters into base model ───────────────────────────────

def merge_adapters():
    print("🔀 Merging LoRA adapters into base model...")

    # Skip if already merged
    if os.path.exists(MERGED_PATH) and os.listdir(MERGED_PATH):
        print(f"  ⚡ Merged model already exists at {MERGED_PATH}, skipping...")
        return True

    os.makedirs(MERGED_PATH, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", BASE_MODEL_PATH,
        "--adapter-path", ADAPTER_PATH,
        "--save-path", MERGED_PATH,
        "--dequantize",  # convert back to full precision for GGUF export
    ]

    success = run_with_retry(cmd, "Merge adapters")

    if not success:
        # Clean up partial merge
        if os.path.exists(MERGED_PATH):
            shutil.rmtree(MERGED_PATH)
            print(f"  🧹 Cleaned up partial merge at {MERGED_PATH}")

    return success


# ── Step 3: Convert to GGUF ───────────────────────────────────────────────────

def convert_to_gguf():
    print("📦 Converting to GGUF format...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, GGUF_FILENAME)

    # Skip if already converted
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  ⚡ GGUF already exists ({size_mb:.0f}MB), skipping...")
        return True

    # Use mlx_lm's built-in GGUF export
    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", BASE_MODEL_PATH,
        "--adapter-path", ADAPTER_PATH,
        "--save-path", MERGED_PATH,
        "--export-gguf",
        "--gguf-path", output_path,
    ]

    success = run_with_retry(cmd, "GGUF conversion")

    if success and os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  📁 GGUF saved: {output_path} ({size_mb:.0f}MB)")

    return success


# ── Step 4: Quantize with llama.cpp (optional but recommended) ────────────────

def quantize_gguf():
    print("⚡ Quantizing GGUF to Q4_K_M...")

    input_path = os.path.join(OUTPUT_DIR, GGUF_FILENAME)
    quantized_name = GGUF_FILENAME.replace(".gguf", f"-{QUANTIZATION}.gguf")
    output_path = os.path.join(OUTPUT_DIR, quantized_name)

    # Check if llama-quantize is available
    llama_quantize = shutil.which("llama-quantize")
    if not llama_quantize:
        print("  ⚠️  llama-quantize not found, skipping quantization step")
        print("  ℹ️  Install llama.cpp to quantize: brew install llama.cpp")
        return True  # not a fatal error

    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  ⚡ Quantized GGUF already exists ({size_mb:.0f}MB), skipping...")
        return True

    cmd = [llama_quantize, input_path, output_path, "Q4_K_M"]
    success = run_with_retry(cmd, "Quantization")

    if success and os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  📁 Quantized GGUF: {output_path} ({size_mb:.0f}MB)")

    return success


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Synapse SLM — Export to GGUF")
    print("=" * 50)
    print()

    # Step 1: Validate
    if not validate():
        sys.exit(1)

    # Step 2: Merge adapters
    if not merge_adapters():
        print("\n❌ Export failed at merge step. Fix errors above and retry.")
        sys.exit(1)

    print()

    # Step 3: Convert to GGUF
    if not convert_to_gguf():
        print("\n❌ Export failed at GGUF conversion step.")
        print("💡 Try installing llama.cpp: brew install llama.cpp")
        sys.exit(1)

    print()

    # Step 4: Quantize (optional)
    quantize_gguf()

    print()
    print("=" * 50)
    print("✅ Export complete!")
    print(f"📁 Your custom GGUF is at: {OUTPUT_DIR}/{GGUF_FILENAME}")
    print()
    print("Next steps:")
    print("  1. Test with Ollama: ollama run " + os.path.join(OUTPUT_DIR, GGUF_FILENAME))
    print("  2. Drop into iOS app via LocalLLMClient")
    print("=" * 50)