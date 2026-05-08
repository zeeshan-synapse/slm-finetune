#!/usr/bin/env bash
set -euo pipefail

# End-to-end training/export/eval cycle for Synapse Qwen runs.
# Usage:
#   scripts/run_cycle.sh [model_tag] [run_id]
#
# Example:
#   scripts/run_cycle.sh synapse-qwen1.5b-v3 v3

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$PROJECT_DIR/llama.cpp}"

MODEL_TAG="${1:-synapse-qwen1.5b-v3}"
RUN_ID="${2:-v3}"

MERGED_DIR="$PROJECT_DIR/models/qwen1.5b-merged"
GGUF_DIR="$PROJECT_DIR/models/gguf"
GGUF_F16="$GGUF_DIR/synapse-qwen1.5b.gguf"
GGUF_Q4="$GGUF_DIR/synapse-qwen1.5b-q4_k_m.gguf"

MODELF="$PROJECT_DIR/ollama/modelfiles/Modelfile.$RUN_ID"
EVAL_DIR="$PROJECT_DIR/eval"
EVAL_PROMPTS="$EVAL_DIR/prompts/v2_50.txt"
EVAL_OUT="$EVAL_DIR/results/eval_results_${RUN_ID}.jsonl"
SMOKE_OUT="$EVAL_DIR/results/eval_smoke_${RUN_ID}.jsonl"

echo "=================================================="
echo "Synapse automation cycle"
echo "Project:   $PROJECT_DIR"
echo "llama.cpp: $LLAMA_CPP_DIR"
echo "Model tag: $MODEL_TAG"
echo "Run id:    $RUN_ID"
echo "=================================================="

if [[ ! -d "$LLAMA_CPP_DIR" ]]; then
  echo "ERROR: llama.cpp directory not found: $LLAMA_CPP_DIR"
  echo "Set LLAMA_CPP_DIR env var and rerun."
  exit 1
fi

if [[ ! -x "$LLAMA_CPP_DIR/build/bin/llama-quantize" ]]; then
  echo "ERROR: llama-quantize not found at $LLAMA_CPP_DIR/build/bin/llama-quantize"
  echo "Build llama.cpp first."
  exit 1
fi

if [[ ! -f "$EVAL_PROMPTS" ]]; then
  echo "ERROR: missing eval prompts file: $EVAL_PROMPTS"
  exit 1
fi

mkdir -p "$EVAL_DIR/results"
mkdir -p "$PROJECT_DIR/ollama/modelfiles"

echo "== [1/11] Cleanup stale export artifacts =="
rm -rf "$MERGED_DIR"
rm -f "$GGUF_F16" "$GGUF_Q4"

echo "== [2/11] Mix datasets =="
cd "$PROJECT_DIR"
python3 dataset/mix_datasets.py

echo "== [3/11] Fine-tune LoRA adapters =="
python3 train/finetune.py

echo "== [4/11] Merge adapters + export prep =="
python3 export/export_gguf.py

echo "== [5/11] Convert merged model to GGUF (f16) =="
cd "$LLAMA_CPP_DIR"
python3 convert_hf_to_gguf.py \
  "$MERGED_DIR" \
  --outfile "$GGUF_F16" \
  --outtype f16

echo "== [6/11] Quantize GGUF (Q4_K_M) =="
"$LLAMA_CPP_DIR/build/bin/llama-quantize" \
  "$GGUF_F16" \
  "$GGUF_Q4" \
  Q4_K_M

echo "== [7/11] Create run-specific Modelfile =="
cat > "$MODELF" <<EOF
FROM $GGUF_Q4
PARAMETER temperature 0.1
PARAMETER top_p 0.8
PARAMETER repeat_penalty 1.2
PARAMETER num_predict 60
EOF

echo "== [8/11] Create Ollama tag: $MODEL_TAG =="
ollama create "$MODEL_TAG" -f "$MODELF"

echo "== [9/11] Run 50-prompt eval =="
cd "$PROJECT_DIR"
python3 - <<PY
import json
import pathlib
import subprocess

model = "$MODEL_TAG"
prompts = [
    p.strip()
    for p in pathlib.Path("$EVAL_PROMPTS").read_text(encoding="utf-8").splitlines()
    if p.strip()
]
out_path = pathlib.Path("$EVAL_OUT")

with out_path.open("w", encoding="utf-8") as f:
    for i, prompt in enumerate(prompts, 1):
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
        )
        response = (result.stdout or "").strip()
        f.write(
            json.dumps(
                {"id": i, "prompt": prompt, "response": response},
                ensure_ascii=False,
            )
            + "\\n"
        )

print(f"Saved: {out_path}")
print(f"Prompts: {len(prompts)}")
PY

echo "== [10/11] Run 5-question smoke check =="
python3 scripts/smoke_eval.py --model "$MODEL_TAG" --output "$SMOKE_OUT"

echo "== [11/11] Build corrective pairs from runtime failures =="
if [[ -f "$PROJECT_DIR/logs/guardrail_failures.jsonl" ]]; then
  python3 scripts/build_corrective_from_logs.py \
    --input "$PROJECT_DIR/logs/guardrail_failures.jsonl" \
    --output "$PROJECT_DIR/data/corrective_pairs_from_logs.jsonl"
else
  echo "No guardrail failure log found yet; skipping corrective pair build."
fi

echo "== Done =="
echo "Model tag:  $MODEL_TAG"
echo "Eval file:  $EVAL_OUT"
echo "Smoke file: $SMOKE_OUT"
echo "Modelfile:  $MODELF"
echo
echo "Next:"
echo "  python3 scripts/score_eval.py --input \"$EVAL_OUT\""
