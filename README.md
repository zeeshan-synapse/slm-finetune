# Synapse SLM Fine-Tuning (Current Working Setup)

This project fine-tunes a local model for Synapse Tech website/domain Q&A and compares fine-tuned vs base behavior.

## Current status (working now)

- Working base family: `Qwen2.5-1.5B-Instruct`
- Working fine-tuned output: `synapse-qwen1.5b-q4_k_m.gguf`
- Working Ollama tags:
  - Existing fine-tune: `synapse-1.5b-v1`
  - New dataset fine-tune: `synapse-1.5b-v2`
  - Base: `qwen-base`

## Important notes from experiments

- `Llama` 1B and 3B instruct variants were tested earlier and were not giving reliable results for this use case in this setup.
- `Qwen` 7B was too heavy for local MLX training/inference on this machine (Metal OOM).
- `Qwen2.5-1.5B-Instruct` is the stable model choice currently.

## End-to-end workflow

### Qwen2.5 3B QLoRA pipeline

The 3B pipeline is isolated from the existing 1.5B V1/V2 artifacts. Run these
commands from the project root in a normal macOS Terminal so MLX can access
Metal:

```bash
./venv/bin/python scripts/prepare_qwen3b_mlx.py
./venv/bin/python train/finetune_qwen3b.py
./venv/bin/python export/export_qwen3b_gguf.py
ollama create synapse-qwen2.5-3b-v1 -f Qwen3BSynapseModelfile
```

Training reads `data/mixed_dataset_v2.jsonl`, creates the MLX-required files at
`data/qwen3b/train.jsonl` and `data/qwen3b/valid.jsonl`, and saves adapters to
`models/qwen2.5-3b-synapse-lora-v1`. The explicit configuration is in
`train/configs/qwen3b_lora.yaml`.

### Qwen2.5 7B QLoRA pipeline

The 7B pipeline is also isolated, but it is significantly heavier than the 3B
setup. The files are ready if you want to try it on this machine with a more
conservative LoRA config:

```bash
./venv/bin/python scripts/prepare_qwen7b_mlx.py
./venv/bin/python train/finetune_qwen7b.py
./venv/bin/python export/export_qwen7b_gguf.py
ollama create synapse-qwen2.5-7b-v1 -f Qwen7BSynapseModelfile
```

Training reads `data/mixed_dataset_v2.jsonl`, creates `data/qwen7b/train.jsonl`
and `data/qwen7b/valid.jsonl`, and saves adapters to
`models/qwen2.5-7b-synapse-lora-v1`. The explicit configuration is in
`train/configs/qwen7b_lora.yaml`.

Run from:

`/Users/zeeshanwaheed/Documents/Projects/Python/slm-finetune`

### 1) Build datasets

```bash
./venv/bin/python dataset/prepare.py
./venv/bin/python dataset/augment_website_pairs.py
./venv/bin/python dataset/mix_datasets.py
```

Optional sanity check:

```bash
wc -l data/dataset.json data/dataset_augmented.jsonl data/mixed_dataset.jsonl
```

### 2) Clean retrain (recommended)

```bash
rm -rf models/qwen1.5b-finetuned
rm -rf models/qwen1.5b-merged
rm -f models/gguf/synapse-qwen1.5b-f16.gguf models/gguf/synapse-qwen1.5b-q4_k_m.gguf
```

Train:

```bash
./venv/bin/python train/finetune.py
```

### 3) Export model

Run export helper:

```bash
./venv/bin/python export/export_gguf.py
```

For Qwen2 models, MLX GGUF export may fail with:

`Model type qwen2 not supported for GGUF conversion.`

In that case, use `llama.cpp` conversion:

```bash
python3 convert_hf_to_gguf.py \
  /Users/zeeshanwaheed/Documents/Projects/Python/slm-finetune/models/qwen1.5b-merged \
  --outfile /Users/zeeshanwaheed/Documents/Projects/Python/slm-finetune/models/gguf/synapse-qwen1.5b-f16.gguf \
  --outtype f16
```

Then quantize:

```bash
./build/bin/llama-quantize \
  /Users/zeeshanwaheed/Documents/Projects/Python/slm-finetune/models/gguf/synapse-qwen1.5b-f16.gguf \
  /Users/zeeshanwaheed/Documents/Projects/Python/slm-finetune/models/gguf/synapse-qwen1.5b-q4_k_m.gguf \
  Q4_K_M
```

### 4) Create Ollama models

Create `Modelfile` (fine-tuned):

```text
FROM /Users/zeeshanwaheed/Documents/Projects/Python/slm-finetune/models/gguf/synapse-qwen1.5b-q4_k_m.gguf
SYSTEM You are a helpful assistant for Synapse Tech Inc. Answer clearly and stay grounded in company information when relevant.
PARAMETER temperature 0.7
```

Create `BaseModelfile`:

```text
FROM qwen2.5:1.5b-instruct
SYSTEM You are a helpful assistant.
PARAMETER temperature 0.7
```

Build tags:

```bash
ollama pull qwen2.5:1.5b-instruct
ollama create synapse-1.5b-v1 -f Modelfile
ollama create qwen-base -f BaseModelfile
ollama list
```

### 5) Run side-by-side comparison (`chat/chat.py`)

The chat UI compares two columns:

- **Fine-tuned column:** for most questions, answers go through the local knowledge base (FAISS retrieval + grounded generation in `scripts/answer_with_kb.py`). Small-talk and policy shortcuts skip retrieval and use `guardrail_stage1.run_with_retry` only for that path.
- **Base column:** a single Ollama chat completion (`BASE_MODEL` in `chat/chat.py`, default `qwen-base`) with **no** KB and **no** RAG.

Prerequisites:

- **Ollama** running with `qwen-base`, `synapse-1.5b-v1`, and, after the new training/export, `synapse-1.5b-v2`.
- **Embedding + generation models** pulled in Ollama (whatever `answer_with_kb` / `query_kb` use for your index, often `nomic-embed-text` plus your Synapse generator tag).
- **KB index on disk** (defaults under `data/knowledge-base/`, see `DEFAULT_INDEX_PATH` / `DEFAULT_META_PATH` in `scripts/answer_with_kb.py`).

Optional env:

- **`KB_ANSWER_MODEL`** — Ollama tag used for KB-grounded answers when set; otherwise the bridge falls back to `guardrail_stage1.GENERATOR_MODEL` (see `chat/kb_answer.py`).
- **`OLLAMA_URL`** — if your Ollama API is not the default.

Run (from repo root):

```bash
cd chat && ../venv/bin/python chat.py
```

In the UI, type **`debug`** to print attempt metadata and a **`fine_tuned_path`** hint (`kb_rag`, `small_talk_no_kb`, etc.). Type **`batch`** to paste multiple lines (then **`END`**) and run the same pipeline for each question.

Regression for the KB stack (not the full interactive UI):

```bash
./venv/bin/python scripts/run_kb_gold_eval.py
```

## Suggested runtime tuning for cleaner outputs

In `chat/chat.py`, if answers are too long/repetitive:

- set `MAX_TOKENS = 120`
- set `TEMPERATURE = 0.3`

## Artifacts to use in iOS

Primary file for on-device usage:

- `models/gguf/synapse-qwen1.5b-q4_k_m.gguf`

Reference high-precision file:

- `models/gguf/synapse-qwen1.5b-f16.gguf`

## Context-aware messaging experiment backup

On July 21, 2026, the uncommitted context-aware messaging experiment was saved as a patch before cleaning the branch.

Repo copy:

```text
backups/context_aware_messaging/context_aware_messaging_uncommitted_2026-07-21.patch
```

Temporary copy outside the repo:

```text
/private/tmp/context_aware_messaging_uncommitted_2026-07-21.patch
```

If the working tree still has those changes and you want to discard them:

```bash
git restore app.py chat/kb_answer.py scripts/answer_with_kb.py eval/results/batch_eval_history.json logs/kb_answer_debug.jsonl
```

If you want to restore the experiment later:

```bash
git apply /private/tmp/context_aware_messaging_uncommitted_2026-07-21.patch
```
