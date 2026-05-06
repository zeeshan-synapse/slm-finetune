# Synapse SLM Fine-Tuning (Current Working Setup)

This project fine-tunes a local model for Synapse Tech website/domain Q&A and compares fine-tuned vs base behavior.

## Current status (working now)

- Working base family: `Qwen2.5-1.5B-Instruct`
- Working fine-tuned output: `synapse-qwen1.5b-q4_k_m.gguf`
- Working Ollama tags:
  - Fine-tuned: `synapse-3b` (tag name kept for compatibility with `chat/chat.py`)
  - Base: `qwen-base`

## Important notes from experiments

- `Llama` 1B and 3B instruct variants were tested earlier and were not giving reliable results for this use case in this setup.
- `Qwen` 7B was too heavy for local MLX training/inference on this machine (Metal OOM).
- `Qwen2.5-1.5B-Instruct` is the stable model choice currently.

## End-to-end workflow

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
ollama create synapse-3b -f Modelfile
ollama create qwen-base -f BaseModelfile
ollama list
```

### 5) Run side-by-side comparison

Make sure `chat/chat.py` uses:

- `FINETUNED_MODEL = "synapse-3b"`
- `BASE_MODEL = "qwen-base"`

Then run:

```bash
./venv/bin/python chat/chat.py
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

