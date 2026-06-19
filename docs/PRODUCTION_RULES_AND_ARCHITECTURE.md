# Synapse Local Model Rules and Architecture

This document is the working production reference for the local Synapse model stack. It captures the current architecture, operating rules, model strategy, evaluation gates, and optimization roadmap.

## 1. Current Production Position

The project currently has two strongest practical paths:

1. `Synapse Qwen 2.5 1.5B` with `Base + RAG`
2. `Synapse Llama V1 3B` with `Fine-tuned + RAG`

Use them differently:

- `Synapse Qwen 2.5 1.5B` is the primary speed-and-balance deployment option.
- `Synapse Llama V1 3B` is the stronger fine-tuned middleweight option when better tuned behavior is worth the extra latency.

Current guidance:

- Default production baseline: `Synapse Qwen 2.5 1.5B | Base + RAG`
- Stronger tuned option: `Synapse Llama V1 3B | Fine-tuned + RAG`
- Quality-first heavy options: `Qwen2.5 7B | Base + RAG` and `Synapse Llama V1 8B | Base + RAG`

## 2. System Architecture

The current system is a layered local inference stack:

1. User question enters the Streamlit app.
2. The app chooses an answer mode:
   - `Base`
   - `Base + RAG`
   - `Fine-tuned`
   - `Fine-tuned + RAG`
3. For RAG paths, the pipeline:
   - optionally applies combined planning
   - retrieves knowledge-base context
   - generates the answer with the selected model
4. The answer is scored by the batch eval pipeline.
5. The result is saved into history with metrics, timing, and issue counts.

Main parts of the repo:

- [app.py](/Users/zeeshanwaheed/Documents/Projects/Python/slm-finetune/app.py): main Streamlit UI, eval workflow, history, automation
- [chat/kb_answer.py](/Users/zeeshanwaheed/Documents/Projects/Python/slm-finetune/chat/kb_answer.py): KB-grounded answer path
- [scripts/answer_with_kb.py](/Users/zeeshanwaheed/Documents/Projects/Python/slm-finetune/scripts/answer_with_kb.py): retrieval/generation logic
- [chat/guardrail_stage1.py](/Users/zeeshanwaheed/Documents/Projects/Python/slm-finetune/chat/guardrail_stage1.py): guardrail and route behavior
- `train/` and `export/`: fine-tuning and GGUF export flows
- `ollama/modelfiles/` and root Modelfiles: Ollama registration layer

## 3. Operating Rules

These are the current practical rules for how the system should be used.

### Model rules

- Treat `Base + RAG` as the default benchmark.
- Do not assume a fine-tuned model is better than the base version.
- Promote a fine-tune only if it beats the active baseline on eval.
- Keep one fast deployment path and one stronger tuned path.

### Deployment rules

- Small/fast production path: `Synapse Qwen 2.5 1.5B | Base + RAG`
- Tuned production path: `Synapse Llama V1 3B | Fine-tuned + RAG`
- Heavy models are optional quality upgrades, not the default.

### Evaluation rules

- Judge models by saved run configuration, not only family labels.
- Use day-scoped history blocks for comparisons.
- Compare like with like:
  - same question set
  - same RAG mode
  - same temperature
  - same max token limit

### Fine-tuning rules

- Fine-tune only after dataset cleaning.
- Reject fine-tunes that increase:
  - over-refusal
  - repetition
  - product-routing errors
  - unsupported claims
- Fine-tuning should improve a measurable production goal, not just create a new artifact.

## 4. Current Strengths and Weaknesses

### Strongest observed strengths

- `Qwen 1.5B Base + RAG` has the best speed-to-quality balance.
- `Llama 3B Fine-tuned + RAG` is the strongest tuned result so far.
- `Qwen 7B Base + RAG` and `Llama 8B Base + RAG` provide top-end quality when latency is acceptable.

### Main observed failure patterns

- over-refusal
- repetition
- wrong product recommendation
- occasional forbidden claims
- retrieval/generation mismatch where retrieval is decent but final answer degrades

## 5. Production Acceptance Gates

A model should only be promoted if it passes these gates against the current baseline.

### Required gates

- total score must meet or beat the target baseline
- refusals must not regress
- repetition must not increase materially
- product recommendation quality must not regress
- unsupported claims must not increase
- latency must stay inside the allowed target for the intended use case

### Practical gate examples

- fast path gate:
  - keep latency near current `Qwen 1.5B Base + RAG`
  - maintain high pass rate
- tuned path gate:
  - beat the baseline on quality
  - stay comfortably below heavy-model latency

## 6. What Makes This Production-Ready

The remaining work is mostly operational, not just model training.

### Must-have production pieces

1. versioned datasets
2. versioned train configs
3. repeatable eval snapshots
4. clear model promotion/rejection rules
5. retrieval confidence safeguards
6. logging and observability
7. rollback-ready model tags
8. feedback loop from failed prompts

### Nice-to-have production pieces

1. concurrency/load testing
2. route-level dashboards
3. automated regression suites
4. richer per-bucket eval reports

## 7. Retrieval Optimization Roadmap

The next speed and quality gains should come from retrieval and prompt efficiency, not from shrinking answer length.

### Phase 1: Measurement and guardrails

1. measure prompt token size per run
2. re-evaluate after each retrieval change

### Phase 2: Basic context reduction

1. reduce `top_k` passed to the final generator
2. keep retrieval breadth higher than generation breadth
3. shorten the context prompt format

### Phase 3: Chunk quality cleanup

1. shrink chunk size
2. add small overlap between chunks
3. remove noisy text from chunks

### Phase 4: Retrieval result cleanup

1. deduplicate similar retrieved chunks
2. dynamically choose context size
3. add a reranking step before generation

### Phase 5: Smarter retrieval behavior

1. route retrieval by question type
2. tune chunking by content type

### Phase 6: Reuse and caching

1. cache retrieval results for repeated eval questions
2. build preselected context packs for common questions

## 8. Evaluation Workflow

Recommended workflow for any meaningful model or retrieval change:

1. make one phase-aligned batch of changes
2. run the batch eval suite
3. compare:
   - score
   - avg latency
   - refusal count
   - issue counts
   - retrieval confidence metrics
4. keep or revert
5. move to the next phase only if the previous one is stable

Recommended cadence:

- one evaluation check after each group of about 3 closely related tasks
- do not mix unrelated retrieval, prompt, and model changes in the same batch

## 9. Model Strategy Going Forward

### Keep as active production baseline

- `Synapse Qwen 2.5 1.5B | Base + RAG`

### Keep as active tuned contender

- `Synapse Llama V1 3B | Fine-tuned + RAG`

### Keep as quality reference models

- `Qwen2.5 7B | Base + RAG`
- `Synapse Llama V1 8B | Base + RAG`

### Do not prioritize for immediate promotion

- current `Qwen 1.5B` fine-tuned variants
- current `Qwen 3B` fine-tuned variant
- current `Gemma 4B` fine-tuned variant

These are not useless, but they should not be promoted until their failure patterns are fixed.

## 10. Recommended Next Steps

1. freeze the current winners as baselines
2. improve retrieval efficiency before training more models
3. expand the eval set beyond the current narrow benchmark
4. add stricter promotion/rejection gates
5. keep fine-tuning focused on models that already show promise

## 11. Short Version

If this project is used for a real delivery path:

- deploy `Synapse Qwen 2.5 1.5B | Base + RAG` as the safe default
- use `Synapse Llama V1 3B | Fine-tuned + RAG` as the tuned upgrade path
- spend the next round of work on retrieval quality, eval quality, and operational discipline

That is the shortest path from “good local system” to “proper production system.”
