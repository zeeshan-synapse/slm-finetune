# Synapse Assistant: Three Inference Paths

## Purpose

`chat/chat.py` is a development comparison tool with four selectable answer
paths. It is not the production UI.

At startup it offers:

1. Synapse 1.5B V1 + RAG
2. Synapse 1.5B V2 + RAG
3. Base + RAG
4. Plain base
5. All answers

The selection controls which model calls run. Choosing one path does not also
run the other two.

## Models

- Existing fine-tuned generator: `synapse-1.5b-v1`.
- New dataset fine-tuned generator: `synapse-1.5b-v2`.
- Base generator: `qwen-base`, with `qwen2.5:1.5b-instruct` as the plain-base
  fallback.
- Embedding model: read from `data/knowledge-base/index_manifest.json`.
- Rewrite model: resolved by `KB_REWRITE_MODEL`, `KB_BASE_MODEL`, or the answer
  model.
- Classifier model: resolved by `KB_CLASSIFIER_MODEL`, `KB_BASE_MODEL`,
  `KB_REWRITE_MODEL`, or `qwen-base`.

All model inference currently runs through the local Ollama API.

## Path 1: Fine-Tuned + RAG

Entry point:

```text
chat/chat.py
-> fine_tuned_result(question)
-> chat/kb_answer.py::kb_grounded_answer_with_meta(...)
-> scripts/answer_with_kb.py
```

Generation model:

```text
synapse-1.5b-v1 or synapse-1.5b-v2
```

Flow:

```text
user question
-> small-talk/policy quick check
-> local-model query rewrite
-> intent classification
-> query embedding
-> FAISS search
-> retrieval augmentation/reranking
-> context selection
-> final prompt containing original question + evidence
-> the selected Synapse V1/V2 model generates the answer
-> answer checks/retries
-> final response + observability metadata
```

Small talk and policy questions can bypass RAG through
`chat/guardrail_stage1.py`.

Current finding: the fine-tuned model frequently copies marketing text,
repeats content, leaks `Answer:` artifacts, and produces long or truncated
answers. The same retrieved evidence is handled better by the base model. This
suggests a fine-tuning/data-format problem rather than a fundamental FAISS
failure.

## Path 2: Base + RAG

Entry point:

```text
chat/chat.py
-> base_rag_result(question)
-> kb_grounded_answer_with_meta(
     question,
     generation_model="qwen-base"
   )
-> the same scripts/answer_with_kb.py pipeline
```

This path intentionally uses the same:

- Query rewrite
- Intent classification
- Embedding model
- FAISS index
- Retrieved chunks
- Context selection
- Prompt construction
- Safety handling
- Validators and debug logging

The only intended difference from Fine-tuned + RAG is the final generation
model:

```text
Fine-tuned V1 + RAG -> synapse-1.5b-v1
Fine-tuned V2 + RAG -> synapse-1.5b-v2
Base + RAG          -> qwen-base
```

This makes it the most useful diagnostic comparison. If Base + RAG answers
well while Fine-tuned + RAG fails on the same question, the likely problem is
the fine-tuned model or its training data.

Current finding: Base + RAG is the strongest demo path. It understands
paraphrases and produces natural Synapse-specific answers. It can still repeat
unsupported marketing claims or percentages present in noisy website chunks,
so it is a working prototype rather than a fully hardened production system.

## Path 3: Plain Base

Entry point:

```text
chat/chat.py
-> generate_base_result(question)
-> generate_model_result("qwen-base", question)
-> Ollama /api/chat
```

Flow:

```text
user question
-> base model system prompt
-> qwen-base generation
-> response
```

This path does not use:

- Query rewriting
- Embeddings
- FAISS
- Synapse KB chunks
- Product cards
- RAG grounding

It measures the base model's general conversational ability. It is usually
fluent, but it may invent Synapse products, industries, contact information,
or capabilities because it has no authoritative company context.

## Experimental Generation Flag

Run with:

```bash
RAG_GENERATE_ORDINARY_ANSWERS=1 python chat/chat.py
```

When enabled, ordinary RAG questions bypass canned final-answer templates and
reach model generation. The experimental final answer call:

- Uses the original question and retrieved evidence.
- Returns natural text rather than JSON.
- Uses temperature `0.3`.
- Records `generation_experiment: true`.

Deterministic behavior remains for:

- Exact contact handling
- Purchase/get-started handling
- Unsupported high-risk claims such as pricing, legal terms, compliance,
  certifications, SLA, roadmap, and guarantees
- Empty-KB fallback

Without this flag, the older deterministic/template-heavy behavior remains.

## Recommendation Question Handling

Questions such as:

```text
I need recruitment automation. Do you have a product for that?
Which product helps with customer support?
```

are detected by:

```text
is_single_product_recommendation_question(...)
```

The system retrieves broadly, groups product chunks by `doc_id`, selects the
highest-scoring product document, and gives only that product's chunks to the
generator. The model still writes the final answer; the code does not return a
canned recommendation.

This was added to prevent unrelated full-catalog answers when one product is
the closest fit.

## Knowledge Base

Important artifacts:

```text
data/knowledge-base/faiss.index
data/knowledge-base/index_meta.jsonl
data/knowledge-base/index_manifest.json
```

Runtime product metadata:

```text
data/runtime/product_cards.json
data/runtime/policies.json
```

Product cards stabilize product names, aliases, and short scope descriptions.
They should be treated as structured context, not as a replacement for model
generation in the experimental path.

## Main Code Responsibilities

### `chat/chat.py`

- Development comparison UI
- Startup mode selection
- Runs only selected model paths
- Prints answers and optional debug metadata
- Supports interactive and batch testing

### `chat/kb_answer.py`

- Bridge between chat UI and the shared KB pipeline
- Resolves the generation model
- Calls retrieval, context selection, and answer generation
- Returns answer plus model/retrieval metadata

### `scripts/query_kb.py`

- Query normalization/profile hints
- Query embedding
- FAISS search
- Retrieval scoring and reranking helpers

### `scripts/answer_with_kb.py`

- Query rewriting and intent classification
- Retrieval orchestration
- Context selection
- Product recommendation isolation
- Prompt construction
- Final model generation
- Safety policies, retries, validation, and observability

### `chat/assistant.py`

- Cleaner production-style single-answer entry point
- Does not show comparison columns
- Currently uses the normal KB-backed assistant path

## Debug Interpretation

For an ordinary experimental RAG answer, expect:

```json
{
  "route": "rag_generation",
  "used_template": false,
  "used_refusal": false,
  "generation_experiment": true,
  "sources": ["relevant-doc-id"],
  "token_usage": {
    "answer_generation": {
      "model": "qwen-base, synapse-1.5b-v1, or synapse-1.5b-v2",
      "input_tokens": 0,
      "output_tokens": 0,
      "total_tokens": 0
    }
  }
}
```

Interpretation:

- `rag_generation`: final prose came from a model call.
- `used_template: false`: no canned final answer was returned.
- `sources`: KB documents used as evidence.
- `answer_generation.model`: identifies the actual final generator.
- `answer_generation.output_tokens > 0`: confirms generation occurred.

Safety/contact paths may correctly report `hard_policy` and
`used_template: true`.

## Commands

Activate the environment:

```bash
cd /Users/zeeshanwaheed/Documents/Projects/Python/slm-finetune
source venv/bin/activate
```

Start Ollama in another terminal if required:

```bash
ollama serve
```

Run the conversational-generation experiment:

```bash
RAG_GENERATE_ORDINARY_ANSWERS=1 python chat/chat.py
```

Run the stable/default behavior:

```bash
python chat/chat.py
```

Run the production-style single assistant:

```bash
python chat/assistant.py
```

## Current Engineering Conclusion

The strongest current architecture is:

```text
base instruct SLM
-> local query rewrite/classification
-> local FAISS RAG
-> model-generated grounded response
-> lightweight claim/safety validation
```

RAG should hold company facts because those facts can change without
retraining. Fine-tuning should primarily improve stable behavior, tone,
formatting, terminology, and evidence use.

The current `synapse-1.5b-v1` fine-tune should be audited before production use.
Likely areas to inspect include malformed chat templates, duplicate or
repetitive completions, copied marketing text, `Answer:` artifacts, empty
assistant turns, and overly long training responses. A future clean LoRA can
be evaluated by comparing it against Base + RAG using the same pipeline.
