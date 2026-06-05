#!/usr/bin/env python3
"""
KB-backed answer bridge for the chat UI (Task 2).

Runs the same pipeline as scripts/answer_with_kb.py:
  check_ollama -> retrieve_hits -> select_context_hits -> generate_grounded_answer

Generation model resolution:
  1. explicit `generation_model` argument
  2. env KB_ANSWER_MODEL
  3. guardrail_stage1.GENERATOR_MODEL (fine-tuned tag, e.g. synapse-qwen1.5b-v5)

Run from repo root (smoke test):
  ./venv/bin/python chat/kb_answer.py "what is opira ai"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

_CHAT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CHAT_DIR.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import answer_with_kb as aw  # noqa: E402
import query_kb  # noqa: E402

from guardrail_stage1 import GENERATOR_MODEL  # noqa: E402


def resolve_ollama_url(ollama_url: str | None) -> str:
    if ollama_url:
        return ollama_url.strip()
    return os.environ.get("OLLAMA_URL", aw.DEFAULT_OLLAMA_URL)


def resolve_generation_model(generation_model: str | None) -> str:
    if generation_model:
        return generation_model.strip()
    return os.environ.get("KB_ANSWER_MODEL", GENERATOR_MODEL).strip()


def kb_grounded_answer(
    question: str,
    *,
    ollama_url: str | None = None,
    generation_model: str | None = None,
    rewrite_model: str | None = None,
    embed_model: str | None = None,
    top_k: int = 5,
    context_k: int = 3,
    temperature: float = 0.1,
    num_predict: int = 120,
    page_types: set[str] | None = None,
) -> str:
    """
    Return a single grounded answer string using the local FAISS KB and Ollama.

    Deterministic paths (contact, catalog summaries, refusals) may not call the
    generation model. When the LLM is used, `model` is the resolved generation tag.
    """
    q = question.strip()
    if not q:
        return aw.DEFAULT_FALLBACK_RESPONSE

    base_url = resolve_ollama_url(ollama_url)
    model = resolve_generation_model(generation_model)

    query_kb.check_ollama(base_url)

    hits, _embedding_model, profile = aw.retrieve_hits(
        question=q,
        index_path=aw.DEFAULT_INDEX_PATH,
        metadata_path=aw.DEFAULT_META_PATH,
        manifest_path=aw.DEFAULT_MANIFEST_PATH,
        embed_model=embed_model,
        ollama_url=base_url,
        rewrite_model=rewrite_model,
        page_types=page_types or set(),
        top_k=top_k,
    )
    eff_context_k = max(context_k, 4) if profile.get("company_overview") else context_k
    eff_context_k = min(eff_context_k, len(hits)) if hits else eff_context_k
    context_hits = aw.select_context_hits(hits, context_k=eff_context_k, profile=profile)
    return aw.generate_grounded_answer(
        question=q,
        context_hits=context_hits,
        profile=profile,
        ollama_url=base_url,
        model=model,
        temperature=temperature,
        num_predict=num_predict,
    )


def kb_grounded_answer_with_meta(
    question: str,
    *,
    ollama_url: str | None = None,
    generation_model: str | None = None,
    rewrite_model: str | None = None,
    embed_model: str | None = None,
    top_k: int = 5,
    context_k: int = 3,
    temperature: float = 0.1,
    num_predict: int = 120,
    page_types: set[str] | None = None,
) -> dict[str, Any]:
    """Same as kb_grounded_answer but includes embedding model name and resolved generator tag."""
    q = question.strip()
    base_url = resolve_ollama_url(ollama_url)
    model = resolve_generation_model(generation_model)

    if not q:
        return {
            "answer": aw.DEFAULT_FALLBACK_RESPONSE,
            "generation_model": model,
            "embedding_model": None,
        }

    query_kb.check_ollama(base_url)

    hits, embedding_model, profile = aw.retrieve_hits(
        question=q,
        index_path=aw.DEFAULT_INDEX_PATH,
        metadata_path=aw.DEFAULT_META_PATH,
        manifest_path=aw.DEFAULT_MANIFEST_PATH,
        embed_model=embed_model,
        ollama_url=base_url,
        rewrite_model=rewrite_model,
        page_types=page_types or set(),
        top_k=top_k,
    )
    eff_context_k = max(context_k, 4) if profile.get("company_overview") else context_k
    eff_context_k = min(eff_context_k, len(hits)) if hits else eff_context_k
    context_hits = aw.select_context_hits(hits, context_k=eff_context_k, profile=profile)
    answer = aw.generate_grounded_answer(
        question=q,
        context_hits=context_hits,
        profile=profile,
        ollama_url=base_url,
        model=model,
        temperature=temperature,
        num_predict=num_predict,
    )
    return {
        "answer": answer,
        "generation_model": model,
        "embedding_model": embedding_model,
        "rewrite": profile.get("rewrite"),
        "usage": profile.get("usage", {}),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test: KB grounded answer (one question).")
    p.add_argument("question", nargs="+", help="Question text")
    p.add_argument("--model", help="Override generation model (default: KB_ANSWER_MODEL or GENERATOR_MODEL)")
    p.add_argument("--ollama-url", help="Override Ollama base URL")
    p.add_argument("--json", action="store_true", help="Print JSON with answer + model metadata")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    q = " ".join(args.question).strip()
    if args.json:
        import json

        out = kb_grounded_answer_with_meta(
            q,
            ollama_url=args.ollama_url,
            generation_model=args.model,
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        ans = kb_grounded_answer(
            q,
            ollama_url=args.ollama_url,
            generation_model=args.model,
        )
        print(ans)


if __name__ == "__main__":
    main()
