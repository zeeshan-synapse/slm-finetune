#!/usr/bin/env python3
"""
KB-backed answer bridge for the chat UI (Task 2).

Runs the same pipeline as scripts/answer_with_kb.py:
  check_ollama -> retrieve_hits -> select_context_hits -> generate_grounded_answer

Generation model resolution:
  1. explicit `generation_model` argument
  2. env KB_ANSWER_MODEL
  3. scripts.answer_with_kb.DEFAULT_GENERATION_MODEL

Run from repo root (smoke test):
  ./venv/bin/python chat/kb_answer.py "what is opira ai"
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

_CHAT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CHAT_DIR.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import answer_with_kb as aw  # noqa: E402
from kb_paths import knowledge_base_dir  # noqa: E402
import query_kb  # noqa: E402

_CHECKED_OLLAMA_URLS: set[str] = set()


def resolve_ollama_url(ollama_url: str | None) -> str:
    if ollama_url:
        return ollama_url.strip()
    return os.environ.get("OLLAMA_URL", aw.DEFAULT_OLLAMA_URL)


def resolve_generation_model(generation_model: str | None) -> str:
    if generation_model:
        return generation_model.strip()
    return os.environ.get("KB_ANSWER_MODEL", aw.DEFAULT_GENERATION_MODEL).strip()


def check_ollama_once(base_url: str) -> None:
    if base_url in _CHECKED_OLLAMA_URLS:
        return
    query_kb.check_ollama(base_url)
    _CHECKED_OLLAMA_URLS.add(base_url)


def resolve_kb_paths(knowledge_domain: str | None) -> tuple[Path, Path, Path]:
    domain = (knowledge_domain or os.environ.get("KB_DOMAIN") or "synapse").strip() or "synapse"
    kb_dir = knowledge_base_dir(domain)
    return (
        kb_dir / "faiss.index",
        kb_dir / "index_meta.jsonl",
        kb_dir / "index_manifest.json",
    )


def resolve_knowledge_domain(knowledge_domain: str | None) -> str:
    return (knowledge_domain or os.environ.get("KB_DOMAIN") or "synapse").strip() or "synapse"


def fallback_response_for_domain(knowledge_domain: str) -> str:
    if knowledge_domain.lower() == "biek":
        return (
            "This detail is not clearly confirmed in the available BIEK information. "
            "Please verify it on the official BIEK website."
        )
    return aw.DEFAULT_FALLBACK_RESPONSE


def adapt_answer_for_domain(answer: str, knowledge_domain: str) -> str:
    if knowledge_domain.lower() != "biek":
        return answer

    biek_fallback = fallback_response_for_domain("biek")
    if answer.strip() == aw.DEFAULT_FALLBACK_RESPONSE:
        return biek_fallback

    adjusted = answer.replace("Please verify with Synapse Tech.", "Please verify it on the official BIEK website.")
    adjusted = adjusted.replace("contact Synapse Tech directly", "check the official BIEK website directly")
    adjusted = adjusted.replace("Contact Synapse Tech directly", "Check the official BIEK website directly")
    return adjusted


def kb_grounded_answer(
    question: str,
    *,
    ollama_url: str | None = None,
    knowledge_domain: str | None = None,
    generation_model: str | None = None,
    rewrite_model: str | None = None,
    classifier_model: str | None = None,
    embed_model: str | None = None,
    top_k: int | None = None,
    context_k: int | None = None,
    temperature: float | None = None,
    num_predict: int | None = None,
    page_types: set[str] | None = None,
) -> str:
    """
    Return a single grounded answer string using the local FAISS KB and Ollama.

    Deterministic paths (contact, catalog summaries, refusals) may not call the
    generation model. When the LLM is used, `model` is the resolved generation tag.
    """
    q = question.strip()
    domain = resolve_knowledge_domain(knowledge_domain)
    if not q:
        return fallback_response_for_domain(domain)

    base_url = resolve_ollama_url(ollama_url)
    model = resolve_generation_model(generation_model)
    model_profile = aw.get_model_profile(model)
    effective_top_k = top_k if top_k is not None else int(model_profile["top_k"])
    effective_context_k = context_k if context_k is not None else int(model_profile["context_k"])
    effective_temperature = (
        temperature if temperature is not None else float(model_profile["temperature"])
    )
    effective_num_predict = (
        num_predict if num_predict is not None else int(model_profile["num_predict"])
    )
    index_path, meta_path, manifest_path = resolve_kb_paths(domain)

    check_ollama_once(base_url)

    hits, _embedding_model, profile = aw.retrieve_hits(
        question=q,
        index_path=index_path,
        metadata_path=meta_path,
        manifest_path=manifest_path,
        embed_model=embed_model,
        ollama_url=base_url,
        rewrite_model=rewrite_model,
        classifier_model=classifier_model,
        page_types=page_types or set(),
        top_k=effective_top_k,
    )
    eff_context_k = (
        max(effective_context_k, 3)
        if profile.get("company_overview")
        else effective_context_k
    )
    eff_context_k = min(eff_context_k, len(hits)) if hits else eff_context_k
    context_hits = aw.select_context_hits(hits, context_k=eff_context_k, profile=profile)
    answer = aw.generate_grounded_answer(
        question=q,
        context_hits=context_hits,
        profile=profile,
        ollama_url=base_url,
        model=model,
        temperature=effective_temperature,
        num_predict=effective_num_predict,
    )
    return adapt_answer_for_domain(answer, domain)


def kb_grounded_answer_with_meta(
    question: str,
    *,
    ollama_url: str | None = None,
    knowledge_domain: str | None = None,
    generation_model: str | None = None,
    rewrite_model: str | None = None,
    classifier_model: str | None = None,
    embed_model: str | None = None,
    top_k: int | None = None,
    context_k: int | None = None,
    temperature: float | None = None,
    num_predict: int | None = None,
    page_types: set[str] | None = None,
) -> dict[str, Any]:
    """Same as kb_grounded_answer but includes embedding model name and resolved generator tag."""
    q = question.strip()
    domain = resolve_knowledge_domain(knowledge_domain)
    base_url = resolve_ollama_url(ollama_url)
    model = resolve_generation_model(generation_model)
    model_profile = aw.get_model_profile(model)
    effective_top_k = top_k if top_k is not None else int(model_profile["top_k"])
    effective_context_k = context_k if context_k is not None else int(model_profile["context_k"])
    effective_temperature = (
        temperature if temperature is not None else float(model_profile["temperature"])
    )
    effective_num_predict = (
        num_predict if num_predict is not None else int(model_profile["num_predict"])
    )
    index_path, meta_path, manifest_path = resolve_kb_paths(domain)

    if not q:
        return {
            "answer": fallback_response_for_domain(domain),
            "generation_model": model,
            "embedding_model": None,
            "knowledge_domain": domain,
        }

    check_ollama_once(base_url)

    total_started = time.perf_counter()
    hits, embedding_model, profile = aw.retrieve_hits(
        question=q,
        index_path=index_path,
        metadata_path=meta_path,
        manifest_path=manifest_path,
        embed_model=embed_model,
        ollama_url=base_url,
        rewrite_model=rewrite_model,
        classifier_model=classifier_model,
        page_types=page_types or set(),
        top_k=effective_top_k,
    )
    timings = profile.setdefault("timings", {})
    eff_context_k = (
        max(effective_context_k, 3)
        if profile.get("company_overview")
        else effective_context_k
    )
    eff_context_k = min(eff_context_k, len(hits)) if hits else eff_context_k
    stage_started = time.perf_counter()
    context_hits = aw.select_context_hits(hits, context_k=eff_context_k, profile=profile)
    timings["context_select_s"] = round(time.perf_counter() - stage_started, 4)
    stage_started = time.perf_counter()
    answer = aw.generate_grounded_answer(
        question=q,
        context_hits=context_hits,
        profile=profile,
        ollama_url=base_url,
        model=model,
        temperature=effective_temperature,
        num_predict=effective_num_predict,
    )
    answer = adapt_answer_for_domain(answer, domain)
    timings["answer_total_s"] = round(time.perf_counter() - stage_started, 4)
    timings["rag_total_s"] = round(time.perf_counter() - total_started, 4)
    if profile.get("observability") is not None:
        profile["observability"]["timings"] = timings
    return {
        "answer": answer,
        "generation_model": model,
        "embedding_model": embedding_model,
        "knowledge_domain": domain,
        "rewrite": profile.get("rewrite"),
        "classification": profile.get("model_classification"),
        "answer_policy": profile.get("answer_policy"),
        "observability": profile.get("observability"),
        "usage": profile.get("usage", {}),
        "timings": timings,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test: KB grounded answer (one question).")
    p.add_argument("question", nargs="+", help="Question text")
    p.add_argument("--model", help="Override generation model (default: KB_ANSWER_MODEL or answer_with_kb default)")
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
