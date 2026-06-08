#!/usr/bin/env python3
"""
Production-style local assistant chat.

This is the clean app path:
  user -> small-talk/policy quick check -> KB/RAG assistant answer

No base-model comparison column is shown here. Keep chat/chat.py for dev eval.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_CHAT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CHAT_DIR.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
for path in (_CHAT_DIR, _SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from guardrail_stage1 import (
    check_ollama,
    is_small_talk_question,
    policy_intent,
    run_with_retry,
)
import answer_with_kb as aw
from kb_answer import kb_grounded_answer_with_meta


def production_answer(question: str) -> dict:
    q = question.strip()
    started = time.perf_counter()

    if not q:
        return {
            "answer": "Please ask a question.",
            "route": "empty_input",
            "elapsed_s": 0.0,
        }

    if is_small_talk_question(q) or policy_intent(q) is not None:
        result = run_with_retry(q)
        observability = {
            "question": q,
            "intent": "small_talk" if is_small_talk_question(q) else "policy",
            "confidence": 1.0,
            "route": "quick_guardrail",
            "rewrite_query": "",
            "sources": [],
            "used_template": True,
            "used_refusal": False,
        }
        aw.log_kb_debug({"stage": "answer_observability", **observability})
        return {
            "answer": result["final_answer"],
            "route": "quick_guardrail",
            "observability": observability,
            "elapsed_s": round(time.perf_counter() - started, 2),
        }

    meta = kb_grounded_answer_with_meta(q)
    return {
        "answer": meta["answer"],
        "route": "kb_rag",
        "observability": meta.get("observability"),
        "usage": meta.get("usage", {}),
        "elapsed_s": round(time.perf_counter() - started, 2),
    }


def main() -> None:
    check_ollama()
    debug = False
    print("=" * 60)
    print("Synapse Assistant — production path")
    print("=" * 60)
    print("Type 'exit' to quit. Type 'debug' to toggle metadata.")

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        if question.lower() == "debug":
            debug = not debug
            print(f"Debug mode: {'ON' if debug else 'OFF'}")
            continue

        result = production_answer(question)
        print("\nAssistant:")
        print(result["answer"])
        if debug:
            print("\n--- Debug ---")
            print(
                json.dumps(
                    {
                        "route": result.get("route"),
                        "observability": result.get("observability"),
                        "usage": result.get("usage"),
                        "elapsed_s": result.get("elapsed_s"),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
