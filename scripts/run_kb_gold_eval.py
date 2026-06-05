#!/usr/bin/env python3
"""
Run the kb_gold_v1 regression set against scripts/answer_with_kb.py behavior.

Requires: Ollama running, embedding model + generation model available,
          data/knowledge-base/faiss.index and metadata built.

Usage (from repo root):
  ./venv/bin/python scripts/run_kb_gold_eval.py
  ./venv/bin/python scripts/run_kb_gold_eval.py --json-out eval/results/kb_gold_v1_last.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Import answer_with_kb from the scripts/ directory
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import answer_with_kb as aw  # noqa: E402
import query_kb  # noqa: E402


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_GOLD_PATH = PROJECT_DIR / "eval" / "prompts" / "kb_gold_v1.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run kb_gold_v1 KB regression eval.")
    p.add_argument(
        "--gold",
        default=str(DEFAULT_GOLD_PATH),
        help="Path to kb_gold_v1.json",
    )
    p.add_argument(
        "--model",
        default=aw.DEFAULT_GENERATION_MODEL,
        help="Ollama generation model (same as answer_with_kb --model).",
    )
    p.add_argument(
        "--ollama-url",
        default=aw.DEFAULT_OLLAMA_URL,
        help="Ollama base URL.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Retrieval top-k (match answer_with_kb default).",
    )
    p.add_argument(
        "--context-k",
        type=int,
        default=3,
        help="Context chunks (match answer_with_kb default).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Generation temperature for non-deterministic answers.",
    )
    p.add_argument(
        "--num-predict",
        type=int,
        default=120,
        help="Max tokens for non-deterministic answers.",
    )
    p.add_argument(
        "--json-out",
        help="Write full results JSON to this path.",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Only print summary line and failures.",
    )
    return p.parse_args()


def normalize(s: str) -> str:
    return " ".join(s.lower().split())


def is_refusal(answer: str) -> bool:
    a = answer.strip()
    if a == aw.DEFAULT_FALLBACK_RESPONSE:
        return True
    low = normalize(a)
    return "not confirmed" in low and "verify" in low


def check_answer_case(answer: str, case: dict[str, Any]) -> tuple[bool, str]:
    if is_refusal(answer):
        return False, "expected grounded answer but got refusal or empty"

    low = normalize(answer)
    for needle in case.get("contains_all") or []:
        if normalize(needle) not in low:
            return False, f"missing required substring (case-insensitive): {needle!r}"

    for needle in case.get("contains_any") or []:
        if normalize(needle) in low:
            break
    else:
        if case.get("contains_any"):
            return False, f"expected at least one of: {case['contains_any']}"

    return True, ""


def check_refuse_case(answer: str) -> tuple[bool, str]:
    if is_refusal(answer):
        return True, ""
    preview = answer.strip()[:200]
    return False, f"expected refusal fallback, got: {preview!r}"


def run_one(
    case: dict[str, Any],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    question = case["question"].strip()
    query_kb.check_ollama(args.ollama_url)
    hits, embedding_model, profile = aw.retrieve_hits(
        question=question,
        index_path=aw.DEFAULT_INDEX_PATH,
        metadata_path=aw.DEFAULT_META_PATH,
        manifest_path=aw.DEFAULT_MANIFEST_PATH,
        embed_model=None,
        ollama_url=args.ollama_url,
        rewrite_model=args.model,
        page_types=set(),
        top_k=args.top_k,
    )
    context_hits = aw.select_context_hits(
        hits, context_k=args.context_k, profile=profile
    )
    answer = aw.generate_grounded_answer(
        question=question,
        context_hits=context_hits,
        profile=profile,
        ollama_url=args.ollama_url,
        model=args.model,
        temperature=args.temperature,
        num_predict=args.num_predict,
    )

    expect = case["expect"]
    if expect == "refuse":
        ok, reason = check_refuse_case(answer)
    elif expect == "answer":
        ok, reason = check_answer_case(answer, case)
    else:
        ok, reason = False, f"unknown expect: {expect!r}"

    return {
        "id": case["id"],
        "category": case.get("category"),
        "question": question,
        "expect": expect,
        "pass": ok,
        "reason": reason,
        "answer": answer,
        "embedding_model": embedding_model,
    }


def main() -> int:
    args = parse_args()
    gold_path = Path(args.gold)
    if not gold_path.is_file():
        print(f"ERROR: gold file not found: {gold_path}", file=sys.stderr)
        return 2

    payload = json.loads(gold_path.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = payload.get("cases") or []
    if not cases:
        print("ERROR: no cases in gold file", file=sys.stderr)
        return 2

    results: list[dict[str, Any]] = []
    failures = 0

    for case in cases:
        try:
            row = run_one(case, args=args)
        except Exception as exc:
            row = {
                "id": case.get("id"),
                "category": case.get("category"),
                "question": case.get("question"),
                "expect": case.get("expect"),
                "pass": False,
                "reason": f"exception: {exc}",
                "answer": "",
                "embedding_model": None,
            }
        results.append(row)
        if not row["pass"]:
            failures += 1
            if not args.quiet:
                print(
                    f"FAIL {row['id']}: {row['question']}\n"
                    f"  reason: {row['reason']}\n"
                    f"  answer: {row['answer'][:300]}{'...' if len(row['answer']) > 300 else ''}\n"
                )
            if args.fail_fast:
                break
        elif not args.quiet:
            print(f"PASS {row['id']}: {row['question']}")

    passed = sum(1 for r in results if r["pass"])
    total = len(results)
    print(
        f"\nkb_gold_v1: {passed}/{total} passed, {failures} failed "
        f"(model={args.model})"
    )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "run_at": datetime.now(timezone.utc).isoformat(),
                    "gold_path": str(gold_path),
                    "model": args.model,
                    "ollama_url": args.ollama_url,
                    "passed": passed,
                    "total": total,
                    "failures": failures,
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote: {out_path}")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
