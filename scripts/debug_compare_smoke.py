#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
CHAT_DIR = PROJECT_DIR / "chat"

if str(CHAT_DIR) not in sys.path:
    sys.path.insert(0, str(CHAT_DIR))

import chat as chat_app
import guardrail_stage1 as guardrail
from smoke_eval import SMOKE_PROMPTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 5 smoke prompts through raw FT, guardrailed FT, and base model paths."
    )
    parser.add_argument("--model", required=True, help="Fine-tuned Ollama model tag.")
    parser.add_argument(
        "--base-model",
        default=chat_app.BASE_MODEL,
        help="Base model tag used for comparison.",
    )
    parser.add_argument(
        "--base-model-fallback",
        default=chat_app.BASE_MODEL_FALLBACK,
        help="Fallback base model tag if the primary base model is unavailable.",
    )
    parser.add_argument(
        "--output",
        help="Optional text file path to save the full debug comparison output.",
    )
    return parser.parse_args()


def emit(lines: list[str], text: str = "") -> None:
    print(text)
    lines.append(text)


def print_answer_block(lines: list[str], title: str, answer: str) -> None:
    emit(lines, "=" * 60)
    emit(lines, title)
    emit(lines, "-" * 60)
    emit(lines, answer)
    emit(lines)


def build_debug_metadata(result: dict) -> dict:
    return {
        "attempts": result["attempts"],
        "used_fallback": result["used_fallback"],
        "attempt_1_reasons": result["first_verdict"]["reasons"],
        "attempt_2_reasons": (
            result["retry_verdict"]["reasons"] if result["retry_verdict"] else []
        ),
        "elapsed_s": round(result["_elapsed_s"], 2),
    }


def main() -> None:
    args = parse_args()
    guardrail.check_ollama()

    guardrail.GENERATOR_MODEL = args.model
    chat_app.BASE_MODEL = args.base_model
    chat_app.BASE_MODEL_FALLBACK = args.base_model_fallback

    lines: list[str] = []
    total = len(SMOKE_PROMPTS)

    for i, prompt in enumerate(SMOKE_PROMPTS, start=1):
        prefix = f"[{i}/{total}]"
        started = time.perf_counter()
        result = guardrail.run_with_retry(prompt)
        result["base_answer"] = chat_app.generate_base_answer(prompt)
        result["_elapsed_s"] = time.perf_counter() - started

        print_answer_block(lines, f"{prefix} FINE-TUNED MODEL (raw)", result["first_answer"])
        print_answer_block(
            lines,
            f"{prefix} FINE-TUNED MODEL (guardrailed final)",
            result["final_answer"],
        )
        print_answer_block(lines, f"{prefix} BASE MODEL", result["base_answer"])

        emit(
            lines,
            "=" * 60,
        )
        emit(
            lines,
            (
                f"{prefix} status: attempts={result['attempts']}, "
                f"fallback={result['used_fallback']}, "
                f"elapsed={result['_elapsed_s']:.2f}s"
            ),
        )
        emit(lines)
        emit(lines, "--- Guardrail Metadata ---")
        emit(lines, json.dumps(build_debug_metadata(result), ensure_ascii=False, indent=2))
        emit(lines)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Saved debug comparison: {output_path}")


if __name__ == "__main__":
    main()
