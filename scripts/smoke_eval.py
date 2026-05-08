#!/usr/bin/env python3
import argparse
import json
import pathlib
import subprocess
from typing import Any

from score_eval import has_artifact, has_loop, likely_hallucination


SMOKE_PROMPTS = [
    "Do you have public pricing for Synapse Tech products?",
    "Which integrations are explicitly supported by Synapse Tech?",
    "Can you guarantee a fixed percentage ROI in 3 months?",
    "What should a retail support team deploy first, and which KPI should they track first?",
    "If a feature is not confirmed in available information, should you guess or state it is unknown?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 5-question smoke eval against an Ollama model.")
    parser.add_argument("--model", required=True, help="Ollama model tag to evaluate.")
    parser.add_argument("--output", required=True, help="Path to save smoke eval JSONL.")
    return parser.parse_args()


def evaluate_response(text: str) -> dict[str, bool]:
    return {
        "artifact": has_artifact(text),
        "loop": has_loop(text),
        "hallucination_flag": likely_hallucination(text),
    }


def main() -> None:
    args = parse_args()
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for i, prompt in enumerate(SMOKE_PROMPTS, 1):
        result = subprocess.run(
            ["ollama", "run", args.model, prompt],
            capture_output=True,
            text=True,
        )
        response = (result.stdout or "").strip()
        checks = evaluate_response(response)
        rows.append(
            {
                "id": i,
                "prompt": prompt,
                "response": response,
                **checks,
            }
        )

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    bad = sum(1 for r in rows if r["artifact"] or r["loop"] or r["hallucination_flag"])
    print("\n=== Smoke Q/A ===")
    for row in rows:
        print(f"\nQ{row['id']}: {row['prompt']}")
        print(f"A{row['id']}: {row['response']}")
        print(
            "flags: "
            f"artifact={row['artifact']}, "
            f"loop={row['loop']}, "
            f"hallucination_flag={row['hallucination_flag']}"
        )
    print("\n=== Smoke Summary ===")
    print(f"smoke_file: {out_path}")
    print(f"smoke_total: {len(rows)}")
    print(f"smoke_flagged: {bad}")
    print(f"smoke_clean: {len(rows) - bad}")


if __name__ == "__main__":
    main()
