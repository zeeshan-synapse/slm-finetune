#!/usr/bin/env python3
import argparse
import json
import pathlib
import re
from collections import Counter


def has_artifact(text: str) -> bool:
    patterns = [
        r"answer according to",
        r"sentence:",
        r"according to the article",
        r"table below",
    ]
    low = text.lower()
    return any(re.search(p, low) for p in patterns)


def has_loop(text: str) -> bool:
    norm = " ".join(text.lower().split())
    if not norm:
        return False
    chunks = re.split(r"[?.!]\s+", norm)
    chunks = [c.strip() for c in chunks if c.strip()]
    counts = Counter(chunks)
    return any(v >= 2 for v in counts.values())


def likely_hallucination(text: str) -> bool:
    low = text.lower()
    triggers = [
        "yes!",
        "subscription basis",
        "flexible payment",
        "guarantee",
        "certified",
    ]
    return any(t in low for t in triggers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick heuristic scorer for eval JSONL.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to eval results JSONL file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = pathlib.Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"Missing eval file: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        raise ValueError("No rows found in eval file.")

    total = len(rows)
    artifact = sum(1 for r in rows if has_artifact(r.get("response", "")))
    loop = sum(1 for r in rows if has_loop(r.get("response", "")))
    halluc = sum(1 for r in rows if likely_hallucination(r.get("response", "")))

    clean = sum(
        1
        for r in rows
        if not has_artifact(r.get("response", ""))
        and not has_loop(r.get("response", ""))
        and not likely_hallucination(r.get("response", ""))
    )

    def pct(n: int) -> str:
        return f"{(100.0 * n / total):.1f}%"

    print(f"file: {path}")
    print(f"total: {total}")
    print(f"artifact_count: {artifact} ({pct(artifact)})")
    print(f"loop_count: {loop} ({pct(loop)})")
    print(f"hallucination_flag_count: {halluc} ({pct(halluc)})")
    print(f"clean_count: {clean} ({pct(clean)})")


if __name__ == "__main__":
    main()
