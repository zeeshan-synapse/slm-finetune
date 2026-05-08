#!/usr/bin/env python3
import argparse
import json
import pathlib
import re


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build corrective JSONL pairs from guardrail failure logs."
    )
    parser.add_argument(
        "--input",
        default="logs/guardrail_failures.jsonl",
        help="Path to guardrail failure log JSONL.",
    )
    parser.add_argument(
        "--output",
        default="data/corrective_pairs_from_logs.jsonl",
        help="Path to output corrective JSONL.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to output instead of overwrite.",
    )
    return parser.parse_args()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def first_two_sentences(text: str) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return ""
    return " ".join(sentences[:2]).strip()


def canonical_answer(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ["pricing", "price", "cost", "quote"]):
        return (
            "Pricing details are not confirmed in available information. "
            "Please request an official quote from Synapse Tech."
        )
    if "integration" in q:
        return (
            "The integration list is not explicitly confirmed in available information. "
            "Please verify supported integrations with Synapse Tech."
        )
    if "roi" in q or ("return" in q and "investment" in q):
        return (
            "No fixed ROI percentage can be guaranteed. "
            "Outcomes depend on implementation scope and execution."
        )
    if any(k in q for k in ["sla", "certification", "compliance", "roadmap"]):
        return (
            "That detail is not confirmed in available information. "
            "Please verify through official Synapse Tech channels."
        )
    return (
        "This detail is not confirmed in available information. "
        "Please verify with Synapse Tech."
    )


def normalize_pair(question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": question.strip()},
            {"role": "assistant", "content": answer.strip()},
        ]
    }


def dedupe_pairs(pairs: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for pair in pairs:
        user = pair["messages"][0]["content"].strip().lower()
        assistant = pair["messages"][1]["content"].strip().lower()
        key = (user, assistant)
        if key in seen:
            continue
        seen.add(key)
        out.append(pair)
    return out


def main() -> None:
    args = parse_args()
    in_path = pathlib.Path(args.input)
    out_path = pathlib.Path(args.output)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input log file: {in_path}")

    rows = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    pairs: list[dict] = []
    for row in rows:
        question = str(row.get("question", "")).strip()
        if not question:
            continue

        retry_answer = str(row.get("retry_answer") or "").strip()
        used_fallback = bool(row.get("used_fallback", False))

        if retry_answer and not used_fallback:
            candidate = first_two_sentences(retry_answer)
        else:
            candidate = ""

        if not candidate:
            candidate = canonical_answer(question)

        # Guard against artifacts in generated training targets.
        low = candidate.lower()
        if any(x in low for x in ["answer:", "according to the article", "list the names"]):
            candidate = canonical_answer(question)

        pairs.append(normalize_pair(question, candidate))

    pairs = dedupe_pairs(pairs)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.append else "w"
    with out_path.open(mode, encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"input_rows: {len(rows)}")
    print(f"output_pairs: {len(pairs)}")
    print(f"saved_to: {out_path}")


if __name__ == "__main__":
    main()
