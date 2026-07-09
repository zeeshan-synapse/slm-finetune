#!/usr/bin/env python3
import argparse
import json
import pathlib
import re
from collections import Counter
from typing import Any

from kb_paths import knowledge_base_dir

DEFAULT_KB_DIR = knowledge_base_dir()
DEFAULT_INPUT_PATH = DEFAULT_KB_DIR / "source_docs.jsonl"
DEFAULT_OUTPUT_PATH = DEFAULT_KB_DIR / "chunks.jsonl"

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build chunked KB JSONL rows from page-level source docs."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Path to the page-level source_docs.jsonl file.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to write chunked JSONL rows.",
    )
    parser.add_argument(
        "--target-words",
        type=int,
        default=180,
        help="Preferred chunk size in words before starting a new chunk.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=240,
        help="Hard ceiling for a chunk before forcing a split.",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=80,
        help="Preferred minimum chunk size. Short final tails are merged when possible.",
    )
    parser.add_argument(
        "--overlap-words",
        type=int,
        default=30,
        help="Approximate word overlap to preserve between neighboring chunks.",
    )
    parser.add_argument(
        "--include-page-type",
        action="append",
        default=[],
        help="Optional page_type filter. Repeat to allow multiple types.",
    )
    parser.add_argument(
        "--exclude-page-type",
        action="append",
        default=[],
        help="Optional page_type exclusion. Repeat to block multiple types.",
    )
    return parser.parse_args()


def load_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_line(text: str) -> str:
    return " ".join(text.split()).strip()


def word_count(text: str) -> int:
    return len(text.split())


def looks_like_heading(text: str) -> bool:
    words = text.split()
    if not words or EMAIL_RE.match(text):
        return False
    if len(words) > 12:
        return False
    if text.endswith(":") and len(words) <= 8:
        return True
    if text.endswith((".", "?", "!")):
        return False

    alpha_words = [word for word in words if any(ch.isalpha() for ch in word)]
    if not alpha_words:
        return False

    titleish_count = sum(word[:1].isupper() for word in alpha_words)
    titleish_ratio = titleish_count / len(alpha_words)
    return titleish_ratio >= 0.65


def split_doc_into_units(clean_text: str) -> list[str]:
    raw_units = [normalize_line(line) for line in clean_text.splitlines()]
    raw_units = [line for line in raw_units if line]

    merged_units: list[str] = []
    pending_headings: list[str] = []

    for unit in raw_units:
        if looks_like_heading(unit):
            pending_headings.append(unit)
            continue

        if pending_headings:
            merged_units.append("\n".join(pending_headings + [unit]))
            pending_headings = []
        else:
            merged_units.append(unit)

    if pending_headings:
        if merged_units:
            merged_units[-1] = merged_units[-1] + "\n" + "\n".join(pending_headings)
        else:
            merged_units.extend(pending_headings)

    return merged_units


def split_oversized_unit(unit: str, max_words: int) -> list[str]:
    if word_count(unit) <= max_words:
        return [unit]

    parts = re.split(r"(?<=[.!?])\s+", unit)
    parts = [normalize_line(part) for part in parts if normalize_line(part)]

    if len(parts) <= 1:
        words = unit.split()
        return [
            " ".join(words[start : start + max_words]).strip()
            for start in range(0, len(words), max_words)
            if " ".join(words[start : start + max_words]).strip()
        ]

    expanded: list[str] = []
    current_parts: list[str] = []

    for part in parts:
        candidate_parts = current_parts + [part]
        candidate_text = " ".join(candidate_parts).strip()
        if current_parts and word_count(candidate_text) > max_words:
            expanded.append(" ".join(current_parts).strip())
            current_parts = [part]
            continue
        current_parts = candidate_parts

    if current_parts:
        expanded.append(" ".join(current_parts).strip())

    final_units: list[str] = []
    for piece in expanded:
        if word_count(piece) <= max_words:
            final_units.append(piece)
            continue
        words = piece.split()
        final_units.extend(
            " ".join(words[start : start + max_words]).strip()
            for start in range(0, len(words), max_words)
            if " ".join(words[start : start + max_words]).strip()
        )

    return final_units


def expand_oversized_units(units: list[str], max_words: int) -> list[str]:
    expanded: list[str] = []
    for unit in units:
        expanded.extend(split_oversized_unit(unit, max_words))
    return expanded


def overlap_units(units: list[str], overlap_words: int) -> list[str]:
    if overlap_words <= 0:
        return []

    selected: list[str] = []
    total_words = 0

    for unit in reversed(units):
        selected.append(unit)
        total_words += word_count(unit)
        if total_words >= overlap_words:
            break

    return list(reversed(selected))


def drop_shared_prefix(previous_units: list[str], current_units: list[str]) -> list[str]:
    max_shared = min(len(previous_units), len(current_units))

    for size in range(max_shared, 0, -1):
        if previous_units[-size:] == current_units[:size]:
            return current_units[size:]

    return current_units


def should_emit_chunk(
    current_units: list[str],
    next_unit: str,
    target_words: int,
    max_words: int,
    min_words: int,
) -> bool:
    if not current_units:
        return False

    current_words = sum(word_count(unit) for unit in current_units)
    projected_words = current_words + word_count(next_unit)

    if projected_words > max_words and current_words >= min_words:
        return True

    return current_words >= target_words and projected_words > target_words + 20


def build_chunk_row(doc: dict[str, Any], chunk_index: int, units: list[str]) -> dict[str, Any]:
    text = "\n\n".join(units).strip()
    return {
        "chunk_id": f"{doc['doc_id']}::chunk-{chunk_index:03d}",
        "doc_id": doc["doc_id"],
        "chunk_index": chunk_index,
        "source_file": doc["source_file"],
        "title": doc["title"],
        "url": doc["url"],
        "page_type": doc["page_type"],
        "word_count": word_count(text),
        "text": text,
    }


def chunk_doc(
    doc: dict[str, Any],
    target_words: int,
    max_words: int,
    min_words: int,
    overlap_words: int,
) -> list[dict[str, Any]]:
    units = split_doc_into_units(doc.get("clean_text", ""))
    units = expand_oversized_units(units, max_words)
    if not units:
        return []

    chunks: list[dict[str, Any]] = []
    current_units: list[str] = []

    for unit in units:
        if should_emit_chunk(current_units, unit, target_words, max_words, min_words):
            chunks.append({"_units": current_units[:]})
            current_units = overlap_units(current_units, overlap_words)

        current_units.append(unit)

    if current_units:
        if chunks and sum(word_count(unit) for unit in current_units) < min_words:
            previous_units = chunks[-1]["_units"]
            extra_units = drop_shared_prefix(previous_units, current_units)
            if extra_units:
                previous_units.extend(extra_units)
        else:
            chunks.append({"_units": current_units[:]})

    final_rows = []
    for idx, chunk in enumerate(chunks, start=1):
        final_rows.append(build_chunk_row(doc, idx, chunk["_units"]))
    return final_rows


def page_type_allowed(
    page_type: str,
    include_page_types: set[str],
    exclude_page_types: set[str],
) -> bool:
    if include_page_types and page_type not in include_page_types:
        return False
    if page_type in exclude_page_types:
        return False
    return True


def main() -> None:
    args = parse_args()

    if args.max_words <= 0 or args.target_words <= 0 or args.min_words <= 0:
        raise ValueError("Chunk size arguments must be positive integers.")
    if args.min_words > args.max_words:
        raise ValueError("--min-words cannot be greater than --max-words.")
    if args.target_words > args.max_words:
        raise ValueError("--target-words cannot be greater than --max-words.")

    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    include_page_types = set(args.include_page_type)
    exclude_page_types = set(args.exclude_page_type)

    docs = load_rows(input_path)
    selected_docs = [
        doc
        for doc in docs
        if page_type_allowed(doc.get("page_type", ""), include_page_types, exclude_page_types)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_rows: list[dict[str, Any]] = []
    skipped_docs = []
    chunk_counts = Counter()

    for doc in selected_docs:
        doc_chunks = chunk_doc(
            doc,
            target_words=args.target_words,
            max_words=args.max_words,
            min_words=args.min_words,
            overlap_words=args.overlap_words,
        )
        if not doc_chunks:
            skipped_docs.append(doc["doc_id"])
            continue

        chunk_rows.extend(doc_chunks)
        chunk_counts[doc["page_type"]] += len(doc_chunks)

    with output_path.open("w", encoding="utf-8") as f:
        for row in chunk_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Input docs: {len(docs)}")
    print(f"Selected docs: {len(selected_docs)}")
    print(f"Written chunks: {len(chunk_rows)}")
    print(f"Skipped docs with no chunks: {len(skipped_docs)}")
    if include_page_types:
        print("Included page types:", ", ".join(sorted(include_page_types)))
    if exclude_page_types:
        print("Excluded page types:", ", ".join(sorted(exclude_page_types)))
    print("Chunk counts by page type:", dict(sorted(chunk_counts.items())))
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
